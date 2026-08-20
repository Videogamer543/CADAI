"""
OAuth 2.0 against Onshape, so a public StressViz never carries a shared key.

The difference this module exists to make: with an API key, every visitor who
uploads an Onshape link is browsing *your* documents under *your* name, and a
single leaked key exposes your whole account. With OAuth each visitor authorises
their own Onshape account, StressViz gets a token scoped to that person, and the
worst case for any one token is that one person's read access.

Two stores, both in memory, both deliberately:

  * `_sessions` maps an opaque cookie value to that browser's tokens. Losing it
    on restart costs a visitor one click on "Connect Onshape". Persisting it
    would mean writing other people's live credentials to disk on a box you do
    not control, to save that click.
  * the CSRF `state` lives *inside* the session rather than in a global set.
    A state stored globally is only a nonce -- any browser could present it. Tied
    to the cookie, it proves the callback belongs to the browser that started
    the flow, which is the whole point of the parameter.

Env (see .env.example):
  ONSHAPE_OAUTH_CLIENT_ID      the on_... id from the developer portal
  ONSHAPE_OAUTH_CLIENT_SECRET  shown once, at creation
  ONSHAPE_OAUTH_REDIRECT       must match a registered Redirect URL character
                               for character, including scheme, port and path
  ONSHAPE_OAUTH_SCOPE          usually left unset -- see NOTE below
"""
from __future__ import annotations
import hmac
import os
import secrets
import threading
import time
import urllib.parse

import requests

AUTH_URL = "https://oauth.onshape.com/oauth/authorize"
TOKEN_URL = "https://oauth.onshape.com/oauth/token"

COOKIE = "sv_onshape"
STATE_TTL = 600          # ten minutes to finish a login is generous
SESSION_TTL = 7 * 86400  # untouched sessions are swept
MAX_SESSIONS = 5000      # a ceiling so a hostile visitor cannot grow the dict

# NOTE on scope: Onshape derives an application's permissions from the
# checkboxes ticked when it was created, not from a scope parameter -- both of
# Onshape's own reference integrations omit it entirely. This is left
# configurable only so a future change on their side does not need a code
# change; empty means "send no scope parameter", which is the correct default.
SCOPE = (os.environ.get("ONSHAPE_OAUTH_SCOPE") or "").strip()


class OAuthError(RuntimeError):
    """Something the visitor should be told about, in words they can act on."""


def client_id():
    return (os.environ.get("ONSHAPE_OAUTH_CLIENT_ID") or "").strip()


def _client_secret():
    return (os.environ.get("ONSHAPE_OAUTH_CLIENT_SECRET") or "").strip()


def redirect_uri():
    return (os.environ.get("ONSHAPE_OAUTH_REDIRECT") or "").strip()


def configured():
    """All three, because two of the three is a flow that fails halfway."""
    return bool(client_id() and _client_secret() and redirect_uri())


def secure_cookie():
    """A cookie marked Secure over plain http is a cookie the browser drops.

    Deciding from the registered redirect rather than from the request means a
    local http://localhost session still works while a deployed https one gets
    the flag, with no extra setting to forget.
    """
    return redirect_uri().lower().startswith("https://")


# --------------------------------------------------------------------------
# session store
# --------------------------------------------------------------------------

class _Sessions:
    def __init__(self):
        self._d = {}
        self._lock = threading.Lock()
        self._swept = 0.0

    def _sweep(self, now):
        if now - self._swept < 3600:
            return
        self._swept = now
        for k in [k for k, v in self._d.items()
                  if now - v.get("seen", 0) > SESSION_TTL]:
            self._d.pop(k, None)
        # If sweeping did not get us under the cap, drop the least recently
        # used. Refusing new sessions instead would lock out real visitors to
        # punish one; forgetting the stalest costs someone a reconnect.
        if len(self._d) > MAX_SESSIONS:
            for k, _ in sorted(self._d.items(),
                               key=lambda kv: kv[1].get("seen", 0)
                               )[:len(self._d) - MAX_SESSIONS]:
                self._d.pop(k, None)

    def new(self):
        sid = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._sweep(now)
            self._d[sid] = {"seen": now}
        return sid

    def get(self, sid):
        if not sid:
            return None
        with self._lock:
            s = self._d.get(sid)
            if s is not None:
                s["seen"] = time.time()
            return s

    def drop(self, sid):
        with self._lock:
            self._d.pop(sid, None)

    def count(self):
        return len(self._d)


_sessions = _Sessions()


# --------------------------------------------------------------------------
# the flow
# --------------------------------------------------------------------------

def begin(sid):
    """Start a login. Returns (authorize_url, sid) -- sid may be newly minted.

    The state is generated here and remembered against this browser's session,
    so `complete` can prove the callback it receives belongs to the browser
    that asked for it rather than to whoever guessed a URL.
    """
    if not configured():
        raise OAuthError(
            "Onshape is not configured on this server. It needs "
            "ONSHAPE_OAUTH_CLIENT_ID, ONSHAPE_OAUTH_CLIENT_SECRET and "
            "ONSHAPE_OAUTH_REDIRECT set.")

    sess = _sessions.get(sid)
    if sess is None:
        sid = _sessions.new()
        sess = _sessions.get(sid)

    state = secrets.token_urlsafe(24)
    sess["state"] = state
    sess["state_at"] = time.time()

    params = {
        "response_type": "code",
        "client_id": client_id(),
        "redirect_uri": redirect_uri(),
        "state": state,
    }
    if SCOPE:
        params["scope"] = SCOPE
    return AUTH_URL + "?" + urllib.parse.urlencode(params), sid


def _post_token(data):
    try:
        r = requests.post(TOKEN_URL, data=data, timeout=30, headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        })
    except requests.RequestException as e:
        raise OAuthError("Could not reach Onshape to finish signing in: %s" % e)
    if r.status_code >= 400:
        # Onshape's body is JSON when it can be and an HTML error page when it
        # cannot; either way the first line of it is more use to whoever is
        # reading the log than the status code alone.
        detail = (r.text or "").strip().splitlines()
        raise OAuthError("Onshape refused the token request (HTTP %d)%s"
                         % (r.status_code,
                            (": " + detail[0][:200]) if detail else ""))
    try:
        return r.json()
    except ValueError:
        raise OAuthError("Onshape returned something that was not a token.")


def _store(sess, tok):
    if not tok.get("access_token"):
        raise OAuthError("Onshape's reply contained no access token.")
    sess["access"] = tok["access_token"]
    # Onshape issues a fresh refresh token on every refresh and retires the old
    # one, so this must overwrite rather than keep the original -- reusing a
    # spent refresh token fails and logs the visitor out for no visible reason.
    if tok.get("refresh_token"):
        sess["refresh"] = tok["refresh_token"]
    try:
        ttl = int(tok.get("expires_in") or 0)
    except (TypeError, ValueError):
        ttl = 0
    sess["expires_at"] = time.time() + (ttl if ttl > 0 else 1800)
    sess.pop("state", None)
    sess.pop("state_at", None)


def complete(sid, code, state):
    """Finish a login. Raises OAuthError with something worth showing a user."""
    sess = _sessions.get(sid)
    if sess is None:
        raise OAuthError(
            "That sign-in did not start here, or it started so long ago the "
            "server forgot it. Press Connect Onshape again.")
    want = sess.get("state") or ""
    if (not want or not state or not hmac.compare_digest(want, state)
            or time.time() - sess.get("state_at", 0) > STATE_TTL):
        sess.pop("state", None)
        raise OAuthError("That sign-in link did not match the one this browser "
                         "started. Press Connect Onshape again.")
    if not code:
        raise OAuthError("Onshape did not send an authorisation code back.")

    _store(sess, _post_token({
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id(),
        "client_secret": _client_secret(),
        # Sent again deliberately: Onshape requires the redirect_uri in the
        # exchange as well as the authorize step, and compares the two.
        "redirect_uri": redirect_uri(),
    }))


def _refresh(sess):
    if not sess.get("refresh"):
        raise OAuthError("Your Onshape session expired. Connect again.")
    _store(sess, _post_token({
        "grant_type": "refresh_token",
        "refresh_token": sess["refresh"],
        "client_id": client_id(),
        "client_secret": _client_secret(),
    }))


def token_for(sid):
    """A usable access token for this browser, or None if it is not connected.

    Refreshes 60 seconds early rather than on expiry, because a token that is
    valid when checked and stale when it arrives is the intermittent failure
    nobody can reproduce.
    """
    sess = _sessions.get(sid)
    if sess is None or not sess.get("access"):
        return None
    if time.time() >= sess.get("expires_at", 0) - 60:
        try:
            _refresh(sess)
        except OAuthError:
            sess.pop("access", None)
            sess.pop("refresh", None)
            return None
    return sess.get("access")


def disconnect(sid):
    """Forget this browser's tokens.

    Onshape's own revocation still has to happen in the user's Onshape account
    settings; this only stops StressViz from holding the token. Saying so in
    the UI is more honest than implying a revoke we did not perform.
    """
    _sessions.drop(sid)


def status(sid):
    sess = _sessions.get(sid)
    connected = bool(sess and sess.get("access"))
    out = {
        "configured": configured(),
        "connected": connected,
        "redirect_uri": redirect_uri(),
        "sessions": _sessions.count(),
    }
    if connected:
        out["expires_in"] = max(0, int(sess.get("expires_at", 0) - time.time()))
    return out
