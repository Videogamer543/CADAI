"""
Onshape read-only client (accuracy upgrade: real CAD geometry, not silhouette).

Two ways to authenticate, and which one is right depends entirely on who can
reach the server:

  * **OAuth bearer token** (app/onshape_oauth.py) -- the only correct choice for
    anything public. Each visitor authorises their own Onshape account and the
    token StressViz holds can read exactly that person's documents. Pass one in
    as `token=` and it is used.
  * **API key + secret** (HMAC, below) -- for development on your own machine.
    Every request made this way reads *your* documents under *your* name, which
    is fine when you are the only one who can open the app and unacceptable the
    moment anyone else can.

Passing no token falls back to the key pair, so local work is unchanged by the
existence of the OAuth path. Credentials come from environment variables, never
hardcoded.

Env:
  ONSHAPE_ACCESS_KEY   your API access key   (read-documents scope only)
  ONSHAPE_SECRET_KEY   your API secret key
"""
from __future__ import annotations
import os
import base64
import datetime
import hashlib
import hmac
import re
import urllib.parse
import requests

BASE = "https://cad.onshape.com"


def _auth_headers(method: str, path: str, query: str = ""):
    ak = os.environ.get("ONSHAPE_ACCESS_KEY", "")
    sk = os.environ.get("ONSHAPE_SECRET_KEY", "")
    if not ak or not sk:
        raise RuntimeError("ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY not set")
    nonce = base64.b64encode(os.urandom(24)).decode()[:25]
    date = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    ctype = "application/json"
    to_sign = (method + "\n" + nonce + "\n" + date + "\n" + ctype + "\n" +
               path + "\n" + query + "\n").lower()
    sig = base64.b64encode(
        hmac.new(sk.encode(), to_sign.encode(), hashlib.sha256).digest()
    ).decode()
    return {
        "Date": date,
        "On-Nonce": nonce,
        "Authorization": f"On {ak}:HmacSHA256:{sig}",
        "Content-Type": ctype,
        "Accept": "application/json",
    }


def _get(path: str, params: dict | None = None, token: str | None = None):
    query = urllib.parse.urlencode(params or {})
    if token:
        headers = {"Authorization": "Bearer " + token,
                   "Content-Type": "application/json",
                   "Accept": "application/json"}
    else:
        headers = _auth_headers("GET", path, query)
    url = BASE + path + (("?" + query) if query else "")
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def get_mass_properties(did, wid, eid, wvm="w", token=None):
    """Mass, centroid, volume for a Part Studio element (read-only)."""
    return _get(f"/api/parts/d/{did}/{wvm}/{wid}/e/{eid}/massproperties",
                token=token)


def get_tessellated_faces(did, wid, eid, wvm="w", chord_tol=0.001, angle_tol=0.2,
                          token=None):
    """Triangle tessellation of all faces — the true 3D geometry."""
    return _get(
        f"/api/partstudios/d/{did}/{wvm}/{wid}/e/{eid}/tessellatedfaces",
        {"outputVertexNormals": "false", "chordTolerance": chord_tol,
         "angleTolerance": angle_tol},
        token=token,
    )


# The address bar is the only place a user can get these three ids without
# reading API documentation, so parsing it is the difference between "paste the
# link to your part" and "find your element id". The middle segment is w, v or
# m -- workspace, version or microversion -- and it has to be carried through
# rather than assumed to be "w", or a link to a released version 404s.
_DOC_RE = re.compile(
    r"/documents/(?P<did>[0-9a-f]{24})"
    r"(?:/(?P<wvm>[wvm])/(?P<wid>[0-9a-f]{24}))?"
    r"(?:/e/(?P<eid>[0-9a-f]{24}))?",
    re.I)


def parse_document_url(url: str):
    """Pull document / workspace / element ids out of a cad.onshape.com URL.

    Returns a dict with did, wvm, wid, eid -- any of which may be None -- or
    None if this is not an Onshape document URL at all.
    """
    if not url:
        return None
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return None
    if parts.netloc and "onshape.com" not in parts.netloc.lower():
        return None
    m = _DOC_RE.search(parts.path or "")
    if not m:
        return None
    d = m.groupdict()
    return {"did": d["did"], "wvm": (d["wvm"] or "w").lower(),
            "wid": d["wid"], "eid": d["eid"]}
