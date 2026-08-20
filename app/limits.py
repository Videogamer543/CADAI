"""What stands between a public StressViz and somebody else's afternoon.

On a laptop this file does nothing. Every limit here is off unless it is turned
on in .env, because the failure mode of a limit on a single-user machine is
"the tool I own refuses to work for me", which is worse than the thing the
limit prevents. Set STRESSVIZ_PUBLIC=1 and the defaults below switch on
together; each one can also be set individually.

Four separate things are being defended, and they are not the same thing:

  the bill      Every chat question is one Groq call plus up to four Tavily
                searches at advanced depth. Nothing about the endpoint tells a
                stranger to stop, so the ceiling has to. Per-IP rate limits stop
                one person hammering it; the global daily count is the backstop
                for a hundred people doing it politely at once, which no per-IP
                limit catches.

  the machine   A tetrahedral solve is tens of seconds of pure CPU and a good
                deal of RAM. Two at once is a busy server; ten at once is a dead
                one, and the tenth request is what kills the nine already
                running. Better to refuse the tenth immediately and say so.

  the disk      An UploadFile is spooled to a temporary file, so a 2 GB "STEP"
                is 2 GB of disk before a single line of our code runs. The
                Content-Length check happens in middleware, before the multipart
                parser is allowed to touch the body.

  the door      Optionally, a password. HTTP Basic rather than a login page:
                no cookie handling, no session store, no new UI, and every
                browser and curl already speaks it. It is only as private as
                the connection carrying it, which is why it insists on HTTPS
                being terminated somewhere in front.

No dependencies beyond the standard library. A rate limiter that needs Redis is
a rate limiter that is not running.
"""
from __future__ import annotations

import base64
import hmac
import os
import threading
import time


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def _flag(name, default=False):
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v not in ("0", "false", "no", "off")


def _num(name, default):
    """A malformed number falls back to the default rather than refusing to
    start. A typo in .env should not take a running server down."""
    try:
        v = (os.environ.get(name) or "").strip()
        return type(default)(v) if v else default
    except (TypeError, ValueError):
        return default


PUBLIC = _flag("STRESSVIZ_PUBLIC")

# Bytes. 12 MB is a generous STEP file; the plate drawings this was built for
# are a few hundred kilobytes.
MAX_UPLOAD_MB = _num("STRESSVIZ_MAX_UPLOAD_MB", 12.0)

# Per-IP, per-window. Two windows each, because a minute limit alone lets
# someone run at the limit all day, and an hour limit alone lets them spend the
# whole hour's allowance in four seconds.
CHAT_PER_MIN = _num("STRESSVIZ_CHAT_PER_MIN", 8 if PUBLIC else 0)
CHAT_PER_HOUR = _num("STRESSVIZ_CHAT_PER_HOUR", 60 if PUBLIC else 0)
SOLVE_PER_MIN = _num("STRESSVIZ_SOLVE_PER_MIN", 6 if PUBLIC else 0)
SOLVE_PER_HOUR = _num("STRESSVIZ_SOLVE_PER_HOUR", 60 if PUBLIC else 0)

# Everybody's questions, added up, per UTC day. This is the number that maps to
# money, so it is the one worth setting deliberately.
CHAT_PER_DAY = _num("STRESSVIZ_CHAT_PER_DAY", 400 if PUBLIC else 0)

# Everybody's analyses, added up, per UTC day. The same backstop argument as
# CHAT_PER_DAY, for a different bill: chat costs Groq and Tavily money, solves
# cost the *host* money, because a solve is the only thing here that is
# genuinely expensive in CPU. Per-IP limits cannot see a hundred well-behaved
# strangers; this can.
SOLVE_PER_DAY = _num("STRESSVIZ_SOLVE_PER_DAY", 200 if PUBLIC else 0)

# The one that actually maps to a hosting bill: seconds of solving, everyone
# added together, per UTC day. Counting solves is a proxy -- a plate outline is
# two seconds and a tetrahedral mesh is ninety -- and a proxy is the wrong thing
# to bill against. Cloud Run's free tier is 180,000 vCPU-seconds a month, so an
# hour a day is about 111,000 a month at one vCPU, leaving comfortable headroom
# for chat requests and cold starts. Raise it if you are paying and want more.
COMPUTE_SECONDS_PER_DAY = _num("STRESSVIZ_COMPUTE_SECONDS_PER_DAY",
                               3600.0 if PUBLIC else 0.0)

# Solves running at once, across everyone.
SOLVE_CONCURRENT = _num("STRESSVIZ_SOLVE_CONCURRENT", 2 if PUBLIC else 0)

PASSWORD = (os.environ.get("STRESSVIZ_PASSWORD") or "").strip()
USERNAME = (os.environ.get("STRESSVIZ_USERNAME") or "stressviz").strip()

# Only trust X-Forwarded-For when something you control is setting it. Anyone
# can send that header; behind no proxy it is a free pass around every per-IP
# limit here, so it is ignored by default.
TRUST_PROXY = _flag("STRESSVIZ_TRUST_PROXY")

# Path prefixes, longest first so /api/chat/stream is classified before
# /api/chat would swallow it. (Both are "chat", but the ordering habit matters
# the moment a prefix is added that is not a superset.)
CHAT_PATHS = ("/api/chat/stream", "/api/chat")
SOLVE_PATHS = ("/api/analyze3d", "/api/analyze", "/api/tessellate")


def enabled():
    """True when anything here is actually doing something."""
    return bool(PASSWORD or CHAT_PER_MIN or CHAT_PER_HOUR or CHAT_PER_DAY
                or SOLVE_PER_MIN or SOLVE_PER_HOUR or SOLVE_PER_DAY
                or COMPUTE_SECONDS_PER_DAY or SOLVE_CONCURRENT)


def kind_of(path):
    for p in CHAT_PATHS:
        if path.startswith(p):
            return "chat"
    for p in SOLVE_PATHS:
        if path.startswith(p):
            return "solve"
    return None


def client_ip(request):
    fwd = request.headers.get("x-forwarded-for") if TRUST_PROXY else None
    if fwd:
        # The left-most entry is the original client; everything after it is the
        # chain of proxies. Take the first and stop.
        first = fwd.split(",")[0].strip()
        if first:
            return first
    return (request.client.host if request.client else "?") or "?"


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class _Windows:
    """Per-key sliding windows, counted by timestamp.

    Timestamps rather than a token bucket because the numbers here are small
    (tens per hour) and a list of times answers "when does this free up" exactly
    -- which is what the 429 needs to say. A bucket would only be able to
    estimate it.

    Memory is bounded by pruning on read, plus a sweep of idle keys when the
    table grows: a public instance sees a long tail of one-request addresses and
    without the sweep every one of them would be remembered forever.
    """

    def __init__(self):
        self._hits = {}
        self._lock = threading.Lock()
        self._swept = 0.0

    def _sweep(self, now, horizon):
        if now - self._swept < 600:
            return
        self._swept = now
        for k in [k for k, v in self._hits.items()
                  if not v or now - v[-1] > horizon]:
            self._hits.pop(k, None)

    def check(self, key, limits):
        """limits: [(window_seconds, max_in_window), ...] -- returns 0 if the
        request is allowed, otherwise whole seconds until it would be."""
        limits = [(w, n) for w, n in limits if n and n > 0]
        if not limits:
            return 0
        horizon = max(w for w, _ in limits)
        now = time.time()
        with self._lock:
            self._sweep(now, horizon)
            times = [t for t in self._hits.get(key, ()) if now - t < horizon]
            for window, cap in limits:
                recent = [t for t in times if now - t < window]
                if len(recent) >= cap:
                    # The oldest hit inside this window is the one whose
                    # expiry frees a slot.
                    return max(1, int(window - (now - recent[0])) + 1)
            times.append(now)
            self._hits[key] = times
            return 0


_win = _Windows()


class _DailyCount:
    """One global counter that resets at UTC midnight.

    Deliberately not persisted. A restart clearing the count is the right
    trade: the alternative is a state file that can go stale, be lost with the
    container, or lock the owner out of their own tool after a crash. The
    per-IP limits are the ones that hold continuously; this is a backstop.
    """

    def __init__(self):
        self.day = None
        self.n = 0
        self._lock = threading.Lock()

    def take(self, cap):
        if not cap or cap <= 0:
            return True
        today = time.gmtime().tm_yday
        with self._lock:
            if self.day != today:
                self.day, self.n = today, 0
            if self.n >= cap:
                return False
            self.n += 1
            return True

    def used(self):
        return self.n if self.day == time.gmtime().tm_yday else 0


_daily = _DailyCount()
_solve_daily = _DailyCount()


class _DailySeconds:
    """A budget spent in seconds of work, not in requests.

    Counting requests is the obvious thing and the wrong thing: a plate outline
    is two seconds of CPU and a tetrahedral mesh is ninety, so a "200 solves a
    day" cap is somewhere between six minutes and five hours of machine time
    depending on what people happen to upload. This counts the thing the host
    actually bills for.

    Charged on the way out, once the elapsed time is known, which means the
    budget can be overrun by at most one solve. Refusing on the way in would
    need an estimate of how long a solve will take before meshing it, and a
    wrong estimate is worse than a bounded overshoot.
    """

    def __init__(self):
        self.day = None
        self.spent = 0.0
        self._lock = threading.Lock()

    def _roll(self):
        today = time.gmtime().tm_yday
        if self.day != today:
            self.day, self.spent = today, 0.0

    def exhausted(self, cap):
        if not cap or cap <= 0:
            return False
        with self._lock:
            self._roll()
            return self.spent >= cap

    def add(self, seconds):
        with self._lock:
            self._roll()
            self.spent += max(0.0, float(seconds))

    def used(self):
        return round(self.spent, 1) if self.day == time.gmtime().tm_yday else 0.0


_compute = _DailySeconds()


def check_rate(kind, ip):
    """None if allowed, else (status, message) ready to become a response."""
    if kind == "chat":
        wait = _win.check("c:" + ip, [(60, CHAT_PER_MIN), (3600, CHAT_PER_HOUR)])
        if wait:
            return (429, "Too many questions from this address. Try again in "
                         "%d second%s." % (wait, "" if wait == 1 else "s"))
        if not _daily.take(CHAT_PER_DAY):
            return (429, "The assistant has answered its daily maximum of %d "
                         "questions. It resets at midnight UTC. Everything else "
                         "in CADAI still works." % CHAT_PER_DAY)
    elif kind == "solve":
        wait = _win.check("s:" + ip, [(60, SOLVE_PER_MIN), (3600, SOLVE_PER_HOUR)])
        if wait:
            return (429, "Too many analyses from this address. Try again in "
                         "%d second%s." % (wait, "" if wait == 1 else "s"))
        # Checked before the count is taken, so a day that has run out of
        # machine time does not also silently eat the day's solve allowance.
        if _compute.exhausted(COMPUTE_SECONDS_PER_DAY):
            return (429, "CADAI has used its daily allowance of analysis "
                         "time. It resets at midnight UTC. The assistant and "
                         "everything already on screen still work.")
        if not _solve_daily.take(SOLVE_PER_DAY):
            return (429, "CADAI has run its daily maximum of %d analyses. "
                         "It resets at midnight UTC. The assistant still "
                         "works." % SOLVE_PER_DAY)
    return None


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------
class _Slots:
    """Refuse immediately rather than queue.

    A queue would make the eleventh visitor wait four minutes behind ten solves
    and then very likely find their browser had given up. A 503 that says "two
    are running, try again shortly" is a worse-sounding answer and a better
    experience.
    """

    def __init__(self):
        self.n = 0
        self._lock = threading.Lock()

    def take(self, cap):
        if not cap or cap <= 0:
            return True
        with self._lock:
            if self.n >= cap:
                return False
            self.n += 1
            return True

    def free(self, cap):
        if not cap or cap <= 0:
            return
        with self._lock:
            self.n = max(0, self.n - 1)

    def busy(self):
        return self.n


_slots = _Slots()


def take_slot():
    return _slots.take(SOLVE_CONCURRENT)


def free_slot():
    _slots.free(SOLVE_CONCURRENT)


def record_solve_seconds(seconds):
    """Charge a finished solve against the day's machine-time budget.

    Wall time, not CPU time: with SOLVE_CONCURRENT at 2 the two are close, and
    wall time is what a host with one instance is actually billed for.
    """
    if COMPUTE_SECONDS_PER_DAY and COMPUTE_SECONDS_PER_DAY > 0:
        _compute.add(seconds)


# --------------------------------------------------------------------------
# The door
# --------------------------------------------------------------------------
def password_ok(request):
    """True when no password is set, or the Basic credentials match.

    hmac.compare_digest, not ==, so the comparison does not return faster on a
    wrong first character. That is close to paranoia over a network, and it
    costs one function call.
    """
    if not PASSWORD:
        return True
    hdr = request.headers.get("authorization") or ""
    if not hdr.lower().startswith("basic "):
        return False
    try:
        raw = base64.b64decode(hdr[6:].strip()).decode("utf-8", "replace")
    except Exception:
        return False
    user, _, pwd = raw.partition(":")
    return (hmac.compare_digest(user, USERNAME)
            and hmac.compare_digest(pwd, PASSWORD))


# --------------------------------------------------------------------------
# Upload size
# --------------------------------------------------------------------------
def max_upload_bytes():
    return int(MAX_UPLOAD_MB * 1024 * 1024)


def declared_too_big(request):
    """The cheap check: the client's own Content-Length, read before the body is.

    Content-Length can lie, which is why read_capped exists as well -- but when
    it is honest this rejects a huge upload before a byte of it has been parsed
    into a temporary file.
    """
    if MAX_UPLOAD_MB <= 0:
        return False
    try:
        n = int(request.headers.get("content-length") or 0)
    except ValueError:
        return False
    return n > max_upload_bytes()


async def read_capped(upload):
    """The whole uploaded file, or a 413 -- never more than the cap in memory.

    Read in chunks and stop at the limit, instead of `await upload.read()`,
    which is happy to hand back however many gigabytes arrived. The
    Content-Length check in middleware catches almost all of these first; this
    is what catches a chunked upload that never declared a length at all.
    """
    from fastapi import HTTPException
    cap = max_upload_bytes()
    if MAX_UPLOAD_MB <= 0:
        return await upload.read()
    out, total = [], 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            raise HTTPException(
                413, "That file is larger than the %g MB limit. A STEP of a "
                     "plate is usually well under a megabyte -- if this is an "
                     "assembly, export just the part you want analysed."
                     % MAX_UPLOAD_MB)
        out.append(chunk)
    return b"".join(out)


# --------------------------------------------------------------------------
# For /api/health
# --------------------------------------------------------------------------
def status():
    return {
        "public": bool(PUBLIC),
        "password": bool(PASSWORD),
        "max_upload_mb": MAX_UPLOAD_MB,
        "chat_per_min": CHAT_PER_MIN,
        "chat_per_hour": CHAT_PER_HOUR,
        "chat_per_day": CHAT_PER_DAY,
        "chat_used_today": _daily.used(),
        "solve_per_min": SOLVE_PER_MIN,
        "solve_per_hour": SOLVE_PER_HOUR,
        "solve_per_day": SOLVE_PER_DAY,
        "solves_used_today": _solve_daily.used(),
        "compute_seconds_per_day": COMPUTE_SECONDS_PER_DAY,
        "compute_seconds_used_today": _compute.used(),
        "solve_concurrent": SOLVE_CONCURRENT,
        "solves_running": _slots.busy(),
        "trust_proxy": bool(TRUST_PROXY),
    }
