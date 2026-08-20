"""
StressViz FastAPI backend — full Python engine.

Endpoints:
  GET  /                 → frontend
  GET  /api/materials    → material library (names)
  POST /api/analyze      → image → FEM stress (+ optional pocketing), saved to disk
  POST /api/parse_load   → plain-English loading description → solver settings
  POST /api/chat         → engineering/FRC assistant (Groq + search + TBA)
  GET  /api/health

Run:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations
import os
import time

# Load .env BEFORE anything reads os.environ. Without this the keys sitting in
# .env were never visible to the process, so every chat call died with
# "GROQ_API_KEY not set" and the UI silently showed "(no answer)".
try:
    from dotenv import load_dotenv
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _ENV = os.path.join(_ROOT, ".env")
    if not os.path.exists(_ENV) and os.path.exists(_ENV + ".example"):
        import shutil
        shutil.copyfile(_ENV + ".example", _ENV)   # so there's a file to fill in
    load_dotenv(_ENV)
    load_dotenv()  # also honour a .env in the current working directory
    # Placeholder values from .env.example must not count as real keys.
    for _k in ("GROQ_API_KEY", "TAVILY_API_KEY", "TBA_API_KEY"):
        _v = (os.environ.get(_k) or "").strip()
        if not _v or _v.startswith("your_") or _v.endswith("_here"):
            os.environ.pop(_k, None)
except Exception:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import (JSONResponse, FileResponse, StreamingResponse,
                               RedirectResponse, HTMLResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, materials, loadctx, limits
from . import onshape as onshape_api
from . import onshape_oauth
from . import chat as chatmod
from .save import save_outputs

app = FastAPI(title="CADAI", version="2.0")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


@app.middleware("http")
async def guard(request, call_next):
    """Everything that has to happen before a request is allowed to cost money.

    One middleware rather than a decorator on each endpoint, because the checks
    that matter most -- the password, and the size of the body -- have to happen
    BEFORE the endpoint's signature is satisfied. By the time an `UploadFile`
    parameter has been filled in, the multipart parser has already written the
    whole upload to a temporary file, which is precisely the thing a size limit
    exists to prevent.

    On a laptop with nothing configured every branch below is skipped, and this
    costs one string comparison per request. See app/limits.py.
    """
    path = request.url.path

    # Health stays open even behind the password, so a host's uptime check does
    # not need credentials and a locked-out owner can still see the server is up.
    # It reports configuration, never secrets.
    if path != "/api/health" and not limits.password_ok(request):
        return JSONResponse(
            {"error": "Not authorised.",
             "answer": "This CADAI needs a password.",
             "detail": "This CADAI needs a password."},
            status_code=401,
            # Without this the browser has no idea it should ask for one, and
            # the visitor just sees a blank 401.
            headers={"WWW-Authenticate": 'Basic realm="CADAI"'},
        )

    kind = limits.kind_of(path)
    if kind is None:
        return await call_next(request)

    if kind == "solve" and limits.declared_too_big(request):
        return JSONResponse(
            {"error": "file too large",
             "detail": "That file is larger than the %g MB limit."
                       % limits.MAX_UPLOAD_MB},
            status_code=413)

    denied = limits.check_rate(kind, limits.client_ip(request))
    if denied:
        status, msg = denied
        # `answer` as well as `detail`, because a refused chat request is
        # rendered in the answer bubble and a bubble reading "(no answer)" tells
        # the reader nothing about why.
        return JSONResponse({"error": msg, "detail": msg, "answer": msg,
                             "sources": []}, status_code=status)

    if kind == "solve":
        if not limits.take_slot():
            return JSONResponse(
                {"error": "busy",
                 "detail": "CADAI is already running its maximum of %d "
                           "analyses. They take under a minute -- try again "
                           "shortly." % limits.SOLVE_CONCURRENT},
                status_code=503)
        t0 = time.monotonic()
        try:
            return await call_next(request)
        finally:
            # try/finally, not a plain call: an endpoint that raises must still
            # give the slot back, or a handful of failed solves permanently
            # closes the server for everyone. The same argument applies to the
            # machine-time budget -- a solve that crashed after eighty seconds
            # cost eighty seconds, and not charging for failures would make a
            # crash loop the cheapest way to burn the host's allowance.
            limits.record_solve_seconds(time.monotonic() - t0)
            limits.free_slot()

    return await call_next(request)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "chat_ready": bool(os.environ.get("GROQ_API_KEY")),
        "search_ready": bool(os.environ.get("TAVILY_API_KEY")),
        "limits": limits.status(),
        # Reports whether OAuth is wired up and which redirect this server will
        # send, because "it does not match what you registered" is the failure
        # every OAuth setup hits first and the hardest one to see from outside.
        "onshape": {
            "oauth_configured": onshape_oauth.configured(),
            "redirect_uri": onshape_oauth.redirect_uri(),
            "api_key_fallback": bool(os.environ.get("ONSHAPE_ACCESS_KEY")),
        },
    }


@app.get("/api/materials")
def material_list():
    """The library, flat and grouped.

    `materials` stays a flat sorted list so nothing that already reads this
    endpoint breaks. `groups` is what the dropdown actually renders, because a
    flat alphabetical list of 145 materials puts "Zirconia" and "Zinc" next to
    the aluminium a team uses every day. `info` rides along so the UI can show
    a material's real numbers without a second request per selection.
    """
    return {
        "materials": materials.names(),
        "default": materials.DEFAULT,
        "groups": materials.catalog(),
        "info": {n: materials.info(n) for n in materials.names()},
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    material: str = Form("Aluminum 6061-T6"),
    mode: str = Form("structural"),
    load_case: str = Form("cantilever"),
    orientation: str = Form("horizontal"),
    load: float = Form(500.0),
    pocketing: bool = Form(False),
    rib_mm: float = Form(3.0),
    density: str = Form("normal"),
    load_context: str = Form(""),
    thickness_mm_in: float = Form(0.0),
):
    data = await limits.read_capped(file)
    name = (file.filename or "").lower()
    px_per_mm = None
    thickness_mm = thickness_mm_in if thickness_mm_in > 0 else 6.35
    # Where the number the solver actually used came from. Returned on every
    # payload, including the boring case, because "6.35" on screen means two
    # very different things depending on whether the file said so.
    thickness_source = ("as entered" if thickness_mm_in > 0
                        else "default — 1/4 in, nothing said otherwise")

    # A typed loading description overrides the dropdowns for anything it
    # actually states. Only the deterministic pass runs here — the LLM pass is
    # the UI's job (via /api/parse_load) so a slow model can never stall a
    # solve, and a hallucinated number can never sneak in behind your back.
    ctx = {}
    if (load_context or "").strip():
        try:
            rx, why = loadctx.parse_regex(load_context)
            ctx = {"text": load_context.strip()[:1000], "applied": rx,
                   "detail": why}
            load_case = rx.get("load_case", load_case)
            orientation = rx.get("orientation", orientation)
            load = float(rx.get("load", load))
            mode = rx.get("mode", mode)
            material = rx.get("material", material)
            thickness_mm = float(rx.get("thickness_mm", thickness_mm))
            if rx.get("thickness_mm") is not None:
                thickness_source = "read from your loading description"
        except Exception as e:
            ctx = {"text": load_context.strip()[:1000], "error": str(e)}

    # STEP files: tessellate → thin-axis silhouette → same FEM pipeline
    if name.endswith(".step") or name.endswith(".stp"):
        # Finer mesh + no decimation for the 2D path: decimation deletes
        # triangles, which punches false gaps in the silhouette, and a coarse
        # mesh rounds small bores away before they're ever rasterized.
        #
        # But "fine" is a request, not a promise. tessellate() runs gmsh in a
        # SUBPROCESS, and a subprocess that is killed -- out of memory on a
        # small container, or past its wall clock on a part with many curved
        # faces -- comes back as "gmsh worker failed" with no hint that the
        # cause was resources rather than geometry. The old code then wrapped
        # that in "STEP could not be read as a 2D plate", which blames the
        # plate detector for something the plate detector never did, and told
        # the user to untick 2D plates -- throwing away the pocketing on a part
        # that is a perfectly good plate.
        #
        # So: try fine, and on failure step down instead of giving up. A 1.5 mm
        # deflection silhouette is worse than a 0.45 mm one -- small bores lose
        # a little diameter -- but it is enormously better than no answer, and
        # the payload says which rung was used so a degraded run never passes
        # itself off as a clean one.
        _LADDER = ((0.45, 400000), (0.8, 250000), (1.5, 120000))
        from .step3d import tessellate as tess, silhouette_png
        from starlette.concurrency import run_in_threadpool
        t = None
        _tess_note = None
        _errs = []
        for _i, (_defl, _cap) in enumerate(_LADDER):
            try:
                t = await run_in_threadpool(tess, data, _defl, _cap)
                if _i:
                    _tess_note = (f"tessellated at {_defl} mm deflection, not the "
                                  f"usual {_LADDER[0][0]} mm — the finer mesh "
                                  f"failed here, so small bores may read a little "
                                  f"undersize")
                break
            except Exception as e:
                _errs.append(f"{_defl} mm: {type(e).__name__}: {e}")
        if t is None:
            # Every rung failed. Report what actually happened, all of it, and
            # only then suggest the 3D fallback -- the last rung is coarse
            # enough that a genuine resource problem has usually cleared by
            # then, so reaching here really does point at the geometry.
            raise HTTPException(400,
                "STEP could not be tessellated at any mesh density. Attempts — "
                + " | ".join(_errs)
                + ". If these say 'gmsh worker failed' the container most likely "
                  "ran out of memory (deploy with --memory 2Gi); otherwise the "
                  "geometry is the problem and unchecking 'Pocketing (2D plates)' "
                  "will show it as a 3D solid instead.")
        try:
            # The solid measures its own thickness, so that wins over anything
            # typed — but say so, otherwise a stated "1/4 in" quietly vanishing
            # looks like the text was ignored.
            _typed_t = thickness_mm
            data, px_per_mm, thickness_mm = silhouette_png(t)
            thickness_source = "measured from the STEP solid"
            if _tess_note:
                ctx.setdefault("detail", []).append(_tess_note)
            if ctx.get("applied", {}).pop("thickness_mm", None) is not None:
                ctx.setdefault("detail", []).append(
                    f"thickness {thickness_mm:g} mm measured from the STEP solid "
                    f"(overrides the {_typed_t:g} mm you wrote)")
        except Exception as e:
            # Tessellation worked, the silhouette did not. That IS the plate
            # detector, so the 3D advice is the right advice here.
            raise HTTPException(400,
                f"STEP tessellated fine but could not be flattened to a plate: "
                f"{type(e).__name__}: {e}. Uncheck 'Pocketing (2D plates)' to "
                f"view it as a 3D solid instead.")
    try:
        res = analysis.run(
            data, material=material, mode=mode, load_case=load_case,
            orientation=orientation, load=load, do_pocketing=pocketing,
            rib_mm=rib_mm, density=density,
            px_per_mm=px_per_mm, thickness_mm=thickness_mm,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(400, f"analysis failed: {e}")

    if ctx:
        res["load_context"] = ctx
    res["orientation"] = orientation
    res["thickness_source"] = thickness_source
    # Top level, not buried in load_context: a run that fell down the
    # tessellation ladder is a DEGRADED run, and on a small part the coarse
    # silhouette can take removal to near zero. Silently returning that reads
    # as "this part cannot be pocketed" when the truth is "the mesh this plan
    # was drawn from is too coarse to trust". The UI shows it in amber.
    if name.endswith(".step") or name.endswith(".stp"):
        res["tessellation_note"] = _tess_note
    layers = res.pop("_pocket_layers", None)
    try:
        res["saved"] = save_outputs(res, res["image_size"],
                                    base_name=file.filename or "part",
                                    pocket_layers=layers)
        # embed the pocketing PNG so the frontend can display it inline
        if res["saved"].get("pocket_png"):
            import base64
            with open(res["saved"]["pocket_png"], "rb") as fh:
                res["pocket_png_data"] = "data:image/png;base64," + \
                    base64.b64encode(fh.read()).decode()
    except Exception as e:
        res["saved"] = {"error": str(e)}
    return JSONResponse(res)


@app.post("/api/analyze3d")
async def analyze3d(
    file: UploadFile = File(...),
    material: str = Form("Aluminum 6061-T6"),
    mode: str = Form("structural"),
    load_case: str = Form("cantilever"),
    orientation: str = Form("horizontal"),
    load: float = Form(500.0),
    load_context: str = Form(""),
    thickness_mm_in: float = Form(0.0),
):
    """Solid stress map — a real tetrahedral FEA of the part as a 3D body.

    This is what runs when 'Pocketing (2D plates)' is unchecked. A STEP file is
    meshed as the actual solid; anything else is the silhouette extruded through
    the stated thickness. Pocketing deliberately has no counterpart here: a
    pocket pattern is a decision about a plate, and there is no defensible way
    to project one onto an arbitrary 3D body.
    """
    data = await limits.read_capped(file)
    thickness_mm = thickness_mm_in if thickness_mm_in > 0 else 6.35

    ctx = {}
    if (load_context or "").strip():
        try:
            rx, why = loadctx.parse_regex(load_context)
            ctx = {"text": load_context.strip()[:1000], "applied": rx,
                   "detail": why}
            load_case = rx.get("load_case", load_case)
            orientation = rx.get("orientation", orientation)
            load = float(rx.get("load", load))
            mode = rx.get("mode", mode)
            material = rx.get("material", material)
            thickness_mm = float(rx.get("thickness_mm", thickness_mm))
        except Exception as e:
            ctx = {"text": load_context.strip()[:1000], "error": str(e)}

    # A volume solve is tens of seconds of pure CPU. Off the event loop, or it
    # blocks every other request on the server for its whole duration.
    from starlette.concurrency import run_in_threadpool
    from . import analysis3d
    try:
        res = await run_in_threadpool(
            analysis3d.run, data,
            filename=file.filename or "part", material=material, mode=mode,
            load_case=load_case, orientation=orientation, load=load,
            thickness_mm=thickness_mm)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(400, f"3D analysis failed: {e}")

    if ctx:
        res["load_context"] = ctx
    res["orientation"] = orientation
    return JSONResponse(res)


@app.post("/api/tessellate")
async def tessellate(file: UploadFile = File(...), size_factor: float = Form(1.0)):
    """STEP file → real 3D solid mesh (gmsh/OpenCASCADE) for the 3D viewer."""
    name = (file.filename or "").lower()
    if not (name.endswith(".step") or name.endswith(".stp")):
        raise HTTPException(400, "tessellation needs a STEP (.step/.stp) file")
    data = await limits.read_capped(file)
    try:
        from .step3d import tessellate as tess
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(tess, data, size_factor)
    except Exception as e:
        raise HTTPException(500, f"STEP tessellation failed: {e}")
    # plate vs 3D from bbox thinness ratio
    spans = sorted(result["bbox"]["spans"])
    ratio = spans[0] / (spans[2] or 1)
    result["is_plate"] = ratio < 0.18
    result["thickness_ratio"] = ratio
    # Measure the stock here, on file load, so the Thickness box is right
    # BEFORE the solve rather than after it. The whole tessellation is already
    # in hand at this point, so this costs one SVD and nothing else.
    try:
        from .step3d import measure_thickness
        result["thickness_mm"] = measure_thickness(result)
    except Exception:
        result["thickness_mm"] = None
    return JSONResponse(result)


class LoadCtxReq(BaseModel):
    text: str
    use_llm: bool = True
    model: str | None = None


@app.post("/api/parse_load")
def parse_load(req: LoadCtxReq):
    """Plain-English loading description → solver settings + a readback.

    Always 200. The deterministic pass needs no key and no network, so this
    still works with the assistant unconfigured; `source` tells you which
    layers actually ran."""
    try:
        return loadctx.parse(req.text, model=req.model,
                             use_llm=req.use_llm and
                             bool(os.environ.get("GROQ_API_KEY")))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "settings": dict(loadctx.DEFAULTS),
                "from_text": [], "assumed": list(loadctx.DEFAULTS),
                "understood": "", "detail": [], "source": "error",
                "error": str(e)}


class ChatReq(BaseModel):
    question: str
    model: str | None = None


def _no_key_answer():
    """Said the same way on both chat endpoints, so a misconfigured install
    cannot get a helpful message on one and a blank panel on the other."""
    return {
        "answer": (
            "The assistant isn't configured yet. Create a file named `.env` "
            "next to START_APP.bat containing:\n\n"
            "GROQ_API_KEY=gsk_your_key_here\n"
            "TAVILY_API_KEY=tvly_your_key_here\n\n"
            "A Groq key is free at console.groq.com/keys (required). Tavily "
            "(tavily.com) is optional and adds live web sources. "
            "Restart the server after saving the file."
        ),
        "sources": [],
        "error": "GROQ_API_KEY not set",
    }


@app.post("/api/chat")
def chat(req: ChatReq):
    # Always answer with 200 + a readable message. A 502 here produced a body
    # with no "answer" field, which the UI rendered as the useless "(no answer)".
    if not os.environ.get("GROQ_API_KEY"):
        return _no_key_answer()
    try:
        return chatmod.ask(req.question, model=req.model)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": f"Assistant error: {e}", "sources": [], "error": str(e)}


@app.post("/api/chat/stream")
def chat_stream(req: ChatReq):
    """The same answer as /api/chat, sent as it is produced.

    Newline-delimited JSON, one event per line, rather than server-sent events:
    the client is fetch() in our own page, not an EventSource, and NDJSON is one
    split on "\\n" instead of a frame parser. The last line is always either a
    "done" event carrying the identical payload /api/chat returns, or an "error"
    event -- so a client can ignore everything else and be exactly where it was
    before this endpoint existed.

    Defined as a normal `def`, not `async def`, on purpose: everything below it
    makes blocking HTTP calls, and FastAPI runs a sync generator in a worker
    thread. As `async def` those calls would block the event loop and stall the
    stress solve running in the next tab.
    """
    import json

    def events():
        if not os.environ.get("GROQ_API_KEY"):
            yield json.dumps({"type": "done", "result": _no_key_answer()}) + "\n"
            return
        try:
            for ev in chatmod.ask_stream(req.question, model=req.model):
                yield json.dumps(ev) + "\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            # An error is delivered as a normal event, not as an HTTP status:
            # by the time it happens the response is already 200 and part of
            # the answer may be on screen. The client shows it in place.
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        # Without this a reverse proxy will happily buffer the whole stream and
        # deliver it in one piece at the end, which is the exact behaviour this
        # endpoint exists to stop. Harmless when nothing is proxying.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/kb")
def kb_status():
    """What is in the local knowledge base, if anything.

    The UI asks once on load so it can say "12 documents indexed" rather than
    leaving the reader to guess whether the corpus they built is being consulted
    at all. An empty or missing knowledge base is a normal state and not an
    error -- the assistant works without one -- so this never fails, it only
    reports `ready: false` alongside the command that fixes it.
    """
    try:
        from . import kb as kbmod
        st = kbmod.stats()
    except Exception as e:
        return {"ready": False, "docs": 0, "chunks": 0, "by_source": {},
                "note": f"could not read the knowledge base: {e}"}
    docs = int(st.get("docs") or 0)
    return {
        "ready": docs > 0,
        "docs": docs,
        "chunks": int(st.get("chunks") or 0),
        "by_source": st.get("by_source") or {},
        "note": ("" if docs else
                 "No local documents yet. Run  python tools/kb_ingest.py seed  "
                 "to index a starter set, or  kb_ingest.py file <path>  to add "
                 "your own notes."),
    }


# ---------------------------------------------------------------------------
# Onshape, per visitor
#
# Every route here is about one thing: nobody's Onshape documents are read with
# anybody else's credentials. The cookie is an opaque random string that means
# nothing outside this process -- it is a key into a dict of tokens, not a token
# -- so an intercepted cookie is useless once the server restarts, and it is
# never readable from JavaScript.
# ---------------------------------------------------------------------------

def _set_session(response, sid):
    response.set_cookie(
        onshape_oauth.COOKIE, sid,
        max_age=onshape_oauth.SESSION_TTL,
        httponly=True,
        # Lax, not Strict: the browser arrives back here as a top-level
        # navigation from oauth.onshape.com, and Strict would withhold the
        # cookie on exactly that request -- the one that has to have it.
        samesite="lax",
        secure=onshape_oauth.secure_cookie(),
        path="/",
    )
    return response


@app.get("/api/onshape/status")
def onshape_status(request: Request):
    return onshape_oauth.status(request.cookies.get(onshape_oauth.COOKIE))


@app.get("/api/onshape/connect")
def onshape_connect(request: Request):
    try:
        url, sid = onshape_oauth.begin(request.cookies.get(onshape_oauth.COOKIE))
    except onshape_oauth.OAuthError as e:
        raise HTTPException(503, str(e))
    # 307 rather than the default 302 so the cookie and the redirect are one
    # response and there is no window in which the browser has been sent to
    # Onshape without knowing which session the reply belongs to.
    return _set_session(RedirectResponse(url, status_code=307), sid)


@app.get("/api/onshape/callback")
def onshape_callback(request: Request):
    """Where Onshape sends the browser back. Registered as a Redirect URL."""
    q = request.query_params
    if q.get("error"):
        return _oauth_page(
            "Onshape did not grant access",
            q.get("error_description") or q.get("error"))
    try:
        onshape_oauth.complete(request.cookies.get(onshape_oauth.COOKIE),
                               q.get("code"), q.get("state"))
    except onshape_oauth.OAuthError as e:
        return _oauth_page("Could not finish connecting to Onshape", str(e))
    return RedirectResponse("/?onshape=connected", status_code=303)


def _oauth_page(title, detail):
    """A readable failure.

    A raw JSON error at the end of a redirect chain is the worst place to put
    one -- the visitor did not type this URL and has no way back except the
    browser button. This is a page, in the app's colours, with a link.
    """
    import html
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8>"
        "<title>CADAI — Onshape</title>"
        "<body style='background:#0a0a0a;color:#e6e6e6;font:14px/1.6 system-ui,"
        "sans-serif;margin:0;display:grid;place-items:center;height:100vh'>"
        "<div style='max-width:34rem;padding:1.5rem'>"
        "<h1 style='font-size:1.1rem;color:#fafafa;margin:0 0 .6rem;"
        "letter-spacing:.02em'>%s</h1>"
        "<p style='color:#a3a3a3;margin:0 0 1.2rem'>%s</p>"
        "<a href='/' style='color:#fafafa'>Back to CADAI</a>"
        "</div></body>" % (html.escape(title), html.escape(detail)),
        status_code=400)


@app.post("/api/onshape/disconnect")
def onshape_disconnect(request: Request):
    onshape_oauth.disconnect(request.cookies.get(onshape_oauth.COOKIE))
    r = JSONResponse({"connected": False})
    r.delete_cookie(onshape_oauth.COOKIE, path="/")
    return r


class OnshapeDocReq(BaseModel):
    url: str


@app.post("/api/onshape/massprops")
def onshape_massprops(req: OnshapeDocReq, request: Request):
    """Mass, volume and centroid for the part in a pasted Onshape link.

    This exists as much to prove the chain as to be useful: if it returns real
    numbers then the application, the redirect, the token exchange and the
    read-documents permission are all correct, and anything else read-only can
    be built on top with confidence.
    """
    ids = onshape_api.parse_document_url(req.url)
    if not ids or not ids["did"]:
        raise HTTPException(400, "That does not look like an Onshape document "
                                 "link. Copy the URL from the address bar with "
                                 "the Part Studio tab open.")
    if not ids["wid"] or not ids["eid"]:
        raise HTTPException(400, "That link is missing the workspace or the "
                                 "tab. Open the Part Studio itself, then copy "
                                 "the address bar.")

    token = onshape_oauth.token_for(request.cookies.get(onshape_oauth.COOKIE))
    # The personal key pair is a development convenience and never a fallback in
    # public. Left unguarded, a visitor who simply declines to connect would be
    # served from the owner's credentials -- reading the owner's documents under
    # the owner's name -- which is the precise failure OAuth was added to
    # remove. Enforced here rather than left as a warning in the deployment
    # guide, because a warning only protects the person who reads it.
    key_ok = bool(os.environ.get("ONSHAPE_ACCESS_KEY")) and not limits.PUBLIC
    if not token and not key_ok:
        if os.environ.get("ONSHAPE_ACCESS_KEY") and limits.PUBLIC:
            raise HTTPException(401, "Connect your own Onshape account to read "
                                     "a document. This server will not read "
                                     "Onshape with its owner's credentials.")
        raise HTTPException(401, "Connect your Onshape account first.")
    try:
        data = onshape_api.get_mass_properties(
            ids["did"], ids["wid"], ids["eid"], wvm=ids["wvm"], token=token)
    except Exception as e:
        code = getattr(getattr(e, "response", None), "status_code", 0)
        if code in (401, 403):
            raise HTTPException(403, "Onshape refused that document. Either the "
                                     "connection expired or the account you "
                                     "authorised cannot open it.")
        if code == 404:
            raise HTTPException(404, "Onshape has no such document, workspace "
                                     "or tab.")
        raise HTTPException(502, "Onshape request failed: %s" % e)
    return {"ids": ids, "massProperties": data,
            "via": "oauth" if token else "api-key"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
