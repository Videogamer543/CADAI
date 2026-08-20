"""
Persist analysis outputs to disk.

Default output location is a `StressViz_Outputs` folder next to the project
(i.e. inside the Claude folder on the Desktop when the app lives at
Claude/stressviz-py). Override with the OUTPUT_DIR environment variable.
"""
from __future__ import annotations
import os
import json
import shutil
import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def output_dir() -> Path:
    env = os.environ.get("OUTPUT_DIR")
    if env:
        d = Path(env)
    else:
        # app/save.py → parents[1] = project root (stressviz-py); .parent = Claude
        d = Path(__file__).resolve().parents[1].parent / "StressViz_Outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stress_color(v: float):
    stops = [(0.0, (0, 200, 70)), (0.25, (120, 210, 0)), (0.5, (234, 179, 8)),
             (0.75, (249, 115, 22)), (1.0, (255, 0, 0))]
    for i in range(1, len(stops)):
        if v <= stops[i][0]:
            a, ca = stops[i - 1]; b, cb = stops[i]
            t = (v - a) / (b - a or 1)
            return tuple(int(ca[j] + (cb[j] - ca[j]) * t) for j in range(3))
    return (255, 0, 0)


_STOPS = [(0.0, (0, 200, 70)), (0.25, (120, 210, 0)), (0.5, (234, 179, 8)),
          (0.75, (249, 115, 22)), (1.0, (255, 0, 0))]


def _ramp_lut():
    """256-entry RGB ramp, so a whole image colours in one vectorised lookup."""
    x = np.linspace(0.0, 1.0, 256)
    pos = np.array([s[0] for s in _STOPS])
    cols = np.array([s[1] for s in _STOPS], float)
    lut = np.stack([np.interp(x, pos, cols[:, j]) for j in range(3)], 1)
    return lut.round().astype(np.uint8)


def _boundary_loops(tris):
    """Outer silhouette and every bore: an edge used by exactly one triangle."""
    t = np.asarray(tris)
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    adj = {}
    for p, q in bnd:
        adj.setdefault(int(p), []).append(int(q))
        adj.setdefault(int(q), []).append(int(p))
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, cur, prev = [], start, -1
        while cur is not None and cur not in seen:
            seen.add(cur)
            loop.append(cur)
            nxt = None
            for v in adj.get(cur, ()):
                if v != prev and v not in seen:
                    nxt = v
                    break
            prev, cur = cur, nxt
        if len(loop) > 3:
            loops.append(loop)
    return loops


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _font(px: int):
    """A real font if the machine has one, the bitmap default if not. The app
    runs on Windows as often as Linux, so no single path can be assumed, and a
    missing font must not cost you the whole PNG."""
    from PIL import ImageFont
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, px)
        except Exception:
            continue
    try:
        return ImageFont.load_default(px)
    except Exception:
        return ImageFont.load_default()


def _text_w(dr, txt, font):
    try:
        b = dr.textbbox((0, 0), txt, font=font)
        return b[2] - b[0], b[3] - b[1]
    except Exception:
        return 7 * len(txt), 11


def _draw_badges(dr, result, W, H, avoid=None):
    """The same WEAK/STRONG callouts the app draws on screen. A saved map that
    only shows colour makes you re-derive where the trouble was; the labels are
    the part of the picture you actually quote in a design review.

    No emoji: the glyphs render on the dev box and turn into boxes on a machine
    whose only font is Arial. A coloured dot says the same thing everywhere."""
    marks = result.get("callouts") or []
    if not marks:
        return
    font = _font(15)
    for m in marks:
        hot = m.get("kind") == "hot"
        pct = int(round(min(float(m.get("pct", 0.0)), 1.0) * 100))
        txt = f"WEAK · {pct}% peak" if hot else f"STRONG · {pct}%"
        tw, th = _text_w(dr, txt, font)
        pad, dot = 12, 9
        bw, bh = tw + pad * 2 + dot + 6, max(th + 12, 26)
        x, y = float(m.get("x", 0)), float(m.get("y", 0))
        bx = min(max(6, x - bw / 2), W - bw - 6)
        by = y - bh - 18
        if by < 6:
            by = min(H - bh - 6, y + 18)
        # The legend sits top-left and the highest-stress node is often right
        # under it. A label hidden behind the key is the one label you most
        # needed, so flip that badge to the other side of its anchor.
        if avoid and _hits(bx, by, bw, bh, avoid):
            alt = min(H - bh - 6, y + 18)
            if not _hits(bx, alt, bw, bh, avoid):
                by = alt
            else:
                bx = min(W - bw - 6, avoid[2] + 10)
        edge = (252, 165, 165) if hot else (134, 239, 172)
        cx, cy = bx + bw / 2, (by + bh if by + bh < y else by)
        dr.line([(cx, cy), (x, y)], fill=edge, width=2)
        dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=edge)
        dr.rounded_rectangle([bx, by, bx + bw, by + bh], radius=bh / 2,
                             fill=(26, 10, 12) if hot else (8, 26, 16),
                             outline=edge, width=2)
        d0 = by + (bh - dot) / 2
        dr.ellipse([bx + pad, d0, bx + pad + dot, d0 + dot],
                   fill=(255, 60, 60) if hot else (34, 197, 94))
        dr.text((bx + pad + dot + 6, by + (bh - th) / 2 - 2), txt,
                fill=(255, 235, 235) if hot else (222, 255, 236), font=font)


def _hits(x, y, w, h, r):
    return not (x + w < r[0] or x > r[2] or y + h < r[1] or y > r[3])


def _draw_key(dr, result, W):
    """Legend panel, same wording as the on-screen one."""
    font_t, font_b, font_f = _font(13), _font(12), _font(11)
    rows = [((34, 197, 94), "STRONG", "— low stress, safe to pocket"),
            ((234, 179, 8), "MODERATE", "— watch under peak load"),
            ((239, 68, 68), "HIGH STRESS", "— carries the load, keep material")]
    title = f"{str(result.get('mode', 'structural')).upper()} STRESS MAP"
    bits = [str(result.get("material", ""))]
    pk = (result.get("peak_vm") or 0) / 1e6
    bits.append(f"peak ≈ {pk:.0f} MPa" if pk >= 100 else
                f"peak ≈ {pk:.2f} MPa" if pk < 10 else f"peak ≈ {pk:.1f} MPa")
    if result.get("safety_factor"):
        bits.append(f"SF {result['safety_factor']:.1f}×")
    pk_ = result.get("pocketing") or {}
    if pk_:
        bits.append(f"{pk_.get('n_pockets', 0)} pockets · "
                    f"{round((pk_.get('removable_frac') or 0) * 100)}% off")
    foot = " · ".join(b for b in bits if b)
    widest = max([_text_w(dr, f"{r[1]} {r[2]}", font_b)[0] + 24 for r in rows]
                 + [_text_w(dr, title, font_t)[0], _text_w(dr, foot, font_f)[0]])
    pw = min(widest + 28, W - 20)
    ph = 34 + len(rows) * 20 + 24
    dr.rounded_rectangle([12, 12, 12 + pw, 12 + ph], radius=10,
                         fill=(10, 14, 24), outline=(40, 54, 80), width=1)
    dr.text((26, 22), title, fill=(226, 232, 240), font=font_t)
    y = 44
    for col, name, tail in rows:
        dr.rounded_rectangle([26, y + 2, 36, y + 12], radius=2, fill=col)
        dr.text((44, y), name, fill=(226, 232, 240), font=font_b)
        w = _text_w(dr, name, font_b)[0]
        dr.text((44 + w + 6, y), tail, fill=(148, 163, 184), font=font_b)
        y += 20
    dr.line([(20, y + 4), (12 + pw - 8, y + 4)], fill=(40, 54, 80))
    dr.text((26, y + 10), foot, fill=(148, 163, 184), font=font_f)
    return (12, 12, 12 + pw, 12 + ph)


def render_png(result, size, path: Path):
    """The map as the app draws it: the field interpolated across each triangle
    rather than one flat colour per element. A flat fill shows you the mesh,
    and the mesh is an artifact of the solver, not a property of the part."""
    W, H = size
    nodes = np.asarray(result["nodes"], float)
    tris = np.asarray(result["tris"])
    # The same array the pocketing engine thresholds against.
    vm = np.asarray(result["von_mises_norm"], float)

    field = np.zeros((H, W), np.float32)
    mask = np.zeros((H, W), bool)
    for t in tris:
        a, b, c = nodes[t[0]], nodes[t[1]], nodes[t[2]]
        x0 = max(0, int(np.floor(min(a[0], b[0], c[0]))))
        x1 = min(W - 1, int(np.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(np.floor(min(a[1], b[1], c[1]))))
        y1 = min(H - 1, int(np.ceil(max(a[1], b[1], c[1]))))
        if x1 < x0 or y1 < y0:
            continue
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if det == 0:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / det
        w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / det
        w2 = 1.0 - w0 - w1
        # a hair of overlap so seams between triangles don't show as hairlines
        inside = (w0 >= -0.0015) & (w1 >= -0.0015) & (w2 >= -0.0015)
        if not inside.any():
            continue
        val = w0 * vm[t[0]] + w1 * vm[t[1]] + w2 * vm[t[2]]
        sub = field[y0:y1 + 1, x0:x1 + 1]
        sub[inside] = val[inside]
        mask[y0:y1 + 1, x0:x1 + 1] |= inside

    idx = np.clip(field * 255.0, 0, 255).astype(np.uint8)
    rgb = _ramp_lut()[idx]
    rgb[~mask] = (5, 8, 14)
    img = Image.fromarray(rgb, "RGB")

    dr = ImageDraw.Draw(img)
    try:
        for loop in _boundary_loops(tris):
            pts = [tuple(nodes[i]) for i in loop]
            dr.line(pts + [pts[0]], fill=(255, 241, 204), width=2, joint="curve")
    except Exception:
        pass
    for poly in result.get("pocket_outlines") or []:
        pts = [tuple(p) for p in poly]
        if len(pts) > 2:
            dr.line(pts + [pts[0]], fill=(8, 12, 20), width=2)
    # Annotations are best-effort: a font or glyph problem must cost you the
    # labels, never the map itself.
    key_rect = None
    try:
        key_rect = _draw_key(dr, result, W)
    except Exception as e:
        print("map legend skipped:", e)
    try:
        _draw_badges(dr, result, W, H, avoid=key_rect)
    except Exception as e:
        print("map labels skipped:", e)
    img.save(path)


# Worded to match what the plan actually draws. The engine no longer leaves a
# whole "keep" band solid -- high stress buys a denser lattice, not more metal --
# so a row promising an amber CAUTION zone described a colour that is not on the
# picture and a rule the engine does not follow.
_POCKET_ROWS = [
    ((34, 197, 94), "POCKET", "— machined out, full depth"),
    ((239, 68, 68), "SOLID", "— peak load path, left full thickness"),
    ((168, 178, 200), "RIB / RIM", "— one rib wide; doubled where stress is high"),
    ((248, 113, 113), "MOUNT HOLE", "— boss collar, pockets stop at it"),
    ((251, 191, 36), "LARGE BORE", "— bearing seat, collar kept solid"),
]


def _dashed_circle(dr, cx, cy, r, colour, width=2, dash=14):
    """A dashed ring, drawn as arcs. PIL has no dash pattern, and the dashes
    matter: a solid ring reads as "this is the edge of the hole", a dashed one
    reads as "keep out of this zone", which is the actual rule for a big bore."""
    import math
    box = [cx - r, cy - r, cx + r, cy + r]
    step = max(6.0, math.degrees(dash / max(r, 1.0)))
    a = 0.0
    while a < 360.0:
        dr.arc(box, a, min(360.0, a + step * 0.55), fill=colour, width=width)
        a += step


def _annotate_pocket_png(path, result):
    """Chips, bore rings, the keep callout and the legend, on the saved plan.

    Drawn from the same records the browser draws from, so the file in the
    outputs folder and the view on screen cannot drift apart.
    """
    from PIL import Image, ImageDraw
    pk = result.get("pocketing") or {}
    plan = Image.open(path).convert("RGB")
    W, H = plan.size

    # The legend goes in a band above the part, not on top of it. On the stress
    # map a floating key is fine -- a covered patch of gradient can be inferred
    # from its neighbours. Here it would hide bores and pockets, and a hole you
    # cannot see is a hole you will cut through.
    band = 34 + len(_POCKET_ROWS) * 20 + 24 + 24
    img = Image.new("RGB", (W, H + band), (7, 10, 18))
    img.paste(plan, (0, band))
    dr = ImageDraw.Draw(img)
    dy = float(band)
    s = max(0.75, min(2.2, max(W, H) / 900.0))
    keep_cut = (result.get("pocket_thresholds") or {}).get("keep", 0.55)

    for h in pk.get("hole_notes") or []:
        x, y, r = float(h["x"]), float(h["y"]) + dy, float(h["r"])
        if h.get("kind") == "large":
            _dashed_circle(dr, x, y, r + 7 * s, (251, 191, 36),
                           width=max(2, int(2 * s)))
        if float(h.get("rim", 0)) >= keep_cut:
            rr = r + 2.5 * s
            dr.ellipse([x - rr, y - rr, x + rr, y + rr],
                       outline=(248, 113, 113), width=max(2, int(2 * s)))

    font = _font(int(13 * s))
    small = _font(int(10 * s))
    # Same rule as the browser: a chip wider than the pocket it names points at
    # the rib wall beside it, which is the one place you must not cut.
    for p in (pk.get("pockets") or [])[:12]:
        if float(p.get("room", 0)) < 13 * s:
            continue
        txt = f"{p['id']} · {int(round(float(p.get('stress', 0)) * 100))}%"
        tw, th = _text_w(dr, txt, font)
        two = float(p["room"]) > 34 * s
        sub = "safe to pocket"
        sw = _text_w(dr, sub, small)[0] if two else 0
        bw = max(tw, sw) + 16 * s
        bh = th + (14 * s if not two else 26 * s)
        bx = min(max(2, float(p["x"]) - bw / 2), W - bw - 2)
        by = min(max(2 + dy, float(p["y"]) + dy - bh / 2), H + dy - bh - 2)
        dr.rounded_rectangle([bx, by, bx + bw, by + bh], radius=4 * s,
                             fill=(6, 20, 12), outline=(74, 222, 128), width=1)
        dr.text((bx + (bw - tw) / 2, by + 5 * s), txt,
                fill=(209, 250, 229), font=font)
        if two:
            dr.text((bx + (bw - sw) / 2, by + 5 * s + th + 3 * s), sub,
                    fill=(110, 231, 183), font=small)

    kn = pk.get("keep_note")
    if kn:
        txt = f"{int(round(float(kn.get('stress', 0)) * 100))}% — do not remove material"
        tw, th = _text_w(dr, txt, font)
        bw, bh = tw + 18 * s, th + 14 * s
        bx = min(max(2, float(kn["x"]) - bw / 2), W - bw - 2)
        by = min(max(2 + dy, float(kn["y"]) + dy - bh / 2), H + dy - bh - 2)
        dr.rounded_rectangle([bx, by, bx + bw, by + bh], radius=4 * s,
                             fill=(30, 6, 8), outline=(248, 113, 113), width=1)
        dr.text((bx + 9 * s, by + 6 * s), txt, fill=(254, 226, 226), font=font)

    _draw_pocket_key(dr, result, W)
    img.save(path)


def _draw_pocket_key(dr, result, W):
    """Legend for the plan view — the rules, in the order you apply them."""
    font_t, font_b, font_f = _font(13), _font(12), _font(11)
    title = "STRESS MAP — MACHINING PLAN"
    pk = result.get("pocketing") or {}
    n_mount = sum(1 for h in (pk.get("hole_notes") or [])
                  if h.get("kind") != "large")
    g = pk.get("geometry") or {}
    bits = [str(result.get("material", "")),
            f"{result.get('holes', 0)} holes ({n_mount} mount)"]
    if pk:
        bits.append(f"{pk.get('n_pockets', 0)} pockets")
        bits.append(f"{round((pk.get('removable_frac') or 0) * 100)}% removed")
    if g.get("length_mm"):
        bits.append(f"{g['length_mm']:g} × {g.get('width_mm', 0):g} mm")
    foot = " · ".join(b for b in bits if b)
    widest = max([_text_w(dr, f"{r[1]} {r[2]}", font_b)[0] + 24
                  for r in _POCKET_ROWS]
                 + [_text_w(dr, title, font_t)[0], _text_w(dr, foot, font_f)[0]])
    pw = min(widest + 28, W - 20)
    ph = 34 + len(_POCKET_ROWS) * 20 + 24
    dr.rounded_rectangle([12, 12, 12 + pw, 12 + ph], radius=10,
                         fill=(10, 14, 24), outline=(40, 54, 80), width=1)
    dr.text((26, 22), title, fill=(226, 232, 240), font=font_t)
    y = 44
    for col, name, tail in _POCKET_ROWS:
        # The two hole rows are rings on screen, so they are rings here too --
        # a filled chip would read as "this colour appears in the picture",
        # and the hole markers are outlines, not fills.
        if name.endswith("HOLE") or name.endswith("BORE"):
            dr.ellipse([26, y + 2, 36, y + 12], outline=col, width=2)
        else:
            dr.rounded_rectangle([26, y + 2, 36, y + 12], radius=2, fill=col)
        dr.text((44, y), name, fill=(226, 232, 240), font=font_b)
        w = _text_w(dr, name, font_b)[0]
        dr.text((44 + w + 6, y), tail, fill=(148, 163, 184), font=font_b)
        y += 20
    dr.line([(20, y + 4), (12 + pw - 8, y + 4)], fill=(40, 54, 80))
    dr.text((26, y + 10), foot, fill=(148, 163, 184), font=font_f)
    return (12, 12, 12 + pw, 12 + ph)


def save_outputs(result, size, base_name="part", pocket_layers=None):
    """Write <timestamp>_<name>.json, stress .png, and (if pocketing) a
    pocketing plan .png. Returns the saved file paths."""
    d = output_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_{Path(base_name).stem}"
    json_path = d / f"{stem}.json"
    png_path = d / f"{stem}_stress.png"
    slim = {k: result[k] for k in
            ("peak_vm", "n_nodes", "n_elems", "load_case", "mode", "material",
             "safety_factor", "holes", "image_size", "pocketing",
             "orientation", "thickness_mm", "load_context",
             "pocket_thresholds", "scale_ref_vm", "callouts", "hot_ref_vm")
            if k in result}
    slim["von_mises_peak_MPa"] = result.get("peak_vm", 0) / 1e6
    with open(json_path, "w") as f:
        json.dump(slim, f, indent=2)
    out = {"dir": str(d), "json": str(json_path), "png": None, "pocket_png": None}
    try:
        render_png(result, size, png_path); out["png"] = str(png_path)
    except Exception as e:
        print("stress PNG failed:", e)
    if pocket_layers is not None:
        try:
            from .pocketing import render_png as pocket_png
            pp = d / f"{stem}_pocketing.png"
            part_mask, hole_mask, core, keep = pocket_layers
            pocket_png(part_mask, hole_mask, core, keep, pp,
                       rib_px=(result.get("pocketing") or {}).get("rib_px", 6))
            out["pocket_png"] = str(pp)
            # Two copies, deliberately. The plain raster is what the browser
            # loads and draws its own live chips over; annotating THAT file gave
            # every pocket two labels, one of them shifted by the height of the
            # legend band. The annotated copy is the one that goes to the
            # machine, where a bare green-and-red raster would say which regions
            # but not which rule -- and "no pocket inside this bore" is a rule
            # you cannot see in a colour.
            try:
                lp = d / f"{stem}_pocketing_labeled.png"
                shutil.copyfile(pp, lp)
                _annotate_pocket_png(lp, result)
                out["pocket_png_labeled"] = str(lp)
            except Exception as e:
                print("pocket PNG labels skipped:", e)
        except Exception as e:
            print("pocket PNG failed:", e)
    return out
