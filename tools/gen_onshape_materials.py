"""
Turn an Onshape material-library CSV export into app/materials_onshape.py.

Run it from the project root after dropping a fresh export in data/:

    python tools/gen_onshape_materials.py

The table is generated rather than loaded at runtime on purpose. Parsing a CSV
on import means the app can fail to start because a data file went missing or
got saved with a stray BOM, and a material library is not something worth
risking startup over. Generating it puts every parse failure here, at a moment
when a human is watching, and ships a plain Python dict that cannot surprise
anyone at 3 am before a competition.
"""
from __future__ import annotations
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "Onshape_Material_Library.csv")
DST = os.path.join(ROOT, "app", "materials_onshape.py")

E = "Young's Modulus [Pa]"
NU = "Poisson's Ratio"
SY = "Tensile Yield Strength [Pa]"
UT = "Ultimate Tensile Strength [Pa]"
CY = "Compressive Yield Strength [Pa]"
RH = "Density [kg/m^3]"

# Metals first because that is what the tool is mostly pointed at, then the
# things an FRC team plausibly machines, then the rest.
CAT_ORDER = {"Metal": 0, "Plastic": 1, "Composite": 2, "Ceramic": 3,
             "Glass": 4, "Wood": 5, "Wood, Composite": 5, "Rubber": 6,
             "Earth": 7}

HEADER = '''"""
Onshape's stock material library, baked in.

Generated from data/Onshape_Material_Library.csv by
tools/gen_onshape_materials.py -- do not hand-edit. Re-run the generator
against a fresh export instead.

Only rows carrying BOTH a Young's modulus and a Poisson's ratio are here.
Onshape leaves those blank for its woods, rubbers and several of its
composites, and an elastic solver has nothing useful to say about a material
whose elasticity is unspecified. Filling the gaps with plausible textbook
numbers would be worse than omitting them: the map would look just as
confident about a balsa part as about a 7075 one. {dropped} of {total} rows
were dropped for that reason and are listed at the bottom, so it is clear
they were seen rather than lost in parsing.

`uts` is carried alongside `sy` because Onshape's own figures sometimes
contradict each other -- PTFE is listed at 131 MPa yield against 25.6 MPa
ultimate, which cannot be true of any material. materials.allowable() takes
the lower of the two rather than the flattering one.

`cys` (compressive yield) is carried for the handful of brittle materials --
brick, concrete, porcelain, silicon carbide -- that Onshape gives no tensile
strength at all, because compression is the only way anyone loads them. It is
a last resort and materials.allowable() labels it loudly when it is used.
"""
from __future__ import annotations

# name -> (category, E [Pa], nu, tensile yield [Pa], ultimate tensile [Pa],
#          compressive yield [Pa], rho [kg/m^3])
# A strength of 0 means Onshape did not state one.
ONSHAPE = {{
'''


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _num(x):
    if x == 0:
        return "0"
    return "%.6g" % x if abs(x) < 1e6 else "%.4g" % x


def main():
    if not os.path.exists(SRC):
        sys.exit("no CSV at %s" % SRC)
    with open(SRC, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    keep, dropped = [], []
    for r in rows:
        name = (r.get("Name") or "").strip()
        if not name:
            continue
        e, nu = _f(r.get(E)), _f(r.get(NU))
        cat = (r.get("Category") or "Other").strip()
        if e <= 0 or nu <= 0:
            dropped.append((cat, name))
            continue
        keep.append((cat, name, e, nu, _f(r.get(SY)), _f(r.get(UT)),
                     _f(r.get(CY)), _f(r.get(RH))))

    keep.sort(key=lambda t: (CAT_ORDER.get(t[0], 9), t[1].lower()))
    dropped.sort(key=lambda t: (CAT_ORDER.get(t[0], 9), t[1].lower()))

    out = [HEADER.format(dropped=len(dropped), total=len(rows))]
    cur = None
    for cat, name, e, nu, sy, uts, cys, rho in keep:
        if cat != cur:
            cur = cat
            out.append("    # --- %s ---\n" % cat)
        key = '"%s":' % name.replace('"', '\\"')
        out.append('    %-40s ("%s", %s, %s, %s, %s, %s, %s),\n'
                   % (key, cat, _num(e), _num(nu), _num(sy), _num(uts),
                      _num(cys), _num(rho)))
    out.append("}\n\n")
    out.append("# Rows Onshape ships with no elastic data. Kept as a list so the omission\n"
               "# is visible in the source instead of being a silent gap in the dropdown.\n")
    out.append("NO_ELASTIC_DATA = (\n")
    for cat, name in dropped:
        out.append('    "%s",  # %s\n' % (name.replace('"', '\\"'), cat))
    out.append(")\n")

    with open(DST, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    print("wrote %s: %d materials, %d dropped for missing elastic data"
          % (os.path.relpath(DST, ROOT), len(keep), len(dropped)))


if __name__ == "__main__":
    main()
