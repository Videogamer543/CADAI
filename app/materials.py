"""
Material library.

Three sources, merged in this order (later ones never overwrite earlier ones):

  1. CURATED  -- the short list an FRC team actually machines, with properties
                 for a specific temper. "Aluminum 6061-T6" is a real, orderable
                 thing; Onshape's plain "Aluminum - 6061" is not, and the
                 difference between them is most of the yield strength.
  2. PRINTED  -- filament and powder-bed plastics, which behave nothing like
                 the stock plastic of the same name (see the note on kd below).
  3. ONSHAPE  -- the full stock Onshape library, so a material picked in CAD can
                 be picked here under the same name.

Every entry is a dict with E [Pa], nu, sy [Pa], rho [kg/m^3], and a `cat` used
to group the dropdown. Optional keys: `uts` [Pa], `kd` (see below) and `note`.

Anisotropy knock-down (`kd`)
----------------------------
The solver is isotropic. A printed part is not: it is a stack of welded layers,
and pulling across those welds is the weakest thing you can do to it. The FEA
has no way to know which way the part was oriented on the bed, so instead of
pretending the problem away, printed materials carry a `kd` factor and the
safety factor is computed against `sy * kd` -- the strength the part would have
if the load happened to land in the worst direction. That is the number worth
designing to, because nobody reprints a bracket for being too strong.

`kd` applies to strength only, not to stiffness. Deflection is far less
sensitive to build direction than fracture is, so E is left alone and the
displacement figure stays honest.
"""
from __future__ import annotations
import csv
import os
from pathlib import Path

from .materials_onshape import ONSHAPE as _ONSHAPE_RAW

# ---------------------------------------------------------------------------
# 1. The short list: what actually gets cut.
# ---------------------------------------------------------------------------
CURATED = {
    "Aluminum 6061-T6":   dict(E=68.9e9, nu=0.33, sy=276e6, uts=310e6, rho=2700,
                               cat="FRC common"),
    "Aluminum 7075-T6":   dict(E=71.7e9, nu=0.33, sy=503e6, uts=572e6, rho=2810,
                               cat="FRC common"),
    "Aluminum 5052-H32":  dict(E=70.3e9, nu=0.33, sy=193e6, uts=228e6, rho=2680,
                               cat="FRC common"),
    "Aluminum 2024-T3":   dict(E=73.1e9, nu=0.33, sy=345e6, uts=483e6, rho=2780,
                               cat="FRC common"),
    "Steel 4130":         dict(E=205e9,  nu=0.29, sy=460e6, uts=560e6, rho=7850,
                               cat="FRC common"),
    "Steel 1018":         dict(E=205e9,  nu=0.29, sy=370e6, uts=440e6, rho=7870,
                               cat="FRC common"),
    "Stainless 304":      dict(E=193e9,  nu=0.29, sy=215e6, uts=505e6, rho=8000,
                               cat="FRC common"),
    "Titanium Ti-6Al-4V": dict(E=113.8e9, nu=0.34, sy=880e6, uts=950e6, rho=4430,
                               cat="FRC common"),
    "Brass 260":          dict(E=110e9,  nu=0.34, sy=310e6, uts=525e6, rho=8530,
                               cat="FRC common"),
    "Polycarbonate":      dict(E=2.4e9,  nu=0.37, sy=62e6,  uts=66e6,  rho=1200,
                               cat="FRC common"),
    "Delrin (POM)":       dict(E=3.1e9,  nu=0.35, sy=70e6,  uts=70e6,  rho=1410,
                               cat="FRC common"),
    "Nylon 6/6":          dict(E=2.9e9,  nu=0.39, sy=82e6,  uts=85e6,  rho=1140,
                               cat="FRC common",
                               note="Stock nylon -- machined bar or sheet, dry. "
                                    "Nylon is hygroscopic: at normal room "
                                    "humidity expect nearer 1.2 GPa and 55 MPa. "
                                    "For printed nylon use a '3D printed' "
                                    "entry, which is a much weaker material."),
    "Carbon fiber (quasi-iso)": dict(E=70e9, nu=0.30, sy=600e6, uts=600e6,
                                     rho=1600, cat="FRC common",
                                     note="A quasi-isotropic laminate treated "
                                          "as one solid. A real layup is only "
                                          "this strong along its fibres."),
}

# ---------------------------------------------------------------------------
# 2. Printed plastics. Deliberately separate from their stock namesakes.
# ---------------------------------------------------------------------------
# The kd figures are the low end of published Z-to-XY tensile ratios for
# well-tuned prints -- conservative on purpose, because a part that breaks
# costs more than a part that is 20% heavier. They assume solid or near-solid
# walls; sparse infill is a different material altogether and no knock-down
# factor rescues an estimate made at 15% gyroid.
PRINTED = {
    "PLA (3D print)":      dict(E=3.5e9, nu=0.36, sy=50e6, uts=50e6, rho=1240,
                                cat="3D printed", kd=0.60,
                                note="Stiff and strong on the bench, but it "
                                     "creeps under sustained load and goes "
                                     "soft in a hot robot. Prototypes and "
                                     "light brackets."),
    "PETG (3D print)":     dict(E=2.1e9, nu=0.40, sy=50e6, uts=50e6, rho=1270,
                                cat="3D printed", kd=0.70,
                                note="The best layer adhesion of the common "
                                     "filaments, so the least penalised across "
                                     "layers. Ductile -- it bends before it "
                                     "snaps."),
    "ABS (3D print)":      dict(E=2.3e9, nu=0.35, sy=40e6, uts=40e6, rho=1050,
                                cat="3D printed", kd=0.45,
                                note="It warps, and the warping is precisely "
                                     "what ruins layer adhesion, hence the "
                                     "heavy knock-down. Needs an enclosure to "
                                     "reach even these numbers."),
    "Nylon (FDM print)":   dict(E=1.4e9, nu=0.39, sy=45e6, uts=48e6, rho=1100,
                                cat="3D printed", kd=0.60,
                                note="Printed nylon, NOT stock nylon -- about "
                                     "half the stiffness of machined bar. "
                                     "Tough and abrasion-resistant, which is "
                                     "why it lasts as gears and rollers. Dry "
                                     "the filament or none of this applies."),
    "Nylon-CF (FDM print)": dict(E=3.5e9, nu=0.38, sy=70e6, uts=75e6, rho=1200,
                                 cat="3D printed", kd=0.40,
                                 note="Chopped carbon fibre stiffens the part "
                                      "in-plane, but the fibres lie down in "
                                      "the layers and bridge nothing across "
                                      "them, so the across-layer penalty is "
                                      "the worst of any filament here."),
    "PA12 (SLS print)":    dict(E=1.7e9, nu=0.40, sy=48e6, uts=48e6, rho=1010,
                                cat="3D printed", kd=0.90,
                                note="Powder-bed nylon fuses in every "
                                     "direction, so it is nearly isotropic and "
                                     "barely penalised. If you can get a part "
                                     "SLS-printed, this is the plastic to "
                                     "design in."),
    "ASA (3D print)":      dict(E=2.2e9, nu=0.35, sy=42e6, uts=42e6, rho=1070,
                                cat="3D printed", kd=0.45,
                                note="ABS that survives sunlight, with the "
                                     "same adhesion problems as ABS."),
    "Polycarbonate (3D print)": dict(E=2.0e9, nu=0.37, sy=55e6, uts=60e6,
                                     rho=1200, cat="3D printed", kd=0.55,
                                     note="The strongest common filament when "
                                          "printed hot and dry, and useless "
                                          "printed wet -- it comes out cloudy "
                                          "and snaps. Takes more heat than "
                                          "anything else here."),
}

DEFAULT = "Aluminum 6061-T6"

# name -> dict. Built once at import.
BUILTIN: dict = {}


def _merge():
    for src in (CURATED, PRINTED):
        for name, d in src.items():
            BUILTIN.setdefault(name, dict(d))
    for name, row in _ONSHAPE_RAW.items():
        cat, E, nu, sy, uts, cys, rho = row
        BUILTIN.setdefault(name.strip(), dict(
            E=E, nu=nu, sy=sy, uts=uts, cys=cys, rho=rho, cat=cat,
            source="Onshape"))


_merge()

# Order the categories appear in the dropdown. Anything unlisted sinks to the
# bottom, alphabetically.
CAT_ORDER = ("FRC common", "3D printed", "Metal", "Plastic", "Composite",
             "Ceramic", "Glass", "Wood", "Rubber")


def _load_csv():
    """Merge a user-supplied Onshape CSV export if one is sitting nearby.

    The bundled table already covers Onshape's stock library, so this is for
    custom materials a team has added to their own Onshape account. It reads
    both the real export headers ("Young's Modulus [Pa]") and the short
    lower-case names an earlier version of this file expected -- that earlier
    version silently imported nothing from a genuine Onshape export, because
    it was looking for column names Onshape does not use.
    """
    root = Path(__file__).resolve().parents[1]
    for cand in (root.parent / "Onshape_Material_Library.csv",
                 root / "Onshape_Material_Library.csv",
                 root / "data" / "Onshape_Material_Library_custom.csv"):
        if not cand.exists():
            continue
        try:
            with open(cand, newline="", encoding="utf-8-sig",
                      errors="ignore") as f:
                for row in csv.DictReader(f):
                    name = (row.get("Name") or row.get("name") or "").strip()
                    if not name:
                        continue

                    def g(*keys):
                        for k in keys:
                            v = row.get(k)
                            if v not in (None, "", "NULL"):
                                try:
                                    return float(v)
                                except ValueError:
                                    pass
                        return None

                    E = g("Young's Modulus [Pa]", "E", "youngsModulus",
                          "elasticModulus")
                    nu = g("Poisson's Ratio", "nu", "poissonsRatio", "poisson")
                    sy = g("Tensile Yield Strength [Pa]", "sy",
                           "yieldStrength", "sigma_y")
                    uts = g("Ultimate Tensile Strength [Pa]", "uts")
                    rho = g("Density [kg/m^3]", "rho", "density")
                    cat = (row.get("Category") or "Other").strip()
                    # No modulus means nothing to solve with, so skip rather
                    # than guess -- the same rule the generator applies.
                    if E and nu:
                        BUILTIN.setdefault(name, dict(
                            E=E, nu=nu, sy=sy or 0, uts=uts or 0,
                            rho=rho or 0, cat=cat, source="custom CSV"))
        except Exception as e:  # a bad CSV must never stop the app booting
            print("material CSV load skipped:", e)
        break


_load_csv()


def get(name: str) -> dict:
    return (BUILTIN.get(name) or BUILTIN.get((name or "").strip())
            or BUILTIN[DEFAULT])


def names():
    return sorted(BUILTIN.keys())


def allowable(mat: dict):
    """Stress the part is allowed to see, in Pa, and the reason for it.

    Three corrections on top of the raw yield figure:

      * A missing yield falls back to ultimate tensile. Several of Onshape's
        metals have no yield point worth quoting, and comparing peak stress
        against zero would report every part as infinitely safe.
      * Yield is clamped to ultimate wherever both exist. Onshape lists PTFE at
        131 MPa yield against 25.6 MPa ultimate; taking that at face value
        would call a part safe at five times the load that breaks it.
      * Failing both, compressive yield is used, for the brittle materials
        Onshape gives no tensile figure at all. This is the weakest claim the
        function makes and it says so in the note: von Mises is a ductile
        criterion, and brick does not obey it.
      * Printed materials are knocked down by `kd` for across-layer weakness.

    Returns (allowable_Pa, note), note being None when nothing was adjusted.
    """
    sy = float(mat.get("sy") or 0)
    uts = float(mat.get("uts") or 0)
    cys = float(mat.get("cys") or 0)
    bits = []

    if sy > 0 and 0 < uts < sy:
        base = uts
        bits.append("library yield exceeds ultimate — using ultimate, "
                    "%.0f MPa" % (uts / 1e6))
    elif sy > 0:
        base = sy
    elif uts > 0:
        base = uts
        bits.append("no yield point published — measured against ultimate, "
                    "%.0f MPa" % (uts / 1e6))
    elif cys > 0:
        base = cys
        bits.append("brittle material, no tensile strength published — "
                    "measured against compressive yield, %.0f MPa; treat this "
                    "as an order of magnitude, not a safety factor"
                    % (cys / 1e6))
    else:
        return 0.0, "no strength data for this material — no safety factor"

    kd = float(mat.get("kd") or 1.0)
    if kd < 1.0:
        base *= kd
        bits.append("printed: allowable cut to %d%% (%.0f MPa) for "
                    "across-layer strength" % (round(kd * 100), base / 1e6))

    return base, ("; ".join(bits) if bits else None)


def catalog():
    """The library grouped for the dropdown: [{label, items:[name, ...]}]."""
    groups: dict = {}
    for name, d in BUILTIN.items():
        groups.setdefault(d.get("cat") or "Other", []).append(name)

    def rank(label):
        return (CAT_ORDER.index(label) if label in CAT_ORDER
                else len(CAT_ORDER), label)

    return [{"label": label, "items": sorted(groups[label], key=str.lower)}
            for label in sorted(groups, key=rank)]


def info(name: str):
    """Everything the UI wants to say about one material."""
    d = get(name)
    a, note = allowable(d)
    return {
        "name": name,
        "E_GPa": round(d["E"] / 1e9, 2),
        "nu": d["nu"],
        "sy_MPa": round((d.get("sy") or 0) / 1e6, 1),
        "uts_MPa": round((d.get("uts") or 0) / 1e6, 1),
        "cys_MPa": round((d.get("cys") or 0) / 1e6, 1),
        "rho": d.get("rho") or 0,
        "cat": d.get("cat") or "Other",
        "kd": d.get("kd"),
        "allowable_MPa": round(a / 1e6, 1),
        "allowable_note": note,
        "note": d.get("note"),
        "source": d.get("source", "built-in"),
    }
