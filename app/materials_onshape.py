"""
Onshape's stock material library, baked in.

Generated from data/Onshape_Material_Library.csv by
tools/gen_onshape_materials.py -- do not hand-edit. Re-run the generator
against a fresh export instead.

Only rows carrying BOTH a Young's modulus and a Poisson's ratio are here.
Onshape leaves those blank for its woods, rubbers and several of its
composites, and an elastic solver has nothing useful to say about a material
whose elasticity is unspecified. Filling the gaps with plausible textbook
numbers would be worse than omitting them: the map would look just as
confident about a balsa part as about a 7075 one. 63 of 189 rows
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
ONSHAPE = {
    # --- Metal ---
    "300 Series Stainless Steel":            ("Metal", 1.93e+11, 0.29, 2.07e+08, 5.17e+08, 0, 7850),
    "400 Series Stainless Steel":            ("Metal", 2e+11, 0.282, 5.85e+08, 7.58e+08, 0, 7720),
    "A2 Stainless Steel":                    ("Metal", 1.93e+11, 0.29, 2.07e+08, 5.17e+08, 0, 8030),
    "A4 Stainless Steel":                    ("Metal", 1.93e+11, 0.29, 2.07e+08, 5.17e+08, 0, 7990),
    "Aluminum":                              ("Metal", 6.89e+10, 0.33, 2.76e+07, 6.89e+07, 0, 2700),
    "Aluminum - 1060":                       ("Metal", 6.89e+10, 0.33, 2.76e+07, 6.89e+07, 0, 2700),
    "Aluminum - 1100":                       ("Metal", 6.89e+10, 0.33, 3.45e+07, 8.96e+07, 0, 2720),
    "Aluminum - 2024":                       ("Metal", 7.24e+10, 0.32, 2.76e+08, 4.27e+08, 0, 2780),
    "Aluminum - 356":                        ("Metal", 7.03e+10, 0.33, 1.38e+08, 2.07e+08, 0, 2685),
    "Aluminum - 380":                        ("Metal", 7.1e+10, 0.33, 1.59e+08, 3.17e+08, 0, 2768),
    "Aluminum - 5052":                       ("Metal", 7e+10, 0.33, 1.59e+08, 2.14e+08, 0, 2680),
    "Aluminum - 6061":                       ("Metal", 6.89e+10, 0.35, 2.41e+08, 2.9e+08, 0, 2720),
    "Aluminum - 7050":                       ("Metal", 7.17e+10, 0.33, 4.69e+08, 5.24e+08, 0, 2800),
    "Aluminum - 7075":                       ("Metal", 7.17e+10, 0.33, 4.55e+08, 5.31e+08, 0, 2810),
    "Aluminum - 7178":                       ("Metal", 7.17e+10, 0.33, 5.38e+08, 6.07e+08, 5.3e+08, 2830),
    "Aluminum bronze (3-10% Al)":            ("Metal", 1.1e+11, 0.316, 2.05e+08, 5.15e+08, 0, 8200),
    "Beryllium":                             ("Metal", 3.03e+11, 0.07, 2.4e+08, 3.7e+08, 2.7e+08, 1840),
    "Beryllium copper":                      ("Metal", 1.15e+11, 0.3, 2.21e+08, 4.83e+08, 0, 8175),
    "Brass":                                 ("Metal", 1.06e+11, 0.318, 2.55e+08, 4.3e+08, 1.44e+08, 8490),
    "Brass - casting":                       ("Metal", 1.05e+11, 0.357, 1.95e+08, 4.9e+08, 1.65e+08, 8550),
    "Brass - rolled and drawn":              ("Metal", 9.7e+10, 0.311, 1.24e+08, 3.25e+08, 0, 8580),
    "Brass 60/40":                           ("Metal", 1.05e+11, 0.35, 1.4e+08, 3.6e+08, 0, 8520),
    "Bronze (8-14% Sn)":                     ("Metal", 1.05e+11, 0.34, 1.5e+08, 3.05e+08, 0, 8150),
    "Bronze - lead":                         ("Metal", 1.15e+11, 0.28, 8.3e+07, 2.55e+08, 0, 8200),
    "Bronze - phosphorous":                  ("Metal", 1.1e+11, 0.34, 1.31e+08, 3.24e+08, 0, 8850),
    "Cadmium":                               ("Metal", 5.52e+10, 0.33, 0, 7.5e+07, 0, 8640),
    "Calcium":                               ("Metal", 2.34e+10, 0.31, 1.37e+07, 4e+07, 0, 1540),
    "Carbon Steel":                          ("Metal", 2.03e+11, 0.29, 6.85e+08, 9.87e+08, 0, 7850),
    "Cast iron":                             ("Metal", 1.47e+11, 0.287, 4.28e+08, 4.97e+08, 1.02e+09, 6975),
    "Chromium":                              ("Metal", 2.48e+11, 0.2, 3.62e+08, 4.13e+08, 0, 7197),
    "Cobalt":                                ("Metal", 2.11e+11, 0.32, 2.25e+08, 8e+08, 0, 8746),
    "Copper":                                ("Metal", 1.1e+11, 0.343, 3.33e+07, 2.1e+08, 0, 8940),
    "Cupronickel":                           ("Metal", 1.52e+11, 0.325, 1.03e+08, 3.59e+08, 0, 8924),
    "Ductile Iron":                          ("Metal", 1.54e+11, 0.283, 5.01e+08, 6.89e+08, 1.26e+09, 7086),
    "Duralumin":                             ("Metal", 7.31e+10, 0.33, 3.45e+08, 4.83e+08, 0, 2790),
    "Gold":                                  ("Metal", 7.72e+10, 0.42, 0, 1.2e+08, 0, 19320),
    "Hardened Alloy Steel":                  ("Metal", 2e+11, 0.285, 4.15e+08, 6.55e+08, 0, 7850),
    "Hardened Carbon Steel":                 ("Metal", 2e+11, 0.292, 8.96e+08, 1.12e+09, 2.16e+09, 7850),
    "Hardened Stainless Steel":              ("Metal", 1.93e+11, 0.29, 9.65e+08, 1.28e+09, 0, 7740),
    "Iron":                                  ("Metal", 2e+11, 0.291, 5e+07, 5.4e+08, 0, 7850),
    "Lead":                                  ("Metal", 1.4e+10, 0.42, 5.5e+06, 1.8e+07, 0, 11349),
    "Light alloy based on Al":               ("Metal", 7.74e+10, 0.327, 2.78e+08, 3.44e+08, 9.57e+07, 2680),
    "Light alloy based on Mg":               ("Metal", 4.52e+10, 0.337, 1.65e+08, 2.47e+08, 1.8e+08, 1815),
    "Manganese Bronze":                      ("Metal", 1.05e+11, 0.34, 4.6e+08, 8.2e+08, 0, 8359),
    "Neodymium":                             ("Metal", 4.14e+10, 0.281, 1.65e+08, 1.7e+08, 0, 7007),
    "Nichrome":                              ("Metal", 1.98e+11, 0.2925, 3.35e+08, 7.15e+08, 0, 8400),
    "Nickel":                                ("Metal", 2.07e+11, 0.31, 5.9e+07, 3.17e+08, 0, 8908),
    "Nickel 20":                             ("Metal", 1.93e+11, 0.31, 3e+08, 6.2e+08, 0, 8090),
    "Nickel 200":                            ("Metal", 1.8e+11, 0.31, 1.48e+08, 4.62e+08, 0, 8890),
    "Nickel silver":                         ("Metal", 1.25e+11, 0.33, 1.86e+08, 4.14e+08, 0, 8650),
    "Palladium":                             ("Metal", 1.17e+11, 0.39, 0, 1.8e+08, 0, 12160),
    "Phosphor bronze":                       ("Metal", 1.1e+11, 0.34, 1.65e+08, 3.79e+08, 0, 8900),
    "Phosphor Bronze 510":                   ("Metal", 1.1e+11, 0.34, 1.31e+08, 3.24e+08, 0, 8850),
    "Platinum":                              ("Metal", 1.71e+11, 0.39, 7e+07, 1.45e+08, 0, 21400),
    "Plutonium":                             ("Metal", 9.65e+10, 0.18, 2.75e+08, 4e+08, 4.15e+08, 19816),
    "Red Brass":                             ("Metal", 1.15e+11, 0.307, 2.7e+08, 3.45e+08, 0, 8746),
    "Silver":                                ("Metal", 7.6e+10, 0.37, 4.5e+07, 1.4e+08, 0, 10490),
    "Stainless Steel":                       ("Metal", 1.93e+11, 0.29, 2.07e+08, 5.17e+08, 0, 7740),
    "Stainless Steel 17-4":                  ("Metal", 1.97e+11, 0.272, 7.23e+08, 9.31e+08, 0, 7750),
    "Stainless Steel 18-8":                  ("Metal", 1.93e+11, 0.29, 2.07e+08, 5.17e+08, 0, 8030),
    "Stainless Steel 2205":                  ("Metal", 1.9e+11, 0.3, 4.85e+08, 6.55e+08, 0, 7820),
    "Stainless Steel 303":                   ("Metal", 1.93e+11, 0.25, 2.41e+08, 6.21e+08, 0, 8000),
    "Stainless Steel 304":                   ("Metal", 1.93e+11, 0.29, 2.07e+08, 5.17e+08, 0, 8030),
    "Stainless Steel 316":                   ("Metal", 1.93e+11, 0.29, 2.07e+08, 5.17e+08, 0, 7990),
    "Stainless Steel 416":                   ("Metal", 2e+11, 0.282, 5.85e+08, 7.58e+08, 0, 7800),
    "Steel":                                 ("Metal", 2e+11, 0.29, 1.8e+08, 3.25e+08, 0, 7850),
    "Steel 1010":                            ("Metal", 2e+11, 0.29, 1.8e+08, 3.25e+08, 0, 7870),
    "Steel 1020":                            ("Metal", 2.05e+11, 0.29, 2.05e+08, 3.8e+08, 0, 7870),
    "Steel 4130":                            ("Metal", 2.05e+11, 0.29, 4.35e+08, 6.7e+08, 0, 7850),
    "Steel 4340":                            ("Metal", 1.92e+11, 0.29, 4.7e+08, 7.45e+08, 0, 7850),
    "Steel 8620":                            ("Metal", 2.05e+11, 0.29, 3.93e+08, 6.69e+08, 0, 7850),
    "Steel ASTM A194 Grade 2H":              ("Metal", 2e+11, 0.29, 1.03e+09, 1.21e+09, 0, 7850),
    "Steel ASTM A285 Grade C":               ("Metal", 2e+11, 0.29, 2.05e+08, 3.8e+08, 0, 7850),
    "Steel ASTM A325 Type 1":                ("Metal", 2e+11, 0.29, 1.03e+09, 1.21e+09, 0, 7850),
    "Steel ASTM A36":                        ("Metal", 2e+11, 0.26, 2.48e+08, 4e+08, 0, 7861),
    "Steel ASTM A490 Type 1":                ("Metal", 2e+11, 0.29, 1.03e+09, 1.21e+09, 0, 7850),
    "Steel ASTM A490 Type 3":                ("Metal", 2e+11, 0.29, 1.03e+09, 1.21e+09, 0, 7850),
    "Steel ASTM A563 Grade A":               ("Metal", 2e+11, 0.3, 3.1e+08, 4.14e+08, 0, 7850),
    "Steel ASTM A563 Grade DH":              ("Metal", 2e+11, 0.3, 7.58e+08, 8.62e+08, 0, 7850),
    "Steel Class 10.9":                      ("Metal", 2.1e+11, 0.3, 9.4e+08, 1.04e+09, 0, 7850),
    "Steel Class 12.9":                      ("Metal", 2.1e+11, 0.3, 1.1e+09, 1.22e+09, 0, 7850),
    "Steel Class 4.8":                       ("Metal", 2.1e+11, 0.3, 2.4e+08, 4e+08, 0, 7850),
    "Steel Class 5.8":                       ("Metal", 2.1e+11, 0.3, 3e+08, 5e+08, 0, 7850),
    "Steel Class 8.8":                       ("Metal", 2.1e+11, 0.3, 6.4e+08, 8e+08, 0, 7850),
    "Steel Grade 2":                         ("Metal", 2e+11, 0.29, 2.48e+08, 4.14e+08, 0, 7850),
    "Steel Grade 5":                         ("Metal", 2e+11, 0.29, 5.58e+08, 7.24e+08, 0, 7850),
    "Steel Grade 8":                         ("Metal", 2e+11, 0.29, 8.96e+08, 1.03e+09, 0, 7850),
    "Titanium":                              ("Metal", 1.16e+11, 0.34, 1.4e+08, 2.2e+08, 0, 4500),
    "Titanium Grade 2":                      ("Metal", 1.05e+11, 0.37, 2.75e+08, 3.44e+08, 0, 4500),
    "Titanium Grade 5":                      ("Metal", 1.14e+11, 0.342, 8.8e+08, 9.5e+08, 9.7e+08, 4500),
    "Titanium Grade 7":                      ("Metal", 1.05e+11, 0.37, 2.75e+08, 3.44e+08, 0, 4500),
    "Tungsten":                              ("Metal", 4e+11, 0.28, 7.5e+08, 9.8e+08, 0, 19600),
    "White metal":                           ("Metal", 5.3e+10, 0.35, 3e+07, 6.3e+07, 3.03e+07, 7100),
    "Wrought Iron":                          ("Metal", 1.93e+11, 0.278, 1.9e+08, 3.03e+08, 0, 7750),
    "Yellow Brass":                          ("Metal", 1.05e+11, 0.34, 3.45e+08, 4.2e+08, 0, 8470),
    "Zinc":                                  ("Metal", 9.65e+10, 0.331, 0, 3.7e+07, 0, 7135),
    # --- Plastic ---
    "ABS":                                   ("Plastic", 2.31e+09, 0.364, 4.48e+07, 4.04e+07, 0, 1052),
    "ABS M30":                               ("Plastic", 2.3e+09, 0.35, 3e+07, 2.5e+07, 9e+07, 1040),
    "Acetal (Delrin)":                       ("Plastic", 2.66e+09, 0.363, 5.95e+07, 5.76e+07, 6.01e+07, 1356),
    "Acrylic":                               ("Plastic", 2.94e+09, 0.35, 6.05e+07, 6.49e+07, 0, 1163),
    "GPPS":                                  ("Plastic", 2.9e+09, 0.4, 3.25e+07, 4.4e+07, 9e+07, 1041),
    "HDPE (High-Density Polyethylene)":      ("Plastic", 9.59e+08, 0.46, 2.59e+07, 2.69e+07, 0, 941),
    "Nylon":                                 ("Plastic", 2.95e+09, 0.39, 7.17e+07, 7.52e+07, 6.68e+07, 1180),
    "PC/ABS":                                ("Plastic", 2.64e+09, 0.353, 5.64e+07, 5.47e+07, 0, 1100),
    "PEEK (Polyether Ether Ketone)":         ("Plastic", 3.93e+09, 0.4, 9.81e+07, 9.97e+07, 0, 1320),
    "PET":                                   ("Plastic", 3.14e+09, 0.37, 6.18e+07, 4.44e+07, 5.06e+07, 1380),
    "PETG":                                  ("Plastic", 2.59e+09, 0.4, 4.73e+07, 4.06e+07, 0, 1301),
    "Phenolic":                              ("Plastic", 7e+09, 0.24, 0, 5.32e+07, 2.06e+08, 1600),
    "PLA":                                   ("Plastic", 2.34e+09, 0.39, 4e+07, 6.45e+07, 0, 1250),
    "Polycarbonate":                         ("Plastic", 2.36e+09, 0.37, 6.33e+07, 6.62e+07, 6.57e+07, 1190),
    "Polyester":                             ("Plastic", 4.03e+09, 0.3, 0, 5.18e+07, 0, 1390),
    "Polypropylene":                         ("Plastic", 1.47e+09, 0.43, 3.26e+07, 7.46e+07, 1e+07, 913),
    "Polystyrene, High-Impact":              ("Plastic", 1.9e+09, 0.41, 1.93e+07, 3.2e+07, 0, 1100),
    "Polyurethane":                          ("Plastic", 1.67e+08, 0.41, 1.29e+07, 2.36e+07, 414000, 1200),
    "PPO":                                   ("Plastic", 2.66e+09, 0.38, 5.1e+07, 5.21e+07, 0, 1050),
    "PTFE":                                  ("Plastic", 5.99e+08, 0.46, 1.31e+08, 2.56e+07, 7.31e+06, 2200),
    "PVC":                                   ("Plastic", 2.8e+09, 0.4, 4.26e+07, 2.34e+07, 0, 1467),
    "Teflon":                                ("Plastic", 5.99e+08, 0.46, 1.31e+08, 2.56e+07, 7.31e+06, 2159),
    "ULTEM":                                 ("Plastic", 7.48e+09, 0.377, 1.14e+08, 1.26e+08, 1.73e+08, 1270),
    # --- Ceramic ---
    "Alumina Oxide":                         ("Ceramic", 3.7e+11, 0.22, 0, 3e+08, 0, 3848),
    "Brick":                                 ("Ceramic", 2.21e+10, 0.212, 0, 0, 3.3e+07, 1765),
    "Concrete":                              ("Ceramic", 3e+10, 0.2, 0, 0, 8.7e+07, 2300),
    "Porcelain":                             ("Ceramic", 1.04e+11, 0.17, 0, 0, 5.9e+08, 2403),
    "Silicon Carbide":                       ("Ceramic", 4.1e+11, 0.14, 0, 0, 4.6e+09, 3100),
    "Silicon Nitride":                       ("Ceramic", 2.9e+11, 0.25, 0, 0, 0, 3211),
    "Zirconia":                              ("Ceramic", 1.86e+11, 0.33, 0, 5.51e+08, 3e+09, 6062),
}

# Rows Onshape ships with no elastic data. Kept as a list so the omission
# is visible in the source instead of being a silent gap in the dropdown.
NO_ELASTIC_DATA = (
    "Alnico",  # Metal
    "Aluminum foil",  # Metal
    "Antimony",  # Metal
    "Babbitt",  # Metal
    "Iridium",  # Metal
    "Lithium",  # Metal
    "Magnesium",  # Metal
    "Manganese",  # Metal
    "Maple, Hard",  # Metal
    "Maple, Soft",  # Metal
    "Mercury",  # Metal
    "Molybdenum",  # Metal
    "Silicon Iron",  # Metal
    "Sodium",  # Metal
    "Solder 50/50 Pb Sn",  # Metal
    "Tin",  # Metal
    "Carbon fiber epoxy (61%)",  # Composite
    "Foamcore",  # Composite
    "FR-4",  # Composite
    "Glass-filled epoxy (35%)",  # Composite
    "Glass-filled nylon (35%)",  # Composite
    "Glass-filled polyester (35%)",  # Composite
    "Kevlar epoxy (53%)",  # Composite
    "Polyetherimide",  # Composite
    "S-glass epoxy (45%)",  # Composite
    "Ferrite",  # Ceramic
    "Glass",  # Glass
    "Ash, White",  # Wood
    "Balsa",  # Wood
    "Basswood",  # Wood
    "Beech, European",  # Wood
    "Birch",  # Wood
    "Cedar, Western Red",  # Wood
    "Cherry, Black",  # Wood
    "Cocobolo",  # Wood
    "Cork",  # Wood
    "Douglas Fir",  # Wood
    "Ebony, African",  # Wood
    "Elm, American",  # Wood
    "Hickory",  # Wood
    "High Density Fiberboard (HDF)",  # Wood, Composite
    "Lignum Vitae (Ironwood)",  # Wood
    "Mahogany, Honduran",  # Wood
    "Medium Density Fiberboard (MDF)",  # Wood, Composite
    "Oak, Red",  # Wood
    "Oak, White",  # Wood
    "Padauk",  # Wood
    "Poplar",  # Wood
    "Purpleheart",  # Wood
    "Redwood",  # Wood
    "Rosewood, Brazilian",  # Wood
    "Rosewood, East Indian",  # Wood
    "Rosewood, Honduran",  # Wood
    "Spruce, Sitka",  # Wood
    "Teak",  # Wood
    "Walnut, Black",  # Wood
    "Butyl",  # Rubber
    "Ethylene Propylene Diene Monomer (EPDM)",  # Rubber
    "Fluoroelastomer (FKM)",  # Rubber
    "Neoprene",  # Rubber
    "Nitrile",  # Rubber
    "Silicone Rubber",  # Rubber
    "Graphite",  # Earth
)
