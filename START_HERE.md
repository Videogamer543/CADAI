# StressViz — Start Here 🚀

Your finite-element stress-analysis app. This is the roadmap to open it and what
happens to your outputs.

---

## ▶ Open the app (easiest way)

1. Open the **stressviz-py** folder (you're in it).
2. **Double-click `START_APP.bat`.**
   - First time only: it builds a Python environment and installs packages
     (a few minutes — a black window shows progress).
   - Every time after: it starts instantly.
3. Your browser opens **http://localhost:8000** automatically.
4. Upload a part image, pick a load case, click **▶ Run FEM Analysis**.
5. To stop the app: close the black command window.

> **Need Python first?** If the window says Python isn't installed, get it from
> <https://www.python.org/downloads/> and **tick "Add Python to PATH"** during
> install. Then double-click `START_APP.bat` again.

---

## 💾 Where your outputs go

Every analysis is saved automatically to:

```
Desktop\Claude\StressViz_Outputs\
```

Each run writes two files, timestamped:
- `YYYYMMDD_HHMMSS_<part>.png` — the stress heatmap image
- `YYYYMMDD_HHMMSS_<part>.json` — peak stress (MPa), node/element counts, load case

Want them somewhere else? Set an `OUTPUT_DIR` in your `.env` file (see below).

---

## 🔑 Keys (optional — only for the AI chat / Onshape)

The stress analysis works with **no keys**. Keys are only needed for the AI
assistant and Onshape import.

1. Copy `.env.example` to a new file named `.env`.
2. Paste in your keys. **Rotate any key you've shared before pasting it.**
3. Restart the app.

`.env` is git-ignored and never leaves your machine.

---

## 📚 Teach the assistant (your own knowledge base)

The assistant can read a library of documents **you** choose, before it ever
touches the web. Anything you put in it is searched first, cited by name, and
works with **no API key and no internet** — which is the state most laptops are
in the night before a competition.

Open a terminal in this folder and run:

```
python tools/kb_ingest.py seed                 # starter set: frcdesign.org design references
python tools/kb_ingest.py frcdesign            # ALL of frcdesign.org, tagged per page
python tools/kb_ingest.py url  <link>          # one web page
python tools/kb_ingest.py hub  <link>          # everything a resource-list page links to
python tools/kb_ingest.py crawl <link> --depth 2 --prefix   # a whole section of a site
python tools/kb_ingest.py file <path.pdf|.md|.txt|.html>    # your own notes, saved pages, manuals
python tools/kb_ingest.py text "..." --title "Shop rule"    # a single note, typed in
python tools/kb_ingest.py pdfcheck <path.pdf>  # see what a PDF turns into, before adding it
python tools/kb_ingest.py stats                # what's in there
python tools/kb_ingest.py search "pocket floor"# what the assistant would find
python tools/kb_ingest.py remove <url>         # take something out
```

### All of FRCDesign.org

```
python tools/kb_ingest.py frcdesign
```

A few hundred pages, in four passes: the **design handbook** (structure,
materials, fasteners, 3D printing), the whole **learning course** — every stage
and every section inside it — the worked **mechanism examples**, and the CAD
**best practices**. Takes several minutes. Re-running it refreshes rather than
duplicates, so it is safe to run again next season, and `--only course` does one
pass on its own.

The reason this is one command rather than four is the tagging, which you would
otherwise have to get right by hand. The learning course is two different kinds
of writing under one address: the pages on motors, gears, belts, ball
trajectory and intake geometry are general engineering and belong in the answer
to a general question, while the numbered exercises set challenges whose stated
limits are true only inside the challenge. Tag the whole course as `exercise`
and the teaching pages are pushed to the bottom of every general question — the
opposite of why you ingested them. Tag it all as `reference` and one challenge's
acceptance criteria get quoted back to you as an engineering requirement. The
recipe decides per page, and prints the tag it chose next to each one as it
goes, so you can see it happen.

Doing it by hand is the same thing with the rule written out:

```
python tools/kb_ingest.py crawl https://www.frcdesign.org/learning-course/ \
    --prefix --depth 4 --max 200 \
    --kind-for "/exercise=exercise" --kind-for "project-overview=exercise"
```

`--kind-for REGEX=KIND` works on any crawl and is repeatable; the first rule
whose pattern matches the page's address wins, and pages matching nothing get
`--kind`.

### Resource pages (FIRST's technical resources, and pages like it)

Some pages are not documents. They are lists of documents — FIRST's technical
resources hub is the example this was built for:

```
python tools/kb_ingest.py hub https://www.firstinspires.org/resources/library/frc/technical-resources --dry-run
python tools/kb_ingest.py hub https://www.firstinspires.org/resources/library/frc/technical-resources
```

Run it with `--dry-run` first. It prints every link it would fetch and every
link it would leave out **with the reason**, and adds nothing — so you can see
that it found the Pneumatics Manual and Prototyping 101 rather than the donate
button, before you spend twenty downloads finding out.

The page itself is deliberately **not** added. Its own text is link labels and
one-line blurbs, which share vocabulary with every question you could ask and
answer none of them — and each document in the library competes for a place in
the search results, so an index entry that can never be the right answer costs
you one that could be. What gets added is what the page points at: the PDFs,
and the sites it recommends, each read properly and titled with the words the
page's editors used to describe it.

Videos, social media, sign-in pages and Office files are skipped by name, not
by failing. A YouTube link downloads perfectly and gives you a page of player
buttons; left in, that lands in your library as a real document with nothing in
it. The run tells you it skipped 3 videos instead of reporting 3 successes.

Worth knowing:

```
--dry-run       list what would come in and what wouldn't, add nothing
--pdf-only      just the manuals and guides, not the sites it recommends
--max 20        stop after 20 (the default is 50)
--onsite-only   stay on the hub's own site
--whole-page    if the list comes back suspiciously short
```

If the site blocks the tool (a 403 — some sites only answer browsers), open the
page, save it from your browser, and use `hubfile` instead:

```
python tools/kb_ingest.py hubfile "technical-resources.html" --base https://www.firstinspires.org/resources/library/frc/technical-resources
```

The `--base` is what the saved page's links get resolved against, so it has to
be the address you saved it from.

### PDFs

A PDF does not contain paragraphs. It contains characters at coordinates, and
turning those back into text worth searching is where most of the work went:

* The **running header and footer** are removed. Left in, "FRC Structural
  Design Handbook / Rev. C" appears on all 92 pages, and the search engine then
  decides that is what the document is mostly about.
* **Words split across a line break** are rejoined, so `toler-` + `ance`
  becomes `tolerance` and a search for it finds the page. Real hyphens are
  kept: `10-32`, `6061-T6` and `1/4-20` come through intact.
* **Two-column pages are un-interleaved**, so you get one column and then the
  other instead of alternating half-sentences.
* **Table rows stay together** — `6061-T6 | 276 | 310 | 68.9` rather than a
  material name in one place and its numbers somewhere else.
* **Page numbers are carried through** to the citation, so a source reads
  *Design Handbook > Fastener call-outs (p. 41)* and you can go and look.

Three options worth knowing:

```
--pages 1-60      only the part of a long manual that matters
--password ...    a PDF that asks for a password to open
--ocr force       a scan: run text recognition even on pages that "have" text
```

`pdfcheck` is a dry run: it prints how many pages had text, what sections it
found, and the first page as the assistant will see it, without adding
anything. **Use it on any PDF you are going to rely on.** A scanned manual with
no text layer ingests as a complete success and holds nothing at all, and this
is the one command that shows you that before the answer does.

Text recognition for scans is optional — it needs Tesseract installed on the
computer. `pdfcheck` says whether it's ready, and names what's missing if not.

**Tag what you add** with `--kind`, because it changes how the answer treats it:

| `--kind` | use it for | how the assistant reads it |
|---|---|---|
| `reference` | general engineering practice *(default)* | trusted as general advice |
| `docs` | official documentation for a tool or product | trusted, product-specific |
| `data` | material properties, standards tables | trusted for numbers |
| `vendor` | a manufacturer's page for one part | trusted for that part only |
| `convention` | **your team's handbook or shop notes** | attributed to you, never stated as a universal rule |
| `exercise` | a design challenge or assignment | its limits apply *inside that exercise only* |
| `rules` | a competition manual | binds one season |
| `forum` / `blog` | a thread or one author's opinion | flagged as one team's experience |

The `convention` tag is the important one: "our pocket floors are 0.100 in" is
perfectly true of your end mill and wrong as general advice, so the assistant
says *whose* convention it is instead of presenting it as a requirement.

Re-ingesting the same page **replaces** it rather than duplicating it, so
re-running a crawl to refresh is safe. The library lives in `data/kb.json` —
one file, yours, deletable.

Sources drawn from your library are marked **your library** in purple in the
chat panel, so you can always tell what came from your documents and what came
from the web.

---

## ✅ Numbers are checked against the sources

Every measurement in an answer is matched against the excerpt it cites. If a
number isn't there, the answer says so under **Check these numbers** and the
source list shows a ⚠ instead of a ✓.

This is a text match, not a judgement. A number gets flagged when the source
gives it in different units, or when the assistant worked it out from two cited
dimensions rather than reading it off the page. **A flag means look, not wrong**
— and no flag means the numbers were copied correctly, not that they're right
for your part.

---

## 🎯 Make the pocketing match parts you've actually made

**Run `CALIBRATE_POCKETS.bat`.**

First, the honest version of what this is. The pocketing isn't a trained model
and there's nothing in it that learns. It's geometry with about a dozen
constants in it — how big a bay should be relative to the part, how much of the
plate to remove, when to switch from a truss to a waffle — and those constants
were picked by eye off parts in the 200–400 mm range. That's exactly why a
610 mm bellypan comes back looking too busy: the rule that gives a nice bay on a
300 mm gusset gives a lot of small ones on a plate twice that size.

So you can't train it. You *can* point it at plates you've already machined and
have it move those constants toward them. That's what this does.

**Put your parts in the `reference_parts` folder** — photos, screenshots or
`.step` files, or drag them onto `CALIBRATE_POCKETS.bat` and it files them for
you. **Name each one with its real size**: `bellypan_610mm.png`. Without that
there's no way to know whether a photo is of a 6-inch gusset or a 24-inch
bellypan, and every measurement here depends on scale.

**Option 2 measures and prints, and changes nothing.** You get a row for your
part and a row for what StressViz would cut from the same outline:

```
  real: bellypan             610      80%     0.115    0.0100  truss
  ours: bellypan             610      53%     0.062    0.0133  truss
```

Everything is divided by the part's own size before the two are compared, which
is the only way a gusset and a bellypan can sit in one table. A 40 mm bay is
generous on a gusset and a rounding error on a bellypan; 0.19 *of the span* is
the same design decision at both sizes.

It also splits your parts into a small half and a large half and prints both,
because that's the number that answers the question you're actually asking. On
the six-part test set the overall average said the bays were 1.1× too fine —
near enough to fine — while the large parts were 1.6× too fine and the small
one was 1.2× too coarse. The two errors cancel in the average, and the average
says nothing is wrong.

**Option 3 searches the constants for the values that come closest to your
parts** and writes them to `data/pocket_cal.json`, which the engine reads at the
start of every run. It shows you the change before writing anything. Two things
worth knowing: constants fitted to a handful of parts are a guess with a decimal
point on it, and they're a better guess only for parts the size of the ones you
fed it — so check the result on a part you didn't include. And a constant that
only buys a rounding error's worth of improvement is left alone rather than
moved, because on six parts that improvement is just which pixels landed where.

**Option 5 puts everything back.** The defaults live in the code and the
calibration file only overrides them, so deleting it makes this install cut
exactly what a fresh one would.

Calibration shifts Conservative, Normal and Aggressive together by the same
amount. Those three are a promise about the distance between them — Aggressive
means *more than I'd usually cut* — and tuning one without the others would
quietly turn two settings into one setting.

---

## 📈 Accuracy roadmap (what makes this better than the browser version, and what's next)

**Already upgraded (this Python version):**
- ✅ Real **plane-stress FEM** solver (scikit-fem) instead of beam theory
- ✅ **Quality Delaunay mesh** (Triangle) that refines around holes
- ✅ **SciPy** sparse solve; robust **OpenCV** hole detection
- ✅ Keys server-side; outputs saved to disk

- ✅ **Material library** (16 built-in + your Onshape CSV) & analysis modes
   (Structural / Fatigue / Manufacturing) with safety factor
- ✅ **Pocketing engine** (lattice/truss, hole clearance, rib walls) → saved PNG,
   **calibratable against parts you've machined** (`CALIBRATE_POCKETS.bat`)
- ✅ **3D solid view + STEP tessellation** — upload a `.step`, click **🧊 3D Solid**
   to rotate the real solid (gmsh/OpenCASCADE), auto plate-vs-3D detection
- ✅ **Engineering assistant** (Groq + credible web search + FRC live data)

**Next levers (ask Claude to add any of these):**
1. **Quadratic elements** — one-line swap in `app/fem.py`
   (`ElementTriP1` → `ElementTriP2`) for smoother stress around holes.
2. **Real load input** — click which holes are fixed and where the load acts,
   instead of inferring it from the load case.
3. **3D solid FEA** — mesh the tessellated STEP into tetrahedra (gmsh 3D) and
   solve full 3D elasticity, to shade the solid by real stress.
4. **Onshape geometry** — pull tessellated faces + material directly
   (`app/onshape.py`); switch to **OAuth** before any public release.

---

## 🌐 Turn it into a shareable app later
- **Web:** `docker build -t stressviz .` → deploy on Render / Railway / Fly.io.
- **Desktop `.exe`:** bundle with `pywebview` + `PyInstaller`.

See `README.md` for the full technical detail.
