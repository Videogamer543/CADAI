# StressViz — Python app (real FEM engine)

A finite-element stress-analysis and pocketing tool. The compute engine runs in
Python (SciPy + scikit-fem + Triangle) so it can use a **quality unstructured
mesh** and a **real plane-stress FEM solve** — the accuracy the browser-only
version couldn't reach. FastAPI serves the API and the frontend.

## Why this breaks the accuracy ceiling
| Browser (JS) version | Python version |
|---|---|
| Beam theory + Kirsch hole factors | Real plane-stress **FEM** (scikit-fem) |
| Staircase pixel grid | Quality **Delaunay** mesh (Triangle), refines at holes |
| Hand-rolled conjugate gradient | SciPy sparse **direct solve** |
| Silhouette guess | OpenCV **contour + hole** extraction (STEP/Onshape optional) |
| Keys hardcoded in HTML | Keys **server-side** in `.env` |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # fill in your keys (rotate any shared ones!)
uvicorn app.main:app --reload --port 8000
# open http://localhost:8000
```

Upload a part image, pick a load case, and hit **Run FEM Analysis** — you'll see
the true von Mises field on the quality mesh, with peak stress in MPa.

## Deploy as an app
- **Web app:** `docker build -t stressviz . && docker run -p 8000:8000 --env-file .env stressviz`
  then push to Render / Railway / Fly.io (all have free tiers). Set the env vars
  in the host dashboard, not in the image.
- **Desktop app:** wrap with `pywebview` + `PyInstaller` to ship a `.exe`/`.dmg`
  that bundles the backend and opens the frontend in a native window.

## The assistant's retrieval pipeline

Three stages, each fixing a different way a cited answer goes wrong.

**1. Analysis → query.** `app/chat.py::analyze()` rewrites the question into
expert search vocabulary and classifies its scope (general practice vs. what
one named document says) before anything is retrieved. Retrieval quality is
bounded by the query, so this runs first and everything downstream inherits it.

**2. Retrieval.** Two retrievers feed one ranked pool:

- **Your local corpus** (`app/kb.py`) — a BM25 index over documents you ingest
  with `tools/kb_ingest.py`. Pure Python, no model download, no torch, works
  offline. It is consulted on every question, before the web, and it is the
  only part of the assistant that keeps working with no API key. Documents are
  chunked with overlap and inherit their section headings, so a chunk pulled
  from the middle of a long page is still readable. PDFs go through
  `app/pdftext.py` first, which is the difference between indexing a manual and
  indexing the noise around it: running headers are dropped before they become
  the most repeated phrase in the corpus, hyphenated line-wraps are rejoined so
  `toler-`/`ance` is one searchable token, two-column pages are un-interleaved,
  table rows are kept whole, and every chunk carries the page it came from so a
  citation can say *(p. 41)*. `kb_ingest.py pdfcheck` shows all of that before
  anything is added — a scan with no text layer otherwise ingests as a complete
  success and holds nothing. Link-hub pages (FIRST's technical-resources
  library) go through `kb_ingest.py hub`, which indexes what the page *points
  at* rather than the page: a list of resources shares vocabulary with every
  question and answers none, so indexing it spends a ranking slot on an entry
  that can never be the right answer. `crawl` cannot do this job — it stays on
  one host, and a hub's whole value is that it points off one, and it hands
  every URL to the HTML extractor, so a linked PDF arrives as tag-stripped PDF
  bytes.
- **Web search** (Tavily), whose ~700-character snippet is the *introduction*
  of an article rather than its answer. So the top results are re-fetched in
  full (`app/webtext.py`), stripped of navigation, and re-excerpted around the
  question's terms — and re-typed, since a page often only reveals that it is a
  narrow design brief well below the fold.

Ranking mixes them by normalising BM25 against the best local hit so the scores
are comparable, then applying a per-source-kind bias whose **sign flips** with
the question's scope: a competition rule or a team convention is the best
possible source for "what does *our* shop do" and the worst for "what is true
in general".

A **list** question retrieves differently from a depth question, because it
fails differently. "How do I design a shooter" wants four chunks of the best
shooter page; "what mechanisms are common in FRC" wants one chunk each of six
pages, and an answer naming four of five categories reads as complete, cites
real sources and passes every check below. `wants_enumeration()` in
`app/chat.py` detects the shape (a regex, for the reason given under grounding),
widens the local budget, drops the per-document cap to one, and relaxes the
score floor.

None of that alone finds the missing category, though, and the reason is worth
stating: BM25 scores the shooter page at **zero** for a question that never says
"shooter". A page scoring zero is not near the bottom of the ranking, it is not
in the ranking, so extra slots retrieve nothing. What is knowable without the
word is the *shape of the corpus* — a site documenting five mechanisms puts them
side by side — so `kb.siblings()` walks from a page that did match to the other
documents in its section and adds them at a low score. Present in the context,
visibly not the answer, and exactly the thing whose absence made a five-category
answer name four.

**3. Grounding check** (`app/grounding.py`) — after the answer is written, every
quantity in it is matched against the excerpt it cites. Deliberately not another
model call: an LLM judging an LLM doubles latency, fails in correlated ways with
the thing it is checking, and produces verdicts nobody can reproduce or fix.
String and interval matching is dumb, instant, and auditable. Unmatched numbers
are surfaced to the reader rather than quietly kept.

See `START_HERE.md` for the `kb_ingest.py` commands and the `--kind` table.

## Pocket calibration (`tools/pocket_ref.py`)

The pocketing engine is deterministic geometry, not a learned model, so it can't
be trained — but its constants were chosen by eye off 200–400 mm parts and are
wrong at other scales. `CAL_DEFAULTS` in `app/pocketing.py` names the ten a real
part can disagree with; `calibration()` layers `data/pocket_cal.json` over them.
The defaults exist in exactly one place, so deleting that file leaves an install
byte-identical in behaviour to a fresh one.

`tools/pocket_ref.py` (`list` / `report` / `fit` / `show` / `revert`) measures
reference parts — PNG/JPG silhouettes or STEP — and fits the file.

Design notes worth not undoing:

**Both sides go through one `_metrics()`.** The reference part and the engine's
output on the same outline are measured by the same function, so measurement
bias (rib width pulled up by the rim, bay size as a median over unequal pockets)
cancels out of the difference. A separate, tidier measurement of our own output
would be more accurate and less useful.

**Voids by subtraction, never by filling contours.** Filling an inner contour
ignores what's inside it, so a rib crossing a pocket or a fillet island counts
as removed material. Our side is a raw mask where those islands are plainly
material, and the resulting bias scales with how busy the lattice is — which is
the exact quantity being fitted. Hence `voids = part & ~mat`. Fixing this moved
the reported gap 0.201 → 0.271: the tool had been flattering the engine.

**A blind pocket is invisible in a silhouette.** A pocket floor is near-planar,
so `silhouette_png` projects a 3 mm pocket in 6 mm stock as solid. `measure_tess`
instead sorts planar faces into height levels per side of the back face: the
outermost level on each side is untouched face, intermediate levels are pocket
floors, and faceless regions are through-holes. Per-side grouping matters —
under a single height ordering a plate pocketed from both sides reads the far
face as one very deep pocket.

**Bore vs pocket needs roundness, not size.** FRC 1.125″ bearing bores are
bigger than plenty of legitimate gusset pockets. Bore if circularity > 0.80 and
Ø < 45 mm, or Ø ≤ 12 mm regardless of shape.

**Ratios, not millimetres, everywhere.** Absolute comparison would measure the
already-known (big parts have big pockets) rather than the question (does the
engine hold its proportions as the part grows).

**The average hides the defect**, so `_size_trend()` splits at the median span
and prints both halves unconditionally. Measured on the synthetic set: aggregate
1.1× while large parts ran 1.6× too fine and the small one 1.2× too coarse.

**Flat FEM field (0.5) on purpose.** The constants being fitted are geometric
and reference parts arrive with no load case, so any field would be a per-part
invented load case — folding a guess about loading into a fit about bay size.

**Calibration shifts all three density presets by the same delta**, since the
presets are a promise about the distance between them.

**Bounds and `MIN_GAIN` are not decoration.** Coordinate descent on six points
will run a constant to an absurd value that happens to fit; the bounds are where
"fits the data" stops being evidence. `MIN_GAIN` refuses a move that buys less
than 2% of the remaining gap — the first fit run moved `cell_rib_f` a third of
its range to buy 0.0015 and wrote `8.925` as though that had been measured.

**The classifier's absolute mm² thresholds are excluded from the numeric
search** (piecewise-constant objective on a handful of parts) and surfaced as a
report-only note instead. The PATTERN column is diagnostic only; `_pattern_guess`
can't separate radial from waffle, and nothing in `fit` optimises it.

## Accuracy roadmap (next levers)
1. **Quadratic elements** — swap `ElementTriP1` → `ElementTriP2` in `app/fem.py`
   for smoother, more accurate stress (esp. around holes).
2. **Real load input** — let the user click which holes are fixed and where the
   load is applied, instead of inferring from the load case.
3. **3D solid FEA** — use `pythonocc-core` to tessellate STEP into a real solid,
   mesh with gmsh (tetrahedra), solve 3D elasticity for non-plate parts.
4. **Onshape geometry** — pull tessellated faces + material directly (see
   `app/onshape.py`); switch to OAuth before going public.

## Security
`.env` is git-ignored. Never hardcode keys. Onshape: use **read-documents scope
only**, and **OAuth (not a shared API key)** for any public deployment.
