# HANDOFF — schematic_extractor (Schematic AI Reasoner)

_Complete handoff to resume this project from any point. Last updated: 2026-06-19._

---

## 0. TL;DR (resume in 30 seconds)
Turns **vector** schematic PDFs into a queryable **Components↔Nets** graph (export SPICE / KiCad / JSON), with an LLM as the final tool-calling layer. **No OCR — purely geometric extraction** (deliberate alternative to OCR, which is unreliable on schematics). Code is on branch `feat/wire-symbol-separation`, **committed locally** (latest: `240358e` Streamlit UI, `f22b2c9` single-linkage clustering). Test suite **green (144/144)**. The page-spanning clustering **blob is fixed** (single-linkage on endpoints); a **visual debug UI** now exists to tune it. **Next real bottleneck = pin→net matching (D3)** + clustering over-segmentation, now *visible* instead of masked. End goal: **public portfolio release** (see §10).

---

## 1. Objectives & why

### 1a. Objective
A pipeline that reconstructs the **electrical topology** of a legacy/vector schematic PDF — components, nets, connectivity — as a bipartite graph, exportable to SPICE/KiCad/JSON and queryable in natural language via an LLM tool-calling layer. Ship with a **reproducible demo** and a polished portfolio artifact.

### 1b. Why
- **Portfolio piece at the intersection of AI integration and electronics domain knowledge** — a differentiated, fully-owned project.
- Legacy schematics (service manuals, CAD exports) are locked as pixels/vectors with no machine-readable topology. Bridging them to modern simulation/versioning is a real, underserved problem.
- Strategic value accrues from the build itself (skills + public artifact), independent of adoption.

### 1c. Design choice: geometric, not OCR
Extraction is **purely geometric** (segments, shapes, text spans from the vector layer) — *not* OCR. OCR is unreliable on dense schematics with tiny labels and crossing wires. This is the deliberate complement to the separate `librechat-ingestion-fix` OCR project (different input: scanned/raster PDFs). This project does **not** read raster PDFs and does **not** replace a simulator — it produces a *structural* model, not validated electrical behavior.

### 1d. Track context
Open-source / career-capital track. Ground truth is **auto-derived from KiCad files** (no manual labeling). Kept as a standalone portfolio project. Skills transfer; the public release is the milestone (§10).

---

## 2. Current status (snapshot)
- **Branch:** `feat/wire-symbol-separation` — **committed locally** (push status: local).
- **Commits on branch (newest first):**
  - `240358e` feat: Add Streamlit UI and headless render for graph visualization
  - `f22b2c9` feat(clustering): single-linkage su endpoint, elimina blob WB1
  - `4a26d13` feat(clustering): wire/symbol separation before DBSCAN
  - `afb63f0` fix(d6): scale-aware stub matching and T-junction detection in graph_builder
  - `01817f5` feat(erc): Phase 4 ERC — electrical rule checks on bipartite graph
- **Tests:** **144 passed** (pytest). ruff 0. mypy: **0 on all touched files**; 3 pre-existing `np.ndarray` type-arg errors remain in `classifier.py`/`feature_extractor.py` (already in history, not introduced here — sandbox mypy stricter than the pinned env; §7).
- **Bryston page 0 (real input), adaptive link_dist (~8.6pt):** 63 components · 18 nets · **35 edges** · 34 isolated. No page-spanning blob.
- **Same, link_dist=12pt:** 51 components · 18 nets · 37 edges · 22 isolated (fewer fragments).

---

## 3. The problem & root cause (current focus)
- **Just fixed — clustering blob (WB1):** the old clustering (DBSCAN on segment **midpoints**, single global `eps≈47pt`) chained densely-packed symbol strokes into **one fake page-sized "component"** (177 segments, 794×505pt). Downstream this collapsed connectivity to 9 edges / 3-of-8 isolated. **Root cause:** midpoint density + one global radius cannot separate components that sit close together. **Fix:** single-linkage on **shared endpoints** (how a symbol is actually drawn — touching strokes; wires that would bridge symbols are already removed), with a **data-derived** linkage distance. Blob eliminated; edges 9→35.
- **Newly visible — pin→net (D3) + over-segmentation:** the blob was masking these. Pins are still virtual bbox-corner points (topologically wrong for most parts), so many real components don't attach to a net (34 isolated). Some clusters are 2-segment noise. This is the next bottleneck.
- **Already filtered:** 280 Bezier-arc fragments (`item_type="curve"`, ~2.0pt) are removed by `separate_wires()` before clustering — they are **not** the residual blob cause (a common misdiagnosis; verified).

---

## 4. Architecture map (data flow)
```
vector PDF
  → PyMuPDF extraction (segments, shapes, text spans)     src/core/pdf_parser.py
  → wire/symbol separation (+ Bezier-curve filter)        src/ml/clustering.py: separate_wires()
  → single-linkage clustering on endpoints → symbols      src/ml/clustering.py: cluster()
  → feature extraction (13 features) + classifier         src/ml/feature_extractor.py, classifier.py
  → text association (refs / values → symbols)            src/core/text_associator.py
  → net reconstruction (segment BFS, junctions)           src/core/graph_builder.py
  → bipartite graph Components↔Nets                        src/core/graph_builder.py
  → export SPICE / KiCad / JSON                            src/core/graph_builder.py
  → visual debug overlay (render + Streamlit)              src/ui/render.py, src/ui/app.py
  → LLM tool calling over the graph                        (Phase 5, not built)
Ground truth: KiCad .kicad_sch → coords → auto-labeled training set (no manual labeling)
```

---

## 5. What was built (file by file)
**Core (`src/core/`):**
- `pdf_parser.py` — PyMuPDF vector extraction: `PDFSegment`/`PDFShape`/`PDFTextBlock`, text-span merge (`_extract_text_blocks`, B3 done), EDA value/ref regex (B4/D1), junction circles (B2).
- `graph_builder.py` — bipartite graph: clustering → classify → text-associate → nets (BFS) → pin→net matching → exports (SPICE/KiCad/JSON). **New:** `cluster_eps` param to tune link_dist from the UI.
- `text_associator.py`, `erc.py` (Phase 4 ERC: ISOLATED_COMPONENT, FLOATING_PIN, DANGLING/UNCONNECTED/UNNAMED_NET), `coordinate_system.py`, `logging_config.py`.

**ML (`src/ml/`):**
- `clustering.py` — **rewritten clustering**: `cluster()` now single-linkage union-find on segment endpoints + O(n) spatial grid; `_link_segments()`, `_estimate_link_dist()` (p60 of nearest-other-endpoint distance, data-derived). `separate_wires()` (wire vs symbol + Bezier-curve filter). `_estimate_eps()` kept for back-compat.
- `feature_extractor.py` (13 features), `classifier.py` (`RuleBasedClassifier` active; `ComponentClassifier` RF path kept, untrained).

**UI (`src/ui/`) — new:**
- `render.py` — headless render: PyMuPDF page raster + diagnostic overlay (component bbox **green=connected / red=isolated**, refs, net segments in blue). Pure/testable; `build_overlay()`, `save_overlay()`.
- `app.py` — minimal Streamlit: PDF picker, DPI + `link_dist` sliders (0 = adaptive), live graph metrics + overlay image.

**Tests (`tests/`):** `test_pdf_parser.py`, `test_graph_builder.py`, `test_kicad_net_reconstructor.py`, `test_erc.py`, **`test_clustering_linkage.py` (new, 7)**, **`test_render_overlay.py` (new, 2 smoke)**.

**Data (`data/`, `test_input/`):** `kicad/synthetic/*.kicad_sch`, `ground_truth/` (auto-derived `.cir`/`.net`/`_graph.json`), `test_input/bryston_schematic.pdf` (real input — **license to verify, §13**).

---

## 6. Key decisions (locked) + rationale
- **Geometric extraction, no OCR.** Deliberate (§1c). Reliable on vector schematics; complements the OCR project.
- **Single-linkage on endpoints (not midpoint-DBSCAN).** Endpoints reflect how symbols are drawn; avoids the global-eps page-blob. Verified: midpoint-DBSCAN cannot separate close components at any single eps; endpoint linkage can.
- **Data-derived link_dist (p60 nearest-other-endpoint).** No absolute constants → adapts to any PDF scale (~8.6pt on Bryston). Explicit `eps`/`cluster_eps` overrides it (tests + UI slider).
- **Bezier-curve fragments filtered before clustering** (`item_type="curve"`). They don't carry connectivity.
- **Rule-based classifier active, RF path kept untrained.** Unblocks downstream without a trained model; RF needs KiCad→PDF aligned pairs first.
- **Ground truth auto-derived from KiCad** — no manual labeling.
- **Visual debug harness before further clustering/pin tuning** — tune by looking, not blind.
- **TDD / tooling gates:** pytest + ruff + mypy must stay green.

---

## 7. Test status
- **144 passed** (`python -m pytest -q`). ruff: **All checks passed** on touched files.
- **mypy:** clean on `clustering.py`, `render.py`, `app.py`, `graph_builder.py`. **3 pre-existing errors** remain: `Missing type arguments for generic type "ndarray"` in `feature_extractor.py:29`, `classifier.py:139,192`. They exist in HEAD, are in files not touched here, and reflect a stricter mypy/numpy-stub version in this environment than the pinned one. **Decide:** fix all 3 with `npt.NDArray[...]` (trivial, identical change) to make the whole repo mypy-0, or leave as env-specific.
- Run: `pytest -q` · `mypy src` · `ruff check .` from project root.
- **Gap:** KiCad CLI not installed → synthetic `.kicad_sch` have no matching rendered PDF → no fully automated KiCad→PDF→graph round-trip test yet (§9).

---

## 8. Verified state / demo (2026-06-19)
- **Clustering fix verified on Bryston:** blob (177 seg / 794×505pt) → component-scale clusters (top sizes 15/15/12/10/9/8), edges **9→35**. New unit tests assert single-linkage groups touching strokes, does not chain gapped groups, assigns/drops shapes correctly.
- **Visual overlay generated** (`src/ui/render.py`, headless): Bryston rendered at 200 DPI with component bboxes (green=connected, red=isolated), refs, and net segments — alignment verified correct against the rendered schematic. Two reference PNGs produced (`link_dist` adaptive vs 12pt).
- **Honest limitation:** the graph is **not yet electrically faithful** — 34/63 components isolated because pin→net (D3) is still virtual-bbox-corner based. This is a *structural draft*, not a validated netlist. Don't over-claim connectivity until D3 is real.
- No LLM query layer yet (Phase 5). No polished end-to-end "PASS" demo on multiple public schematics yet (Phase 5/6 + §10).

---

## 9. Remaining work (Definition of Done)
1. ~~Fix clustering blob~~ — **DONE** (`f22b2c9`).
2. ~~Visual debug UI~~ — **DONE** (`240358e`).
3. **Tune link_dist** definitively using the UI (compare p60 vs ~p70/12pt across schematics); set the default.
4. **D3 — real pin positions** from symbol geometry (replace 4 bbox-corner virtual pins). Biggest lever for connectivity; now visible in the overlay.
5. **Reduce over-segmentation** (drop 2-segment noise clusters / merge split symbol bodies).
6. **Phase 5 — LLM tool calling** (`get_neighbors`, `get_path`, `get_net_components`) + 20-question topology benchmark (doubles as regression test).
7. **Phase 6 — UI polish** beyond the debug harness; portfolio framing.
8. **B1 ML upgrade** — train RF on KiCad→PDF pairs (needs KiCad CLI); rule-based stays fallback.
9. **Public-release gate** — §10.

**DoD (target = public portfolio release):**
- ✅ vector PDF → bipartite graph pipeline, tests green.
- ✅ clustering produces component-scale clusters (no page-blob).
- ✅ visual debug harness.
- ⬜ electrically meaningful connectivity (D3 real, isolated≈0 on a clean schematic).
- ⬜ LLM topology queries + benchmark.
- ⬜ reproducible public demo on **synthetic** (and license-cleared) schematics.
- ⬜ public repo + README + sample data.

---

## 10. Public-release checklist (the irreversible gate — do carefully)
Before making the repo public:
- [ ] **Bryston schematic license VERIFIED** (§13). If not redistributable, **remove it from public fixtures** and ship only synthetic `.kicad_sch`-derived samples.
- [ ] Final scan: no proprietary/employer data in code/tests/fixtures/demo — synthetic/public only.
- [ ] LICENSE present (pyproject declares MIT) + attribution intact.
- [ ] README: problem → approach (geometric, no-OCR) → quickstart → demo → honest limitations (structural model, not a simulator).
- [ ] Reproducible demo runs from a clean clone on synthetic input.
- [ ] Tooling green from clean clone (pytest/ruff/mypy), incl. the 3 ndarray fixes (§7) if going for mypy-0.
- [ ] Screenshots/GIF of the Streamlit overlay for the portfolio.

Then: create public repo (account **D4NGYY**), push, write the portfolio writeup.

---

## 11. Environment & tooling
- **OS:** Windows 11. **Python 3.12.** Project venv at `schematic_extractor/.venv` (Windows; Linux sandbox uses its own interpreter).
- **Deps** (`pyproject.toml`): pymupdf, networkx, numpy, scikit-learn, scikit-image, scipy, matplotlib, structlog, pydantic, typer, pandas, **pillow**, pyyaml. Dev/extra: pytest, ruff, mypy, black, **streamlit**, watchdog. `[project.scripts] schematic-extractor = "src.ui.app:main"`.
- **Run the UI:** `pip install streamlit` → `streamlit run src/ui/app.py` (sliders: DPI, link_dist; 0 = adaptive).
- **KiCad CLI:** **not installed** → no auto KiCad→PDF render → no full round-trip test yet (§9).
- **Filesystem quirk (this Cowork session):** file-tool writes (Windows path) and the Linux sandbox mount fell out of sync once (truncated/stale reads); resolved by writing via the sandbox shell. git `.git/*.lock` files could not be removed from the sandbox (permission) and blocked commits — cleared on the Windows side.

---

## 12. Artifacts & locations
- Working branch: `feat/wire-symbol-separation` (local).
- Source: `src/core/`, `src/ml/`, `src/ui/`.
- Tests: `tests/` (144 passing).
- Real input: `test_input/bryston_schematic.pdf` (license to verify).
- Synthetic + ground truth: `data/kicad/synthetic/`, `data/ground_truth/`.
- Reference overlays (this session): produced via `src/ui/render.py:save_overlay`.
- This handoff: `HANDOFF.md`. Roadmap/next: `TODO.md`. Prior handoff archived under `_archive/`.

---

## 13. Guardrails (non-negotiable)
- **Bryston schematic license is UNVERIFIED.** Treat as **not redistributable until confirmed**. Keep it local; do **not** put it in a public repo's fixtures. Public demo/test data = **synthetic only** (auto-generated from KiCad) until cleared.
- **No OCR** — geometric extraction only (design choice, not a TODO).
- **No manual labeling** — ground truth auto-derived from KiCad.
- **Honest claims** — this is a structural model, not a validated simulator; "assist, not authority". Don't claim electrical correctness while D3 (pins) is virtual.
- Keep tooling (pytest/ruff/mypy) green; update this HANDOFF and TODO.md after each session.

---

## 14. How to resume (first actions for whoever picks this up)
1. Read §0–3 for goal, status, and the current bottleneck.
2. Check out `feat/wire-symbol-separation`; run `pytest -q` → expect 144 passed.
3. Launch the debug UI: `streamlit run src/ui/app.py`; load Bryston; move the `link_dist` slider and watch components/edges/isolated. Red boxes = the D3 pin→net problem to attack next.
4. Pick up at §9 step 3 or 4: finalize `link_dist`, then make pin positions real (D3) — the highest-leverage next move, now visible in the overlay.
