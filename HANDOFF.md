# HANDOFF — schematic_extractor (Schematic AI Reasoner)

_Complete handoff to resume this project from any point. Last updated: 2026-06-20._

---

## 0. TL;DR (resume in 30 seconds)
Turns schematic PDFs into a queryable **Components↔Nets** graph (export SPICE / KiCad / JSON), with an LLM as the final tool-calling layer. Extraction is geometric (vector layer); an **OCR fallback** (RapidOCR, optional `[ocr]` extra) kicks in for text-less CAD-export PDFs. Branch `feat/wire-symbol-separation`, committed locally. Test suite **green (210 passed)**. Phase 5 (LLM tool calling) **debugged end-to-end against real Ollama** — qwen2.5:7b winner (25/25), single default (§15). Net connectivity improved over the last sessions (§3): scale-based `pin_tol` fix (micro **F1 0.21 → 0.36**), `#PWR` GT honesty, `label_tol_factor=6` net merge. **OCR fallback this session** unblocked text-less PDFs: **arduino_nano F1 0.03 → 0.16** (refs 0 → 12). **Root-caused the F1 ceiling (§3):** it is a **ref→cluster collision** (45 GT ref-texts collapse onto 24 clusters → 24 refs lost); two targeted fixes both regress F1 — the real cure is geometric re-segmentation (large). End goal: **public portfolio release** (§10).

---

## 1. Objectives & why

### 1a. Objective
A pipeline that reconstructs the **electrical topology** of a legacy/vector schematic PDF — components, nets, connectivity — as a bipartite graph, exportable to SPICE/KiCad/JSON and queryable in natural language via an LLM tool-calling layer. Ship with a **reproducible demo** and a polished portfolio artifact.

### 1b. Why
- **Portfolio piece at the intersection of AI integration and electronics domain knowledge** — a differentiated, fully-owned project.
- Legacy schematics (service manuals, CAD exports) are locked as pixels/vectors with no machine-readable topology. Bridging them to modern simulation/versioning is a real, underserved problem.
- Strategic value accrues from the build itself (skills + public artifact), independent of adoption.

### 1c. Design choice: geometric first, OCR fallback
Extraction is **geometric by default** (segments, shapes, text spans from the vector layer). For PDFs that ship **no text layer** (CAD exports with outlined fonts, e.g. arduino_nano), a **RapidOCR fallback** rasterizes the page and recovers ref/label text *with bounding boxes* so association still works (optional `[ocr]` extra, lazy import, fires only when the native text layer is empty — see §3/§17). This project does **not** replace a simulator — it produces a *structural* model, not validated electrical behavior. _(Historical note: earlier the project was strictly no-OCR; the OCR fallback was added deliberately once the user provided a working RapidOCR extractor to adapt.)_

### 1d. Track context
Open-source / career-capital track. Ground truth is **auto-derived from KiCad files** (no manual labeling). The public release is the milestone (§10).

---

## 2. Current status (snapshot)
- **Branch:** `feat/wire-symbol-separation` — committed locally (push status: local).
- **Pipeline (latest on branch):** V7 text-guided clustering, V5/V6 T-/dot-junction fixes, KiCad GT parser, Phase 5 LLM layer, GT honesty + label-based net merging + scale-based pin_tol, **OCR fallback for text-less PDFs** (this session).
- **Tests:** **210 passed** (`uv run python -m pytest -q`). ruff + mypy clean on `src/core/ocr_fallback.py`, `src/core/pdf_parser.py`, `src/core/graph_builder.py`, `src/llm/`. (Legacy lint in `app.py`/`kicad_gt_reader.py` left untouched.)
- **F1 (GT-measured, `PYTHONHASHSEED=0` for stable greedy tie-break):** arduino_micro **0.356** (overlap 24/48), arduino_nano **0.159** (overlap 10/34, was 0.03 pre-OCR).
- **Bryston page 0 graph:** 13 components · 119 nets · 53 edges · **0 isolated**. Adaptive link_dist ≈ 8.25pt.

---

## 3. The problem & root cause (current focus)
- **Resolved — clustering blob (WB1):** single-linkage on endpoints eliminated the page-spanning blob.
- **Resolved — T-junction/Dot-junction (V5/V6):** wires at T-/dot-junctions merge into degree-3+ nets.
- **Resolved — over-segmentation (V7):** `_text_guided_merge` uses ref-designator texts as gravity wells (Arduino Micro 193→112 components, F1 0.08→0.21).
- **Resolved — LLM tool layer broken on real graphs (this session, §15):** `GraphContext` was reading the wrong node/edge schema; fixed + re-benchmarked.
- **Net connectivity — scale-based pin_tol fix (measured on GT F1).** `_connect_pins_to_nets` used `pin_tol = 3×wire_tol` (≈4 on micro), ~6× smaller than real pin→net gaps. Fixed: `pin_tol = max(3×wire_tol, scale × pin_tol_factor)`, `pin_tol_factor=2.0`. arduino_micro **F1 0.209 → 0.356**, Bryston isolated 3 → 0. `label_tol_factor=6.0` then consolidated GND/VCC (netmaxdeg 6→12, F1-neutral).
- **OCR fallback (this session) — unblocked text-less PDFs.** arduino_nano had `texts=0` (outlined fonts) → refs 0/34. `src/core/ocr_fallback.py` (RapidOCR, keeps bounding boxes) fires when the native text layer < 16 chars. **nano: texts 0→281, refs 0→12, overlap 3→10/34, F1 0.030→0.159.** micro unchanged (has a text layer). See §17.
- **ROOT CAUSE of the F1 ceiling — ref→cluster collision (diagnosed this session).** The metric is gated on `overlap_refs` (micro 24/48). Measured: **45 GT ref-texts ARE extracted**, but `_nearest_cluster` binds each to the nearest cluster center and `_create_component_node` keeps **only one ref per cluster** → they collapse onto 24 distinct clusters, **losing 24 GT refs**. Two failure modes: (A) under-segmentation — adjacent components fused into one giant cluster (R5/R6/R7/R8 all at dist 6.7 in one 54-segment cluster; TP1+TP2 together); (B) refs with no real cluster falling onto tiny noise fragments.
  - **Two targeted fixes tried, both regress F1 (reverted):** (1) one node per distinct ref (share cluster) → recall recovers (overlap 24→41) but **fp explodes 13→58** (co-clustered comps share nets) → F1 0.356→0.352; (2) + partition pins by ref-pos → fp drops 58→8 but **tp collapses 50→37** → F1 0.356→**0.335**. You cannot rebuild connectivity from a fused blob without true **geometric segment-level re-segmentation** (split a cluster's *segments* per ref, not just its pins). That is a substantial `src/ml/clustering.py` rewrite with uncertain payoff — **not yet attempted.** See TODO P4 item 5.
- **Measured-dead levers (do not retry):** pin over-generation (fp already ~13, not 988), aggressive `label_tol_factor` (collapses distinct nets, F1 down), higher OCR dpi (300→500 flat). Reproduce overlay: `save_overlay(...)` in `src/ui/render.py`.

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
  → bipartite graph Components↔Nets (bipartite=0/1)        src/core/graph_builder.py
  → export SPICE / KiCad / JSON                            src/core/graph_builder.py
  → visual debug overlay (render + Streamlit)              src/ui/render.py, src/ui/app.py
  → LLM tool calling over the graph                        src/llm/agent.py, src/llm/tools.py, src/cli/query.py
Ground truth: KiCad .kicad_sch → coords → auto-labeled training set (no manual labeling)
```

---

## 5. LLM layer (Phase 5) — file by file
- `src/llm/tools.py` — `GraphContext` wraps the bipartite graph and exposes **7 tools**:
  `get_neighbors`, `get_path`, `get_net_components`, `find_isolated`,
  `get_component_info`, `search_by_value`, **`get_nets_summary(min_components)`** (added this
  session for "nets connecting ≥N components"). Node classification accepts both the
  builder schema (`bipartite=0/1`, edges `pin_id`) and the test-fixture schema (`type=`, `pin=`).
- `src/llm/agent.py` — `LLMClient`/`OllamaClient`/`MockClient`, `SchematicAgent` agent loop
  (native tool_calls + hardened ReAct text fallback). `DEFAULT_MODEL` constant (benchmark winner)
  is the single source of truth. `_execute_tool` rejects non-dict args gracefully.
- `src/cli/query.py` — Typer CLI (`schematic-extractor query "..." --pdf ... [--mock] [--model ...]`),
  default model = `DEFAULT_MODEL`.
- `src/ui/app.py` — Streamlit chat; model selector defaults to `DEFAULT_MODEL`.

**Benchmark harness:** `diagnosi_d3/benchmark_llm.py` (in-process, deterministic tool-gated
scoring) → `diagnosi_d3/benchmark_llm_results.json`. Manual/report: **`TEST_MANUAL.md`**.

---

## 6. Key decisions (locked) + rationale
- **Geometric extraction, no OCR.**
- **Single-linkage on endpoints** (not midpoint-DBSCAN) — avoids the page-blob.
- **Data-derived link_dist** (p60 nearest-other-endpoint) — scale-adaptive.
- **Rule-based classifier active, RF path untrained.**
- **Ground truth auto-derived from KiCad.**
- **LLM default = qwen2.5:7b** — winner of the honest re-benchmark (§15); centralised in `DEFAULT_MODEL`.
- **Dual-schema `GraphContext` + name-gated ReAct parser** — robust to real graphs and to the
  varied text shapes local models emit, without coupling to one backend.
- **TDD / tooling gates:** pytest + ruff + mypy stay green.

---

## 7. Test status
- **210 passed** (`uv run python -m pytest -q`). ruff + mypy clean on `src/core/ocr_fallback.py`,
  `src/core/pdf_parser.py`, `src/core/graph_builder.py`, `src/llm/`.
- New tests this session: **OCR fallback** — 11 tests in `tests/test_ocr_fallback.py` (rows→PDF-coord
  blocks, confidence filter, char-count threshold, VectorExtractor hook fires only when text layer
  empty — all with a fake engine, no ONNX). Plus label-tol semantics in `tests/test_graph_builder.py`.
- Note: legacy lint left untouched in `src/ui/app.py` and `src/core/kicad_gt_reader.py`.

---

## 8. Verified state / demo (2026-06-20)
- **LLM end-to-end works on real Ollama** (default qwen2.5), e.g. multi-step:
  `"Quali componenti sono isolati e quali net collegano almeno 2 componenti?"` →
  `RB14, R45, DX7` + `Net-6 (WB1,PR2), Net-46 (WB1,RB10), Net-81 (RF1,R37)` — correct on real data.
- **Benchmark (3 models × 5 Bryston queries, tool-gated):** qwen2.5 **25/25 (5/5, 3.24s)**,
  llama3.1 21/25 (4/5, 3.67s), mistral 20/25 (4/5, 4.64s). See `TEST_MANUAL.md` §4.
- **Net connectivity F1 (GT-measured; use `PYTHONHASHSEED=0` — the greedy net-mapping tie-break is hash-order-sensitive, ±1 tp otherwise):** arduino_micro **0.356** (overlap 24/48), arduino_nano **0.159** (overlap 10/34, was 0.030 before the OCR fallback). Run `PYTHONHASHSEED=0 uv run python diagnosi_d3/true_f1_validation.py`.
- No polished public end-to-end demo on multiple schematics yet (Phase 6 + §10).

---

## 9. Remaining work (Definition of Done)
1. ~~Fix clustering blob~~ / ~~Visual debug UI~~ — **DONE**.
2. ~~Phase 5 LLM tool calling + real-Ollama benchmark~~ — **DONE** (this session).
3. **Tune link_dist** definitively via the UI; set the default.
4. ~~Net connectivity — scale-based `pin_tol`~~ + ~~`label_tol_factor`~~ + ~~OCR fallback (nano 0.03→0.16)~~ — **DONE**.
5. **Geometric re-segmentation (THE remaining F1 lever, hard):** split under-segmented clusters per ref at the *segment* level so each GT designator becomes its own component with its own pins/nets. Root-caused + two simple fixes proven to regress F1 (§3, TODO P4 item 5). Big `src/ml/clustering.py` rewrite, uncertain payoff — decide before investing.
6. **Phase 6 — UI polish** beyond the debug harness; portfolio framing.
7. **B1 ML upgrade** — train RF on KiCad→PDF pairs (needs KiCad CLI).
8. **Public-release gate** — §10.

**DoD (target = public portfolio release):**
- ✅ vector PDF → bipartite graph pipeline, tests green.
- ✅ clustering produces component-scale clusters (no page-blob).
- ✅ visual debug harness.
- ✅ LLM topology queries working end-to-end + scored benchmark.
- 🟡 electrically meaningful connectivity — improved (micro F1 0.36, nano 0.16 via OCR, Bryston isolated 0) but capped by the ref→cluster collision (§3); geometric re-segmentation remains.
- ⬜ reproducible public demo on synthetic (and license-cleared) schematics.
- ⬜ public repo + README + sample data.

---

## 10. Public-release checklist (irreversible gate — do carefully)
- [ ] **Bryston schematic license VERIFIED** (§13). If not redistributable, remove from public fixtures; ship only synthetic samples.
- [ ] Final scan: no proprietary/employer data in code/tests/fixtures/demo.
- [ ] LICENSE present (pyproject declares MIT) + attribution intact.
- [ ] README: problem → approach (geometric, no-OCR) → quickstart → demo → honest limitations.
- [ ] Reproducible demo from a clean clone on synthetic input.
- [ ] Tooling green from clean clone (pytest/ruff/mypy).
- [ ] Screenshots/GIF of the Streamlit overlay + LLM chat for the portfolio.

Then: create public repo (account **D4NGYY**), push, write the portfolio writeup.

---

## 11. Environment & tooling
- **OS:** Windows 11. **Python 3.12.** uv-managed venv.
- **Deps:** pymupdf, networkx, numpy, scikit-learn, scipy, matplotlib, structlog, pydantic, typer, pandas, pillow, pyyaml, **openai** (Ollama OpenAI-compat client). Dev: pytest, ruff, mypy, black, streamlit, watchdog. `[project.scripts] schematic-extractor = "src.ui.app:main"` and CLI via `src.cli.query`.
- **LLM runtime:** Ollama on `http://localhost:11434`. Models: `qwen2.5:7b-instruct-q4_K_M` (default), `mistral:7b-instruct-v0.3-q4_K_M`, `llama3.1:8b-instruct-q4_K_M`. Target HW: 3070 Ti 8 GB.
- **KiCad CLI:** not installed → no auto KiCad→PDF render → no full round-trip test yet.

---

## 12. Artifacts & locations
- Working branch: `feat/wire-symbol-separation` (local).
- Source: `src/core/`, `src/ml/`, `src/ui/`, `src/llm/`, `src/cli/`.
- Tests: `tests/` (210 passing).
- Real input: `test_input/bryston_schematic.pdf` (license to verify).
- Synthetic + ground truth: `data/kicad/synthetic/`, `data/ground_truth/`.
- LLM benchmark: `diagnosi_d3/benchmark_llm.py` + `diagnosi_d3/benchmark_llm_results.json`.
- This handoff: `HANDOFF.md`. LLM manual/report: `TEST_MANUAL.md`. Roadmap: `TODO.md`.

---

## 13. Guardrails (non-negotiable)
- **Bryston schematic license UNVERIFIED** — treat as not redistributable; keep local; public demo data = synthetic only until cleared.
- **OCR only as fallback** — geometric extraction is primary; RapidOCR fires *only* when a page has no usable text layer (optional `[ocr]` extra). Never OCR a page that already has vector text.
- **No manual labeling** — ground truth auto-derived from KiCad.
- **Honest claims** — structural model, not a validated simulator. The LLM answers strictly from tool results; do not let it invent components/nets.
- Keep tooling (pytest/ruff/mypy) green; update HANDOFF + TODO after each session.

---

## 14. How to resume (first actions)
1. Read §0–3 for goal, status, and the current bottleneck (ref→cluster collision).
2. `pytest -q` → expect **210 passed**. For LLM: `ollama serve` + pull the 3 models (§11). For OCR: `uv sync --extra ocr --extra dev`.
3. LLM: `uv run schematic-extractor query "Quali componenti sono isolati?" --pdf test_input/bryston_schematic.pdf` → expect `RB14, R45, DX7`.
4. Net-connectivity F1: `PYTHONHASHSEED=0 uv run python diagnosi_d3/true_f1_validation.py` → expect micro **0.356** / nano **0.159**. **The only remaining F1 lever is geometric re-segmentation** (§3, TODO P4 item 5) — a hard `src/ml/clustering.py` rewrite; two simpler fixes are already proven to regress F1, and pin-over-generation / label-tol / OCR-dpi are measured-dead. Decide whether to invest before coding.

---

## 15. Phase 5 LLM — journey log
**Phase 5 implementation (earlier):** 7 tools, `SchematicAgent` with native tool_calls + ReAct fallback, Typer CLI, Streamlit chat.

**Phase 5 end-to-end debug (2026-06-20, this session):**
- **Discovery:** the prior "Qwen 5/5" benchmark was run against a `GraphContext` that read `data["type"]`/`edge["pin"]`, but `graph_builder` emits `bipartite=0/1`/`pin_id`. On real graphs the tools returned empty/"not found" and the models **hallucinated** plausible answers. Unit tests passed only because the fixtures used the old schema.
- **Fixes (all in `src/llm/`):** dual-schema `GraphContext` (`_is_component`/`_is_net`/`_net_name`, `pin_id` first); `_execute_tool` rejects non-dict args; `get_nets_summary` coerces `min_components` to int; added `get_nets_summary` tool; model-agnostic system prompt; ReAct parser hardened (envelope JSON, prefixed/bare object, positional — all name-gated). Centralised `DEFAULT_MODEL`.
- **Honest re-benchmark (Bryston, 5 queries adapted to real refs WB1/R37/RF1, tool-gated scoring):**
  qwen2.5 **25/25 (5/5)**, llama3.1 21/25 (4/5), mistral 20/25 (4/5). Winner **qwen2.5:7b**.
  Discriminator: on "components connected to WB1", mistral/llama wrongly call `get_net_components` (WB1 is a component); qwen disambiguates.
- Full details + ground truth + reproduction: **`TEST_MANUAL.md`**.

---

## 16. Phase 5 — How to use
```bash
ollama serve
ollama pull qwen2.5:7b-instruct-q4_K_M     # default, ~4.7GB

# CLI (real Ollama, default Qwen 2.5)
uv run schematic-extractor query "quali componenti sono isolati?" --pdf test_input/bryston_schematic.pdf
# CLI (mock, no Ollama)
uv run schematic-extractor query "..." --pdf test_input/bryston_schematic.pdf --mock

# Scored multi-model benchmark
uv run python diagnosi_d3/benchmark_llm.py

# Streamlit UI (PDF picker, link_dist slider, Chat tab)
streamlit run src/ui/app.py
```

---

## 17. OCR fallback (Phase: text-less PDFs)
- **Why:** CAD-export PDFs with outlined fonts (arduino_nano) have no text layer → `get_text` returns nothing → ref/label association starves (refs 0/34).
- **What:** `src/core/ocr_fallback.py` — rasterizes the page (`fitz`, dpi=300) and runs **RapidOCR**, emitting `PDFTextBlock` with **bounding boxes** in PDF coords (pixel ÷ scale). Adapted from the librechat `rag_api/app/utils/ocr.py` approach, but that module discarded `box[0]`; here the box is kept (positions are what bind a ref to its component). Engine is injectable → unit-testable without ONNX.
- **Hook:** `VectorExtractor._maybe_ocr` runs only when the native text layer has < 16 non-whitespace chars, so text-rich pages (micro: 498 blocks) are untouched. Lazy import; degrades gracefully if the `[ocr]` extra isn't installed.
- **Dependency:** `rapidocr-onnxruntime` as optional `[ocr]` extra (`uv sync --extra ocr`). Pulls onnxruntime + opencv.
- **Result:** nano texts 0→281, refs 0→12, overlap 3→10/34, **F1 0.030→0.159**. Residual: OCR misreads in labels (V↔U, 7↔Z) — but these are not the F1 blocker (the blocker is the ref→cluster collision, §3). Higher dpi (400/500) measured flat.
