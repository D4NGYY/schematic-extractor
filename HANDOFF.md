# HANDOFF — schematic_extractor (Schematic AI Reasoner)

_Complete handoff to resume this project from any point. Last updated: 2026-06-20._

---

## 0. TL;DR (resume in 30 seconds)
Turns **vector** schematic PDFs into a queryable **Components↔Nets** graph (export SPICE / KiCad / JSON), with an LLM as the final tool-calling layer. **No OCR — purely geometric extraction.** Branch `feat/wire-symbol-separation`, committed locally. Test suite **green (192 passed)**. Phase 5 (LLM tool calling) has now been **debugged end-to-end against real Ollama** — see §15. The earlier "Qwen 5/5" benchmark in prior handoffs was **invalid** (it ran against a broken `GraphContext` returning empty data; models hallucinated). The schema bug is fixed, the benchmark re-run honestly, and **qwen2.5:7b confirmed as winner (25/25, 5/5, fastest)** and wired as the single default. **Next real bottleneck remains pin→net matching (D3)** + clustering over-segmentation. End goal: **public portfolio release** (§10).

---

## 1. Objectives & why

### 1a. Objective
A pipeline that reconstructs the **electrical topology** of a legacy/vector schematic PDF — components, nets, connectivity — as a bipartite graph, exportable to SPICE/KiCad/JSON and queryable in natural language via an LLM tool-calling layer. Ship with a **reproducible demo** and a polished portfolio artifact.

### 1b. Why
- **Portfolio piece at the intersection of AI integration and electronics domain knowledge** — a differentiated, fully-owned project.
- Legacy schematics (service manuals, CAD exports) are locked as pixels/vectors with no machine-readable topology. Bridging them to modern simulation/versioning is a real, underserved problem.
- Strategic value accrues from the build itself (skills + public artifact), independent of adoption.

### 1c. Design choice: geometric, not OCR
Extraction is **purely geometric** (segments, shapes, text spans from the vector layer) — *not* OCR. OCR is unreliable on dense schematics with tiny labels and crossing wires. This project does **not** read raster PDFs and does **not** replace a simulator — it produces a *structural* model, not validated electrical behavior.

### 1d. Track context
Open-source / career-capital track. Ground truth is **auto-derived from KiCad files** (no manual labeling). The public release is the milestone (§10).

---

## 2. Current status (snapshot)
- **Branch:** `feat/wire-symbol-separation` — committed locally (push status: local).
- **Pipeline (latest on branch):** V7 text-guided clustering, V5/V6 T-/dot-junction fixes, KiCad GT parser, Phase 5 LLM layer, Phase 5 end-to-end Ollama debug (this session).
- **Tests:** **192 passed** (`uv run python -m pytest -q`). ruff clean on touched files; mypy clean on `src/llm/`, `src/cli/query.py`. (3 pre-existing `np.ndarray` type-arg notes in `classifier.py`/`feature_extractor.py` only under stricter sandbox stubs.)
- **Bryston page 0 graph:** 13 components · 125 nets · 44 edges · 3 isolated (RB14, R45, DX7). Adaptive link_dist ≈ 8.25pt.

---

## 3. The problem & root cause (current focus)
- **Resolved — clustering blob (WB1):** single-linkage on endpoints eliminated the page-spanning blob.
- **Resolved — T-junction/Dot-junction (V5/V6):** wires at T-/dot-junctions merge into degree-3+ nets.
- **Resolved — over-segmentation (V7):** `_text_guided_merge` uses ref-designator texts as gravity wells (Arduino Micro 193→112 components, F1 0.08→0.21).
- **Resolved — LLM tool layer broken on real graphs (this session, §15):** `GraphContext` was reading the wrong node/edge schema; fixed + re-benchmarked.
- **Open bottleneck — net connectivity (re-diagnosed 2026-06-20 via visual overlay + GT F1).** NOT "virtual pins" (stale). Extracted nets are shattered (arduino_micro pin-connection 16%, max net degree 2). Disproven by experiment: `separate_wires` threshold is not the lever (no gain, worsens Bryston blob). **Dominant lever (next): power symbols (GND/+5V/VCC) are clustered as components (isolated red boxes in the overlay) instead of net anchors — GND being the biggest net, it shatters into dozens of isolated boxes. KiCad GT anchors nets via `#PWR` by name.** Done this session: GT honesty (`#PWR` excluded) + label-based net merging (`_merge_nets_by_label`, mirrors GT same-name merge) — correct & tested but currently under-fires because GND labels sit on power symbols, not wires. Remaining: (1) treat power symbols as named net anchors → feeds the label-merge; (2) pin over-generation (`select_pins` returns symbol-internal/text free-endpoints); (3) noisy `net_label` association (grabs part numbers). Second independent lever: ref-recovery (nano 3/34 GT refs). Reproduce overlay: `save_overlay(...)` in `src/ui/render.py`. See TODO P4.

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
- **192 passed** (`uv run python -m pytest -q`). ruff clean on `src/llm/`, `src/cli/query.py`,
  `diagnosi_d3/benchmark_llm.py`; mypy clean on `src/llm/` + `src/cli/query.py`.
- New tests this session: real-`bipartite`-schema GraphContext + `get_nets_summary`
  (`tests/test_llm_tools.py`); ReAct parser shapes + non-dict guard + no-false-positive
  (`tests/test_llm_agent.py`).
- Note: `src/ui/app.py` still carries pre-existing trailing-whitespace ruff warnings (untouched legacy).

---

## 8. Verified state / demo (2026-06-20)
- **LLM end-to-end works on real Ollama** (default qwen2.5), e.g. multi-step:
  `"Quali componenti sono isolati e quali net collegano almeno 2 componenti?"` →
  `RB14, R45, DX7` + `Net-6 (WB1,PR2), Net-46 (WB1,RB10), Net-81 (RF1,R37)` — correct on real data.
- **Benchmark (3 models × 5 Bryston queries, tool-gated):** qwen2.5 **25/25 (5/5, 3.24s)**,
  llama3.1 21/25 (4/5, 3.67s), mistral 20/25 (4/5, 4.64s). See `TEST_MANUAL.md` §4.
- No polished public end-to-end demo on multiple schematics yet (Phase 6 + §10).

---

## 9. Remaining work (Definition of Done)
1. ~~Fix clustering blob~~ / ~~Visual debug UI~~ — **DONE**.
2. ~~Phase 5 LLM tool calling + real-Ollama benchmark~~ — **DONE** (this session).
3. **Tune link_dist** definitively via the UI; set the default.
4. **D3 — real pin positions** from symbol geometry (replace 4 bbox-corner virtual pins). Biggest connectivity lever.
5. **Reduce over-segmentation** (drop 2-segment noise clusters / merge split symbol bodies).
6. **Phase 6 — UI polish** beyond the debug harness; portfolio framing.
7. **B1 ML upgrade** — train RF on KiCad→PDF pairs (needs KiCad CLI).
8. **Public-release gate** — §10.

**DoD (target = public portfolio release):**
- ✅ vector PDF → bipartite graph pipeline, tests green.
- ✅ clustering produces component-scale clusters (no page-blob).
- ✅ visual debug harness.
- ✅ LLM topology queries working end-to-end + scored benchmark.
- ⬜ electrically meaningful connectivity (D3 real, isolated≈0 on a clean schematic).
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
- Tests: `tests/` (192 passing).
- Real input: `test_input/bryston_schematic.pdf` (license to verify).
- Synthetic + ground truth: `data/kicad/synthetic/`, `data/ground_truth/`.
- LLM benchmark: `diagnosi_d3/benchmark_llm.py` + `diagnosi_d3/benchmark_llm_results.json`.
- This handoff: `HANDOFF.md`. LLM manual/report: `TEST_MANUAL.md`. Roadmap: `TODO.md`.

---

## 13. Guardrails (non-negotiable)
- **Bryston schematic license UNVERIFIED** — treat as not redistributable; keep local; public demo data = synthetic only until cleared.
- **No OCR** — geometric extraction only.
- **No manual labeling** — ground truth auto-derived from KiCad.
- **Honest claims** — structural model, not a validated simulator. The LLM answers strictly from tool results; do not let it invent components/nets.
- Keep tooling (pytest/ruff/mypy) green; update HANDOFF + TODO after each session.

---

## 14. How to resume (first actions)
1. Read §0–3 for goal, status, and the current bottleneck.
2. `pytest -q` → expect 192 passed. `ollama serve` + pull the 3 models (§11).
3. LLM: `uv run schematic-extractor query "Quali componenti sono isolati?" --pdf test_input/bryston_schematic.pdf` → expect `RB14, R45, DX7`. Re-run the benchmark with `uv run python diagnosi_d3/benchmark_llm.py`.
4. Highest-leverage next move: **D3 real pin positions** (§9.4) — visible in the Streamlit overlay (red boxes = isolated). This is what limits connectivity, not the LLM layer.

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
