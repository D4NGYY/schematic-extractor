# Schematic AI Reasoner

Reconstruct the **electrical topology** of a vector-PDF schematic — components, pins, and the nets that connect them — as a queryable **Components↔Nets** graph, exportable to SPICE / KiCad / JSON and answerable in natural language through an LLM tool-calling layer.

Legacy schematics (service manuals, CAD exports) are locked as pixels or vector strokes with **no machine-readable connectivity**. This project bridges them to the modern ecosystem: from a flat PDF to a graph you can simulate, version, and ask questions about.

> **What it is / isn't.** It produces a *structural* model (which pin connects to which net), not a validated electrical simulation. It's a high-quality first-pass reconstruction + an assistant for exploring legacy schematics — not an authoritative netlist you'd SPICE without review.

---

## Results (honest, measured)

Net-connectivity is scored with an F1 over component↔net memberships (did each component's pins land on the correct nets), against ground truth **auto-derived from KiCad** (no manual labeling).

| Configuration | Net-connectivity F1 | Notes |
|---|---|---|
| Geometric baseline | **0.42** | pure vector extraction (32 boards, same-board) |
| **+ component detector (hybrid)** | **0.56** | +0.13; detector wins ~20/32 boards, geometric fallback never regresses |
| **In-scope boards only** | **~0.71** | excluding ultra-dense / bus-dominated boards (density gate) |
| Clean analog/mixed boards | **0.80–0.89** | e.g. rectifier, sallen_key, complex_hierarchy, esvideo |

**Read the number honestly:** F1 ~0.6 means roughly 6 of 10 component→net connections are reconstructed correctly — a reliable *draft / assistant*, not a drop-in netlist. On normal-density schematics it's 0.7–0.9; on ultra-dense legacy digital (memory arrays, large buses) it degrades, and the pipeline **flags those as low-confidence** rather than returning unreliable output.

### Measurement-regime caveat (important)
All F1 figures and the detector/color gains are measured on **KiCad-rendered PDFs**, which is also the only regime we can auto-label and evaluate. The project's *stated target* is **legacy / CAD monochrome** schematics, where the color signal is inert and the detector faces an untested domain shift. **Performance on true legacy scans is unverified and expected to be lower.** Do not read 0.71 as a general-domain number.

---

## The finished system

A complete loop: **load a schematic → extract the topology → ask questions about it.**

1. **Load** — upload any vector PDF in the Streamlit UI (or pass `--pdf` to the CLI). Demo schematics are available in-repo.
2. **Extract** — the pipeline runs the **YOLO component detector** (hybrid, with automatic geometric fallback) → builds the bipartite Components↔Nets graph.
3. **Scope** — a density gate (`scope.py`) tells you up front whether the board is within the supported envelope. Ultra-dense boards are flagged low-confidence instead of returning unreliable topology.
4. **Ask** — an LLM tool-calling layer (default: local Ollama `qwen2.5:7b`, 25/25 on a scored benchmark) answers questions through 7 graph tools: isolated components, neighbors, paths, net summaries, value search, etc.

The detector weights (~100 MB) are **not in the repo** — train locally (`scripts/train_detector.py`) or download the release asset. When absent the app silently falls back to the geometric pipeline (F1 ~0.42 vs ~0.56 with the detector).

---

## How it works

```
vector PDF
  -> PyMuPDF extraction (segments, shapes, text spans, stroke color)   src/core/pdf_parser.py
  -> wire/symbol separation (geometry; optional color-aware)           src/ml/clustering.py
  -> single-linkage clustering on shared endpoints -> symbols          src/ml/clustering.py
  -> OCR fallback (RapidOCR) when the page has no text layer           src/core/ocr_fallback.py
  -> component detector (YOLO) with geometric fallback (optional)      src/ml/detector_source.py
  -> text association (refs/values -> components)                      src/core/text_associator.py
  -> net reconstruction (segment BFS over T-/dot-junctions)           src/core/graph_builder.py
  -> bipartite Components<->Nets graph -> SPICE / KiCad / JSON export  src/core/graph_builder.py
  -> density/scope gate (flags ultra-dense boards as low-confidence)   src/core/scope.py
  -> LLM tool-calling over the graph (7 tools, Ollama)                 src/llm/
ground truth: KiCad .kicad_sch -> exact mm->pt transform -> auto-labeled (no manual labeling)
```

### Key design decisions
- **Geometric first, OCR/detector as fallbacks.** Extraction works off the vector layer by default; RapidOCR fires only for text-less PDFs; the YOLO detector augments component recall but **falls back to geometry** when it covers <50% of a page's refs — so it never regresses a board it can't see.
- **Ground truth is free from KiCad.** The KiCad->PDF render is an exact similarity (`x_pt = 72/25.4 * x_mm`, zero translation), so symbol/pin coordinates map to the PDF with no manual annotation. This is the project's main differentiator vs. CNN approaches that need hand-labeled datasets.
- **The pipeline knows its limits.** A density gate (`nets/component > 8` or `> 80 components`) marks ultra-dense boards as low-confidence instead of emitting unreliable topology.
- **Honest, measured methodology.** Most candidate improvements were rejected *with measurements* (see HANDOFF.md): an oracle upper-bound experiment, "measured-dead" levers, interpreter-fragility analysis, and a qualitative error analysis that pins the dominant loss to short low-degree signal wires (not power rails).

---

## Quickstart

Three ways to run the finished system. Pick the one that fits.

### A. Docker (recommended — no local setup)
```bash
docker compose up                       # builds app, pulls qwen2.5, opens port 8501
# then open http://localhost:8501
```
The compose stack runs two services: **ollama** (GPU passthrough, pulls the default `qwen2.5:7b` model on first start) and **app** (Streamlit). Mount your trained detector weights at `./runs` (read-only); if absent, the app silently runs the geometric pipeline. For CPU-only (no NVIDIA GPU), drop the `deploy:` block under the `ollama` service.

### B. Local install (full system, your GPU)
```bash
pip install -e ".[dev,detector,ocr]"    # core + detector + OCR fallback

# Ollama must be running locally for LLM Q&A:
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M   # ~4.7 GB

# Web UI: upload a PDF, see overlay + scope gate, chat
streamlit run src/ui/app.py

# Or CLI with the detector path (auto-on if weights exist)
python -m src.cli.query query "Which components are isolated?" --pdf path/to/schematic.pdf
python -m src.cli.query query "..." --pdf ... --no-detector    # force geometric
python -m src.cli.query query "..." --pdf ... --mock           # no Ollama, plumbing only
```

### C. Mock mode (no GPU, no Ollama — plumbing/demo only)
```bash
pip install -e ".[dev]"
python -m src.cli.query query "Which components are isolated?" --pdf path/to/schematic.pdf --mock
```
Runs the full extraction + graph pipeline with a stub LLM that exercises the tool layer. Good for verifying the install without any model download.

### End-to-end demo scripts (no UI)
```bash
python scripts/demo_end_to_end.py --board sallen_key                 # PDF -> graph -> scope -> Q&A
python scripts/demo_end_to_end.py --weights runs/detect/<run>/weights/best.pt --dpi 150 --llm
```

### Component detector (optional, auto-labeled)
```bash
python scripts/build_detector_dataset.py --out data/detector   # auto-label from KiCad
python scripts/train_detector.py --imgsz 1280                   # train (GPU)
python diagnosi_d3/compare_detector.py --weights <best.pt> --hybrid   # geo vs detector, in-scope mean
```
See docs/DETECTOR.md.

---

## Limitations
- **Ultra-dense / bus-dominated boards** (memory arrays, wide buses) are below the supported envelope — flagged low-confidence by the scope gate; F1 there is low even with perfect geometry (an intrinsic cap, not a tunable).
- **Domain shift to true legacy scans is unverified** (see regime caveat above).
- **Structural, not electrical.** No simulation/validation; the LLM answers strictly from graph tool results.
- The F1 metric's greedy net-matching is mildly interpreter-sensitive (run with `PYTHONHASHSEED=0`).

---

## Repository layout
```
src/core/    extraction, graph builder, OCR fallback, scope gate, KiCad GT reader
src/ml/      clustering, classifier, color / detector integration, detector_runner (production glue)
src/llm/     GraphContext (7 tools), agent (Ollama/Mock), tool-calling loop
src/cli/     query CLI (with --detector/--no-detector)
src/ui/      Streamlit app (PDF upload + overlay + scope gate + chat)
scripts/     dataset builder, detector training, end-to-end demo
diagnosi_d3/ evaluation harnesses (F1, oracle upper-bound, error analysis, detector compare)
tests/       238 unit tests
Dockerfile, docker-compose.yml   local two-service stack (ollama + app)
docs/        DETECTOR.md ;  HANDOFF.md = full measured engineering log
```

## Status & tests
Working hybrid system; 238 tests green, ruff clean. Full decision log (every measured lever, including the rejected ones) is in **HANDOFF.md**.

## License
MIT
