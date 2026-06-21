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

```bash
pip install -e ".[dev]"          # core; add ".[ocr]" for the RapidOCR fallback

# End-to-end demo: PDF -> graph -> scope -> topology Q&A (no Ollama needed)
python scripts/demo_end_to_end.py --board sallen_key

# With the trained detector (clean components) and natural-language Q&A (Ollama)
python scripts/demo_end_to_end.py --weights runs/detect/<run>/weights/best.pt --dpi 150 --llm

# Natural-language query over a schematic (Ollama, qwen2.5:7b)
python -m src.cli.query "Which components are isolated?" --pdf path/to/schematic.pdf
# ...or --mock to run without Ollama

# Streamlit UI (overlay + chat)
streamlit run src/ui/app.py
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
src/ml/      clustering, classifier, color / detector integration
src/llm/     GraphContext (7 tools), agent (Ollama), tool-calling loop
src/cli/     query CLI                       src/ui/    Streamlit app
scripts/     dataset builder, detector training, end-to-end demo
diagnosi_d3/ evaluation harnesses (F1, oracle upper-bound, error analysis, detector compare)
tests/       238 unit tests
docs/        DETECTOR.md ;  HANDOFF.md = full measured engineering log
```

## Status & tests
Working hybrid system; 238 tests green, ruff clean. Full decision log (every measured lever, including the rejected ones) is in **HANDOFF.md**.

## License
MIT
