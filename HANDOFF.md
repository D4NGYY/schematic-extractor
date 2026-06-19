# HANDOFF — schematic_extractor (Schematic AI Reasoner)

**Updated:** 2026-06-19 · **Status:** Phases 0–3 structurally complete, NOT yet functional on real schematics (4 blockers open)

---

## 1. TL;DR (read this first)

Pipeline that turns **vector** schematic PDFs into a queryable Components↔Nets graph (export to SPICE / KiCad / JSON), with an LLM as the final tool-calling layer. No OCR — purely geometric extraction (the deliberate alternative to OCR, which is unreliable on schematics).

**Honest current state:** the test suite is green (44/44) but exercises *synthetic/ideal* inputs. On a **real** KiCad/EDA schematic the pipeline would currently extract ~0% of references/values, detect no junctions, and classify every component as `"unknown"`. It is "structurally complete, functionally hollow." The 4 blockers below are what stand between "passes tests" and "works on a real drawing."

## 2. Goal & scope

**Does:** geometric extraction from vector PDFs → DBSCAN clustering → component classification → bipartite Component/Net graph → export SPICE/KiCad/JSON → LLM topology queries (`get_neighbors`, `get_path`, `get_net_components`).

**Does NOT:** read scanned/raster PDFs (that is the separate `librechat-ingestion-fix` OCR project — complementary, different input). Does not replace a real simulator; produces a structural model, not validated electrical behavior.

## 3. Architecture (data flow)

```
vector PDF
  → PyMuPDF extraction (segments, shapes, text)        src/core/extraction.py
  → DBSCAN clustering of primitives → symbol clusters  src/core/clustering.py
  → feature extraction (13 features) + RF classifier   src/core/feature_extractor.py, classifier.py
  → text association (refs / values → symbols)          src/core/text_associator.py
  → net reconstruction (segment BFS, junctions)         src/core/*nets*
  → bipartite graph Components↔Nets                     src/core/graph_builder.py
  → export SPICE / KiCad / JSON                         src/core/export*.py
  → LLM tool calling over the graph                     (Phase 5, not built)
Ground truth: KiCad .kicad_sch → coords → auto-labeled training set (no manual labeling)
```

## 4. Status by phase

- **Phase 0** ✅ complete and working.
- **Phase 1** ✅ complete (mypy fixes already applied in code).
- **Phase 2** ✅ structurally complete — BUT functionally broken on real input: `_merge_text_spans()` is a stub (B3), `is_value` regex inadequate (B4), `is_ref_designator` incomplete (D1), junction detection unreachable (B2).
- **Phase 3** ✅ structurally complete — BUT ML classifier never trained (B1), DBSCAN eps estimation wrong for large schematics (D2), pin assignment from bbox corners topologically wrong (D3).
- **Phase 4** ⬜ ERC (electrical rule check) — not started; depends on B1, B2, D2 to give meaningful results.
- **Phase 5** ⬜ LLM tool calling + 20-question topology benchmark — not started.
- **Phase 6** ⬜ Streamlit UI (`src/ui/app.py` does not exist) + portfolio — not started.

## 5. Known issues / risks

### 🔴 Blockers (output is unusable on real schematics until fixed)
- **B1 — ML classifier inert.** No training set, no `fit()` call, no `models/`. Every component → `"unknown"`, confidence 0.0. (`classifier.py`, `graph_builder.py:116`)
- **B2 — Junction detection unreachable.** No code path ever creates `PDFShape(item_type="circle")`; `junction_candidates()` always returns `[]`. Must read drawing-level circles from PyMuPDF `get_drawings()`. (`extraction.py`)
- **B3 — `_merge_text_spans()` is a stub.** KiCad fragments labels ("R"+"1" → "R 1"); refs/values don't match. Use `get_text("dict")` spans merged by line + gap. (`extraction.py:348`)
- **B4 — `is_value` regex too narrow.** Misses `49R9`, `4k7`, `10K0`, `2N2222`, `+5V`, `3V3`, `100R`… Needs EDA R-notation + part-number + voltage patterns. (`extraction.py`)

### 🟡 Should-fix
- **D1** `is_ref_designator` misses `U1A`, `QB1`, multi-letter/suffix designators.
- **D2** DBSCAN eps via `pdist` (all-pairs median) → huge eps on large schematics → one giant cluster. Use k-NN distance percentile.
- **D3** Pin assignment = 4 bbox-corner virtual pins → wrong topology for 2-pin / multi-pin parts.
- **D4** `_nearest_cluster` dict-key collision silently drops refs (two refs nearest same cluster).
- **D5** TextAssociator (shapes) vs `_nearest_cluster` (DBSCAN centroids) — incoherent double mapping.
- **D6** `_segments_touch()` fixed 1.0px tolerance, not format-aware.
- **D7** `export_json()` uses inline `open()`/`import json` vs `path.write_text()` elsewhere.

### 🟢 Nice-to-have
- N1 HANDOFF/README stale (claimed 10 mypy errors — already 0). N2 perf O(n²/n³) on large schematics. N3 test-coverage gaps (no real-PDF test, no topological-correctness test). N4 `node_id` may collide with real `U1`. N5 `stub_length` not configurable.

## 6. Next steps (ordered)
1. B3 + B4 + D1 — text-span merge & value/ref regex (unblocks any real-schematic test).
2. B2 — junction detection (read drawing-level circles).
3. D2 — DBSCAN eps via k-NN.
4. B1 — ML training pipeline (or interim rule-based classifier from feature vectors).
5. D3/D4/D5 — pin & association correctness.
6. Then Phase 4 (ERC) → Phase 5 (LLM tools + benchmark) → Phase 6 (UI).

## 7. Run / test
```
cd C:\Users\danie\Desktop\Projects\schematic_extractor
pytest -q          # expect 44 passed
mypy .             # 0 errors
ruff check .       # 0 errors
```
Gaps: KiCad CLI not installed → synthetic `.kicad_sch` have no matching PDF → no real end-to-end test yet.

## 8. Key files
`src/core/extraction.py` (PyMuPDF, text/shape parsing — B2/B3/B4 live here), `clustering.py` (DBSCAN — D2), `feature_extractor.py` + `classifier.py` (RF — B1), `text_associator.py` (D5), `graph_builder.py` (bipartite, pins — D3/D4), `export_*.py` (D7), `tests/` (44 tests).

## 9. Constraints / conventions
- No OCR — geometric extraction only (design choice).
- Tests + mypy + ruff must stay green. Black/ruff formatting.
- Ground truth auto-derived from KiCad files (no manual labeling).
- Update this HANDOFF and TODO.md after each work session (keep them non-stale).

## 10. Open decisions
- Train the RF properly (needs KiCad→PDF alignment) vs ship an interim rule-based classifier first?
- Install KiCad CLI to generate real test PDFs, or hand-pick a few public vector schematics as the real-input test set?
