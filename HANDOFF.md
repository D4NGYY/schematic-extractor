# HANDOFF — schematic_extractor (Schematic AI Reasoner)

**Updated:** 2026-06-19 (P2 done) · **Status:** Phases 0–3 complete + P1+P2 fixes; functional on real vector schematics (1 blocker remains: B1 ML training)

---

## 1. TL;DR (read this first)

Pipeline that turns **vector** schematic PDFs into a queryable Components↔Nets graph (export to SPICE / KiCad / JSON), with an LLM as the final tool-calling layer. No OCR — purely geometric extraction (the deliberate alternative to OCR, which is unreliable on schematics).

**Honest current state:** the test suite is green (73/73). On the Bryston real schematic: 168 refs, 120 values, 162 junctions detected; DBSCAN gives 7 clusters (1→7 with k-NN eps). Classifier still produces `"unknown"` for all components (B1 not yet fixed). Net reconstruction and graph export work structurally.

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
- ~~**B2** — Junction detection: FIXED. `_try_extract_circle()` in `pdf_parser.py`; 162 junctions on Bryston.~~
- **B3 — `_merge_text_spans()` is a stub.** KiCad fragments labels ("R"+"1" → "R 1"); refs/values don't match. Use `get_text("dict")` spans merged by line + gap. (`extraction.py:348`)
- **B4 — `is_value` regex too narrow.** Misses `49R9`, `4k7`, `10K0`, `2N2222`, `+5V`, `3V3`, `100R`… Needs EDA R-notation + part-number + voltage patterns. (`extraction.py`)

### 🟡 Should-fix
- **D1** `is_ref_designator` misses `U1A`, `QB1`, multi-letter/suffix designators.
- ~~**D2** eps pdist: FIXED — k-NN p90×1.5; Bryston 1→7 cluster.~~
- **D3** Pin assignment = 4 bbox-corner virtual pins → wrong topology for 2-pin / multi-pin parts. Minimum fix (drop unconnected) already in place; full fix needs symbol geometry (P3).
- ~~**D4** collision: FIXED — dict-of-list + min(distance).~~
- **D5** TextAssociator/DBSCAN two-step mapping still conceptually split; minimal fix (`symbol_center` in `_nearest_cluster`) applied. Full reconciliation deferred to P3.
- **D6** `_segments_touch()` fixed 1.0px tolerance, not format-aware.
- **D7** `export_json()` uses inline `open()`/`import json` vs `path.write_text()` elsewhere.

### 🟢 Nice-to-have
- N1 HANDOFF/README stale (claimed 10 mypy errors — already 0). N2 perf O(n²/n³) on large schematics. N3 test-coverage gaps (no real-PDF test, no topological-correctness test). N4 `node_id` may collide with real `U1`. N5 `stub_length` not configurable.

## 6. Next steps (ordered)
1. **B1 — ML training** (or interim rule-based classifier): currently every component → `"unknown"`. Options: train RF on the 7 synthetic `.kicad_sch`, or write a rule-based classifier from the 13 feature vector dimensions.
2. **D3 full** — real pin positions from symbol geometry (after B1, so we know component class → pin count/layout).
3. **D5 full** — collapse TextAssociator+`_nearest_cluster` into one pass using DBSCAN cluster bboxes directly.
4. Then Phase 4 (ERC) → Phase 5 (LLM tools + benchmark) → Phase 6 (UI).

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
