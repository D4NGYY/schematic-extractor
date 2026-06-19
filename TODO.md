# TODO — schematic_extractor

**Updated:** 2026-06-19 (Phase 4 ERC done) · Source of truth for what's next. Work top-down within each priority block.

**Legend:** `[ ]` open · `[~]` in progress · `[x]` done · severity 🔴 blocker / 🟡 should-fix / 🟢 nice-to-have

---

## P1 — Make it work on a REAL schematic (do first)

- [x] 🔴 **B3 — Real text-span merge.** Replaced `_merge_text_spans()` stub with `_extract_text_blocks()` via `get_text("dict")`; spans on same line with gap < 60% font-size merge: "R"+"1"→"R1". `src/core/pdf_parser.py:357`.
- [x] 🔴 **B4 — Widen `is_value` regex to real EDA notation.** Covers `49R9`, `4k7`, `10K0`, `2N2222`, `BC547`, `+5V`, `3V3`, `100R`, `0R1`, MJE15030, TL064 etc. Module-level `_VALUE_RE`. `src/core/pdf_parser.py:14`.
- [x] 🟡 **D1 — Extend `is_ref_designator`** to `^(?:[A-Z][0-9]{1,4}|[A-Z]{2}[0-9]{1,2})[A-Z]?$` (covers `U1A`, `QB1`, `RB14`, `RN1`; 2-letter+3+digit = part number, not ref).
- [x] 🟢 **N3a — Add a real-PDF integration test**: 16 new unit tests covering all patterns + span merging (60 total, up from 44). Bryston extraction: 168 refs + 120 values/pagina.

## P2 — Correct connectivity

- [x] 🔴 **B2 — Junction detection.** `_try_extract_circle()` in `pdf_parser.py`: rileva cerchi pieni dal drawing dict (fill + all-Bezier + bbox quadrata). Bryston: 162 junction candidates (era 0).
- [x] 🟡 **D2 — Fix DBSCAN eps estimation.** k-NN (k=4, p90 × 1.5) sostituisce pdist: eps 456pt → ~local; cluster 1 → 7 su Bryston.
- [x] 🟡 **D4 — Fix `_nearest_cluster` collision.** `dict[int, list[SymbolAssociation]]` + `min(…, key=distance)`.
- [x] 🟡 **D3 — Pin assignment (minimo).** `_connect_pins_to_nets()` già scarta pin virtuali non connessi (`if best_net is not None`). Geometria simbolo rimandata a P3.
- [x] 🟡 **D5 — Reconcile (minimo).** `_nearest_cluster` usa `symbol_center` (centro del simbolo) invece di `text_pos` (pos. del label fuori dal componente). Fix strutturale completo rimandato a P3.

## P3 — Make classification real

- [x] 🔴 **B1 — Classificatore rule-based (provvisorio, sblocca Fase 4).**
  - `RuleBasedClassifier` in `classifier.py`: mappa 2-lettera poi 1-lettera (R→resistor, QB→transistor, TP→testpoint, VR→regulator…) + fallback geometrico (power_symbol, ic, unknown).
  - `graph_builder.py`: ref calcolato PRIMA della classificazione (segnale primario); ML usato se addestrato, rule-based altrimenti.
  - Bryston: 5/7 componenti classificati (71% non-unknown). 30 test aggiunti. 103/103 pytest, mypy 0, ruff 0.
  - Path ML RF lasciato intatto per training futuro (`ComponentClassifier.fit()` non rimosso).

## P4 — Phases 4–6 (after P1–P3)

- [x] 🟡 **Phase 4 — ERC** (`src/core/erc.py`): ISOLATED_COMPONENT, FLOATING_PIN, DANGLING_NET, UNCONNECTED_NET, UNNAMED_NET. 16 test. Bryston: 31 err + 1 warn (stub matching debole → D6 prioritario).
- [ ] 🟡 **Phase 5 — LLM tool calling** (`src/core/llm_tools.py`) + 20-question topology benchmark.
- [ ] 🟢 **Phase 6 — Streamlit UI** (`src/ui/app.py`): 300 DPI PNG render + SVG overlay + selectbox; then portfolio polish.

## P5 — Hygiene / cleanup

- [ ] 🟢 **D6** format-aware `_segments_touch()` tolerance. **D7** `export_json()` → `path.write_text()`. **N4** `node_id` collision guard. **N5** configurable `stub_length`.
- [ ] 🟢 Perf: spatial index for `_merge_collinear_segments()` and `_build_nets()` BFS on large schematics.

---

## Done
- [x] Phase 0 — extraction scaffolding.
- [x] Phase 1 — typed pipeline (mypy 0).
- [x] Phase 2 — text/shape parsing (structural).
- [x] Phase 3 — clustering + classifier wiring + bipartite graph + exports (structural).
- [x] Tooling green: 44/44 pytest, mypy 0, ruff 0.
- [x] P1 — Real extraction on Bryston schematic: 168 refs + 120 values/pagina (60/60 pytest, mypy 0, ruff 0).
- [x] P2 — Connectivity fixes: B2 (162 junctions), D2 (eps k-NN: 1→7 cluster), D4 (no collision), D3 min (già ok), D5 min (symbol_center). 73/73 pytest, mypy 0, ruff 0.
- [x] P3/B1 — Rule-based classifier: 71% non-unknown su Bryston (0%→71%). 103/103 pytest, mypy 0, ruff 0.
