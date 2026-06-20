# TODO — schematic_extractor

**Updated:** 2026-06-20 (Phase 5 LLM done; net-connectivity re-diagnosed) · Source of truth for what's next. Work top-down within each priority block.

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

- [~] 🔴 **Net-connectivity (visual-first).** _Aggiornato 2026-06-20 con misure su F1 GT (arduino_micro/nano)._
  - **GT honesty (DONE, commit 041502f):** `build_gt_graph` escludeva `#PWR`/`#FLG` (power port) dai componenti → ~70/94 falsi negativi incancellabili. Ora esclusi. `num_gt` micro 94→48, nano 71→34.
  - **Baseline F1 (post-fix):** micro f1_new=0.21 (overlap 24/48), nano f1_new=0.00 (overlap **3/34**). Il metric è gated sui common_refs e guidato da `fn` (connettività net).
  - **Causa radice misurata:** pin connection **16%** (98/612), net degree max **2** (la GND reale GT tocca 17 pin). Le net estratte sono schegge.
  - **Smentito per esperimento:** abbassare la soglia di `separate_wires` (factor 3→1) **non** aumenta `netmaxdeg` (resta 2) e **peggiora** Bryston (comps 13→9, blob). Quindi `separate_wires` NON è la leva.
  - **Diagnosi visiva (overlay arduino_micro, 2026-06-20):** problema multi-fattore =
    1. **🎯 LEVER DOMINANTE — pin_tol di scala sbagliata (DONE, commit pin_tol):** `_connect_pins_to_nets` ignorava `scale` e usava `pin_tol=3×wire_tol` (≈4), ~6× più piccolo dei gap reali pin→net (~25). Fix: `pin_tol=max(3×wire_tol, scale×pin_tol_factor)`, `pin_tol_factor=2.0` (sperimentato: plateau a 2.0). **micro F1 0.209→0.356, netmaxdeg 2→6, morte 77→24, isolati 76→22, Bryston isolati 3→0.** +2 test, 197 passed.
    2. **Label-merge per nome (DONE):** `_merge_nets_by_label` fonde net con stesso nome label (GND/+5V/RESET), skippa label numeriche. **Sbloccato dal fix pin_tol** (ora GND si fonde davvero, i pin raggiungono le net).
    3. **Pin sovra-generati (DA FARE):** `select_pins`/`free_endpoints` ritorna endpoint interni ai simboli e del testo (988 pin per 112 comp). Gonfia i fp; prossima leva.
    4. **net_label association rumorosa (DA FARE):** pesca part-number come label (`ATMEGA32U4-XUMU`, `M20-9980345`). Da raffinare in `text_associator`.
  - **2° leva indipendente:** ref-recovery (overlap nano 3/34) — l'estrattore non recupera i designator (text_associator); è il tetto della recall.
  - **Overlay riproducibile:** `uv run python -c "from src.ui.render import save_overlay; save_overlay('test_input/multi_schematic/arduino_micro/arduino_micro.pdf','diagnosi_d3/overlay_micro.png',dpi=200)"`.

- [x] 🟡 **Phase 4 — ERC** (`src/core/erc.py`): ISOLATED_COMPONENT, FLOATING_PIN, DANGLING_NET, UNCONNECTED_NET, UNNAMED_NET. 16 test. Bryston: 31 err + 1 warn (stub matching debole → D6 prioritario).
- [x] 🔴 **Phase 5 — LLM tool calling** (`src/llm/tools.py`, `src/llm/agent.py`, `src/cli/query.py`). 7 tools, dual-mode parsing, Ollama. **Debugged end-to-end (2026-06-20):** fixed broken GraphContext schema, 2 crash bugs, hardened ReAct parser. Scored benchmark → qwen2.5:7b 25/25 (5/5). See `TEST_MANUAL.md`.
- [~] 🟢 **Phase 6 — Streamlit UI** (`src/ui/app.py`): debug overlay + chat tab done; portfolio polish + SVG overlay pending.

## P5 — Hygiene / cleanup

- [x] 🟢 **D6** `_estimate_scale()` p10 segment lengths; stub `min(w,h)*0.5`; T-junction `<=0`. Edges 1→4 su Bryston; 127/127 pytest, mypy 0, ruff 0.
- [x] 🟡 **Wire/symbol separation**: `SpatialClusterer.separate_wires()` (axis-aligned AND ≥p25×3); DBSCAN su soli symbol_segs. Bryston: edges 4→9, isolated 6→3. 135/135 pytest, mypy 0, ruff 0. Bottleneck residuo: frammenti Bezier arco.
- [x] 🔴 **Clustering single-linkage su endpoint** (sostituisce midpoint-DBSCAN). Union-find sulla prossimita degli endpoint dei segmenti + griglia spaziale O(n); link_dist adattivo data-derived (p60 nearest-other-endpoint, ~8.6pt). Shapes assegnate al gruppo piu vicino, orfane scartate. Bryston: blob WB1 eliminato, edges 9→35, cluster a scala-componente. 142/142 pytest (+7), ruff 0, mypy 0 sul file. Commit f22b2c9. Bottleneck successivo svelato: D3 (pin→net) + over-segmentation.
- [ ] 🟢 **D7** `export_json()` → `path.write_text()`. **N4** `node_id` collision guard. **N5** configurable `stub_length`.
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
