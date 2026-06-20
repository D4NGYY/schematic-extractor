# Component detector (item 3) — auto-labeled, no manual annotation

## Why
All geometric-heuristic levers for component recall are measured-dead (HANDOFF
§19/§21/§22): the dominant remaining loss is components that don't form a clean
cluster (fused into neighbours, or symbol-less). The oracle shows the net-tracing
algorithm is fine — what's missing is **true component locations + pin positions**.
A detector supplies exactly that. Its ground truth is **free from KiCad** (zero
manual labeling), which is the project's differentiator vs CNN approaches that
need hand-annotated datasets.

## Pipeline (this phase)
1. **Auto-label:** `scripts/build_detector_dataset.py` renders page 0 of each
   rendered KiCad PDF and maps every root symbol via the EXACT KiCad→PDF
   similarity (`x_pt = 2.835·x_mm`, zero translation — verified on every board).
   For each non-power symbol it writes a YOLO box (class from ref prefix) plus a
   rich JSON sidecar with **true pin pixel coords** for a future keypoint model.
   - Output: `data/detector/{images,labels,json}`, `data.yaml`, board-level
     80/20 `train.txt`/`val.txt`. **973 boxes / 32 boards** at 150 dpi, 16 classes
     (resistor 252, capacitor 246, ic 114, diode 79, connector 78, …).
   - Validated: boxes/pins land exactly on the symbols (see overlays).
2. **Train (GPU box):** `scripts/train_detector.py` (Ultralytics YOLO). Small set
   → transfer-learn from yolov8s, imgsz≥1280 (small symbols), no rotate/flip
   (pin semantics). Not runnable in-sandbox.

## Integration target (next)
Feed detected boxes+pins back into `BipartiteGraphBuilder`: replace the
cluster→component step (or augment `recover_lost_refs`) with detector components
whose pins are TRUE positions, then run the existing (oracle-validated) net
tracer. This is the path that should move net-F1 — heuristic recovery couldn't
because it guessed pins from wires; the detector removes the guess.

## Caveats (honest)
- Class imbalance (R/C dominate; relay/transformer rare).
- 150 dpi may be low for the smallest glyphs — try 200–300 if recall on small
  parts is weak (dataset builder takes `--dpi`).
- Oracle showed perfect components alone gave inconsistent F1 gains; the win
  depends on combining detector pins with the existing wire extraction. Measure.
