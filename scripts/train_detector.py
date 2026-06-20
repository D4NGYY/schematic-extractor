"""Train the component detector on the auto-labeled dataset (runs on a GPU box).

Thin wrapper over Ultralytics YOLO. NOT runnable in the sandbox (no GPU / heavy
dep); authored for the user's 3070 Ti. Build the dataset first:

    python scripts/build_detector_dataset.py --out data/detector
    pip install ultralytics
    python scripts/train_detector.py --epochs 150 --imgsz 1280 --batch 4

Notes:
- Schematic pages are large (~1754 px) and symbols are small -> keep imgsz high
  (1280+). On 8 GB VRAM use batch 2-4; lower imgsz if OOM.
- Dataset is small (~970 boxes / 32 boards) -> start from a pretrained yolov8n/s
  (transfer learning), heavy augmentation, and watch val mAP for overfit.
- data.yaml already defines a board-level 80/20 train/val split (no leakage).
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/detector/data.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default="schematic_detector")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so --help works without the dep

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=30,
        degrees=0.0,      # schematics are axis-aligned; don't rotate
        fliplr=0.0,       # mirroring changes pin semantics
        flipud=0.0,
        mosaic=0.5,
    )
    print("done -> runs/detect/" + args.name)


if __name__ == "__main__":
    main()
