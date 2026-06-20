"""Auto-label a component-detector dataset from (PDF, .kicad_sch) pairs.

No manual labeling: the KiCad->PDF transform is the exact similarity
x_pt = (72/25.4)*x_mm, y_pt = (72/25.4)*y_mm with zero translation (verified:
the rendered page is the KiCad sheet at 2.835 pt/mm). Page 0 of the rendered PDF
== the root .kicad_sch, so root symbols label page 0 unambiguously.

Per board it renders page 0 to PNG and writes, for each non-power symbol:
  * a YOLO box line  `cls cx cy w h`  (normalized) derived from the symbol's pin
    extent + margin (covers the body; thin 2-pin parts get a perpendicular pad),
  * a rich JSON sidecar with box + true pin pixel coords + ref + class, so a
    pin-aware (keypoint) model can be trained later without re-deriving anything.

Run: PYTHONPATH=. python scripts/build_detector_dataset.py [--dpi 150] [--out data/detector] [board ...]
Training (YOLO) runs on the user's GPU machine; this only builds the dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz  # type: ignore

from src.core.kicad_gt_reader import parse_kicad_sch

MM_TO_PT = 72.0 / 25.4
BASE = Path("test_input/multi_schematic")

# Compact detector class set; ref-prefix -> class. Unknown prefixes -> "other".
_PREFIX_CLASS = {
    "R": "resistor", "RV": "potentiometer", "RN": "resistor",
    "C": "capacitor", "L": "inductor", "D": "diode", "LED": "diode",
    "Q": "transistor", "U": "ic", "IC": "ic", "X": "crystal", "Y": "crystal",
    "J": "connector", "P": "connector", "CN": "connector", "K": "relay",
    "SW": "switch", "S": "switch", "F": "fuse", "T": "transformer",
    "TP": "testpoint", "BT": "battery", "B": "battery",
}
CLASSES = [
    "resistor", "capacitor", "inductor", "diode", "transistor", "ic",
    "crystal", "connector", "relay", "switch", "fuse", "transformer",
    "testpoint", "battery", "potentiometer", "other",
]
_CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


def ref_to_class(ref: str) -> str:
    head = "".join(ch for ch in ref if ch.isalpha())
    for n in (3, 2, 1):  # longest prefix first (RV before R)
        if head[:n] in _PREFIX_CLASS:
            return _PREFIX_CLASS[head[:n]]
    return "other"


def build_board(name: str, out: Path, dpi: float, margin_mm: float) -> dict:
    d = BASE / name
    pdfs = list(d.glob("*.pdf"))
    schs = list(d.glob("*.kicad_sch"))
    if not (pdfs and schs):
        return {"skipped": "no pdf/sch"}
    sch = parse_kicad_sch(schs[0])
    doc = fitz.open(str(pdfs[0]))
    page = doc[0]
    s = dpi / 72.0  # pt -> px
    pix = page.get_pixmap(matrix=fitz.Matrix(s, s))
    W, H = pix.width, pix.height

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "json").mkdir(parents=True, exist_ok=True)
    img_path = out / "images" / f"{name}.png"
    if not img_path.exists():  # mount may forbid overwrite; render once
        pix.save(str(img_path))

    margin_px = margin_mm * MM_TO_PT * s
    lines: list[str] = []
    records: list[dict] = []
    counts: dict[str, int] = {}
    for sym in sch.symbols:
        if sym.ref.startswith("#") or not sym.pins:
            continue  # power ports / pinless -> skip
        pins_px = [(p.x * MM_TO_PT * s, p.y * MM_TO_PT * s) for p in sym.pins]
        xs = [p[0] for p in pins_px]
        ys = [p[1] for p in pins_px]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        x0 -= margin_px; y0 -= margin_px; x1 += margin_px; y1 += margin_px
        x0 = max(0.0, x0); y0 = max(0.0, y0); x1 = min(float(W), x1); y1 = min(float(H), y1)
        if x1 <= x0 or y1 <= y0:
            continue
        cls = ref_to_class(sym.ref)
        counts[cls] = counts.get(cls, 0) + 1
        cx = (x0 + x1) / 2 / W
        cy = (y0 + y1) / 2 / H
        bw = (x1 - x0) / W
        bh = (y1 - y0) / H
        lines.append(f"{_CLASS_IDX[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        records.append({
            "ref": sym.ref, "class": cls,
            "box_xyxy_px": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "pins_px": [[round(px, 2), round(py, 2)] for px, py in pins_px],
        })

    (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
    (out / "json" / f"{name}.json").write_text(json.dumps(
        {"image": f"images/{name}.png", "w": W, "h": H, "dpi": dpi,
         "components": records}, indent=2))
    return {"image": f"{W}x{H}", "labeled": len(records), "counts": counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("boards", nargs="*", help="board names (default: all)")
    ap.add_argument("--dpi", type=float, default=150.0)
    ap.add_argument("--margin-mm", type=float, default=1.27)
    ap.add_argument("--out", default="data/detector")
    args = ap.parse_args()
    out = Path(args.out)
    names = args.boards or [
        p.name for p in sorted(BASE.iterdir())
        if p.is_dir() and list(p.glob("*.pdf")) and list(p.glob("*.kicad_sch"))
    ]
    summary = {}
    total = 0
    agg: dict[str, int] = {}
    for n in names:
        try:
            r = build_board(n, out, args.dpi, args.margin_mm)
        except Exception as e:  # noqa: BLE001
            r = {"error": str(e)}
        summary[n] = r
        total += r.get("labeled", 0)
        for c, k in r.get("counts", {}).items():
            agg[c] = agg.get(c, 0) + k
    # Deterministic 80/20 train/val split BY BOARD (no leakage between splits).
    labeled_names = [n for n in names if summary[n].get("labeled", 0) > 0]
    labeled_names.sort()
    val = {n for i, n in enumerate(labeled_names) if i % 5 == 0}
    train_list = [f"images/{n}.png" for n in labeled_names if n not in val]
    val_list = [f"images/{n}.png" for n in labeled_names if n in val]
    (out / "train.txt").write_text("\n".join(train_list))
    (out / "val.txt").write_text("\n".join(val_list))
    (out / "data.yaml").write_text(
        f"path: .\ntrain: train.txt\nval: val.txt\nnc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (out / "classes.txt").write_text("\n".join(CLASSES))
    (out / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"boards": len(names), "total_labeled": total,
                      "by_class": dict(sorted(agg.items(), key=lambda kv: -kv[1]))}, indent=2))


if __name__ == "__main__":
    main()
