"""Generalized net-connectivity F1 over EVERY board that has both a PDF and a
.kicad_sch ground-truth file.

Unlike `true_f1_validation.py` (hardcoded to arduino_micro / arduino_nano), this
walks `test_input/multi_schematic/<board>/` and scores any folder containing a
`*.pdf` + `*.kicad_sch` pair. Use it after `scripts/expand_dataset.py` has
rendered more real KiCad projects, to see whether the headline F1 (~0.36) is
representative or an artifact of two boards.

Run:  PYTHONHASHSEED=0 PYTHONPATH=. python diagnosi_d3/f1_all_boards.py

Two correctness refinements vs the original 2-board script:
- Multi-page (hierarchical) PDFs are scored across ALL pages, not page 0 only:
  components on sub-sheets live in the GT, so page-0-only scoring deflates F1.
  Each page is built in an ISOLATED builder (build_from_page accumulates and
  re-connects pins on every call, so reusing one builder contaminates nets
  across pages); net ids are namespaced per page and per-ref membership unioned.
- The aggregate excludes boards with num_gt == 0 (empty hierarchy sub-sheets
  with no component symbols) — they are not circuits and would drag the mean
  toward 0 misleadingly. They are reported separately.

Note on determinism: the greedy net-mapping tie-break is ordering-sensitive, so
F1 can shift by ~1 tp across Python versions even with PYTHONHASHSEED=0. Report
the interpreter version alongside the number.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import fitz
from PIL import Image
from ultralytics import YOLO

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.kicad_gt_reader import build_gt_graph, parse_kicad_sch
from src.core.pdf_parser import VectorExtractor
from src.ml.detector_source import Detection, DetectorComponentSource

BASE = Path("test_input/multi_schematic")

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = YOLO("runs/detect/schematic_detector-7/weights/best.pt")
    return MODEL

def score_board(pdf: Path, sch: Path) -> dict:
    pages = VectorExtractor().extract(str(pdf))
    doc = fitz.open(str(pdf))
    num_ext_comps = 0
    ext_cn: dict[str, set] = defaultdict(set)
    
    detector_src = DetectorComponentSource(dpi=150.0)
    model = get_model()
    
    for pi, page in enumerate(pages):
        fitz_page = doc[pi]
        s = 150.0 / 72.0
        pix = fitz_page.get_pixmap(matrix=fitz.Matrix(s, s))
        if pix.n != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        results = model.predict(img, imgsz=1280, verbose=False)
        result = results[0]
        
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                detections.append(Detection(class_name=class_name, bbox_px=(x0, y0, x1, y1), confidence=conf))
                
        comps = detector_src.components(detections, page)

        b = BipartiteGraphBuilder(cluster_eps=None)
        b.build_from_page(page, detector_components=comps)
        num_ext_comps += len(b.components)
        for net in b.nets.values():
            for ref, c in b.components.items():
                for p in c.pins:
                    if p.connected_net == net.net_id:
                        ext_cn[ref].add(f"{pi}:{net.net_id}")

    gt = build_gt_graph(parse_kicad_sch(sch))
    common = set(ext_cn) & set(gt.components)

    gt_cn: dict[str, set] = defaultdict(set)
    for nid, pins in gt.nets.items():
        for ref, _ in pins:
            gt_cn[ref].add(nid)

    enr: dict = defaultdict(set)
    gnr: dict = defaultdict(set)
    for ref, nets in ext_cn.items():
        if ref in common:
            for n in nets:
                enr[n].add(ref)
    for ref, nets in gt_cn.items():
        if ref in common:
            for n in nets:
                gnr[n].add(ref)

    scores = [
        (len(er & gr), en, gn)
        for en, er in enr.items()
        for gn, gr in gnr.items()
        if er & gr
    ]
    scores.sort(reverse=True)
    nm: dict = {}
    ue: set = set()
    ug: set = set()
    for _ov, en, gn in scores:
        if en not in ue and gn not in ug:
            nm[en] = gn
            ue.add(en)
            ug.add(gn)

    tp = fp = fn = 0
    for ref in common:
        e = ext_cn[ref]
        g = gt_cn[ref]
        mapped = {nm.get(n) for n in e if n in nm}
        for gn in g:
            if gn in mapped:
                tp += 1
            else:
                fn += 1
        for en in e:
            if nm.get(en) not in g:
                fp += 1

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "num_ext": num_ext_comps,
        "num_pages": len(pages),
        "num_gt": len(gt.components),
        "overlap_refs": len(common),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }


def discover_boards() -> list[dict]:
    boards = []
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        pdfs = list(d.glob("*.pdf"))
        schs = list(d.glob("*.kicad_sch"))
        if pdfs and schs:
            boards.append({"name": d.name, "pdf": pdfs[0], "sch": schs[0]})
    return boards


def main() -> None:
    boards = discover_boards()
    if not boards:
        print(f"No (pdf, kicad_sch) pairs found under {BASE}", file=sys.stderr)
        sys.exit(1)

    results = {}
    for board in boards:
        try:
            results[board["name"]] = score_board(board["pdf"], board["sch"])
        except Exception as e:  # noqa: BLE001 - report and continue
            results[board["name"]] = {"error": str(e)}

    print(json.dumps(results, indent=2))

    scored = {n: r for n, r in results.items() if "f1" in r}
    real = {n: r for n, r in scored.items() if r.get("num_gt", 0) > 0}
    empty = [n for n, r in scored.items() if r.get("num_gt", 0) == 0]
    errored = [n for n, r in results.items() if "error" in r]

    if real:
        f1s = [r["f1"] for r in real.values()]
        print(f"\nPython {sys.version.split()[0]}", file=sys.stderr)
        print(f"real boards (num_gt>0): {len(real)}", file=sys.stderr)
        print(f"mean F1 = {sum(f1s) / len(f1s):.4f}", file=sys.stderr)
        print(f"median  = {sorted(f1s)[len(f1s) // 2]:.4f}", file=sys.stderr)
        print(f"min/max = {min(f1s):.4f} / {max(f1s):.4f}", file=sys.stderr)
    if empty:
        print(
            f"excluded (num_gt==0, empty sub-sheets): {len(empty)} -> "
            f"{', '.join(sorted(empty))}",
            file=sys.stderr,
        )
    if errored:
        print(f"errored: {len(errored)} -> {', '.join(sorted(errored))}", file=sys.stderr)


if __name__ == "__main__":
    main()
