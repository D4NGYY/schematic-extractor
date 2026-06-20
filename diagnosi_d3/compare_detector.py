"""Side-by-side net-F1: geometric pipeline vs detector pipeline, SAME boards /
script / interpreter. This is the only honest way to read the detector delta
(don't compare a py3.12 detector run to a py3.10 geometric run on a different
board subset).

Geometric  = BipartiteGraphBuilder().build_from_page(page)            (baseline)
Detector   = build_from_page(page, detector_components=<YOLO boxes>)  (new)

Run on the GPU/Windows box AFTER training:
  PYTHONHASHSEED=0 PYTHONPATH=. python diagnosi_d3/compare_detector.py \
      --weights runs/detect/schematic_detector/weights/best.pt --dpi 150

Notes:
- Detector currently scores PAGE 0 only (the dataset labels page 0); multi-page
  boards (e.g. video, 8 pages) will show low detector recall by construction —
  that's a dataset-coverage limit, not a detector miss. num_pages is printed.
- Reuses f1_all_boards.score_board for the geometric column so the metric is
  identical.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.kicad_gt_reader import build_gt_graph, parse_kicad_sch
from src.core.pdf_parser import VectorExtractor
from src.ml.detector_source import Detection, DetectorComponentSource

BASE = Path("test_input/multi_schematic")


def is_container(sch: Path, num_gt: int) -> bool:
    """A hierarchical ROOT whose sub-sheets are scored as their own boards (e.g.
    `video` -> muxdata/pal-ntsc/...). It has sheet refs but few own components;
    scoring it double-counts other boards. Excluded from the aggregate."""
    txt = sch.read_text(errors="ignore")
    return txt.count("Sheetfile") > 0 and num_gt <= 5


def _f1(ext_cn: dict, gt_cn: dict, common: set) -> float:
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
    sc = sorted(
        ((len(er & gr), en, gn) for en, er in enr.items() for gn, gr in gnr.items() if er & gr),
        reverse=True,
    )
    nm: dict = {}
    ue: set = set()
    ug: set = set()
    for _o, en, gn in sc:
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
            tp += 1 if gn in mapped else 0
            fn += 0 if gn in mapped else 1
        for en in e:
            if nm.get(en) not in g:
                fp += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def _membership(b: BipartiteGraphBuilder, pi: int) -> dict:
    valid = set(b.nets)
    out: dict = defaultdict(set)
    for ref, c in b.components.items():
        for pin in c.pins:
            if pin.connected_net in valid:
                out[ref].add(f"{pi}:{pin.connected_net}")
    return out


def score(pdf, sch, model, dpi, src, images_dir, hybrid=False, min_frac=0.5):  # type: ignore[no-untyped-def]
    pages = VectorExtractor().extract(str(pdf))
    gt = build_gt_graph(parse_kicad_sch(sch))
    gt_cn: dict = defaultdict(set)
    for nid, pins in gt.nets.items():
        for ref, _ in pins:
            gt_cn[ref].add(nid)

    def build(detector: bool) -> dict:
        ext: dict = defaultdict(set)
        for pi, page in enumerate(pages):
            b = BipartiteGraphBuilder(cluster_eps=None)
            if detector:
                if pi != 0:
                    continue  # dataset labels page 0 only
                r = model(str(pdf_image(pdf, dpi, images_dir)), verbose=False)[0]
                dets = [
                    Detection(class_name=r.names[int(c)], bbox_px=tuple(xyxy), confidence=float(p))
                    for xyxy, c, p in zip(
                        r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                        strict=False,
                    )
                ]
                comps = (
                    src.components_or_fallback(dets, page, min_frac)
                    if hybrid
                    else src.components(dets, page)
                )
                b.build_from_page(page, detector_components=comps)
            else:
                b.build_from_page(page)
            for ref, nets in _membership(b, pi).items():
                ext[ref] |= nets
        common = set(ext) & set(gt.components)
        return {"f1": round(_f1(ext, gt_cn, common), 4), "overlap": len(common)}

    geo = build(False)
    det = build(True)
    return {"num_gt": len(gt.components), "pages": len(pages),
            "geo_f1": geo["f1"], "det_f1": det["f1"],
            "geo_ovl": geo["overlap"], "det_ovl": det["overlap"]}


def pdf_image(pdf: Path, dpi: float, images_dir: Path) -> Path:
    """Page-0 image for the detector. Prefer a prebuilt image in images_dir; else
    render at `dpi`. The render dpi MUST match the dpi the detector was trained at
    AND the --dpi passed to DetectorComponentSource (px->pt mapping)."""
    cand = images_dir / f"{pdf.parent.name}.png"
    if cand.exists():
        return cand
    import fitz  # type: ignore
    cache = Path(f"/tmp/_cmp_img_{int(dpi)}")
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{pdf.parent.name}.png"
    if not out.exists():
        page = fitz.open(str(pdf))[0]
        page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72)).save(str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dpi", type=float, default=150.0)
    ap.add_argument("--hybrid", action="store_true",
                    help="fall back to geometric when the detector is sparse")
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--images-dir", default="data/detector/images",
                    help="prebuilt page-0 images (must match --dpi); else rendered")
    ap.add_argument("--keep-containers", action="store_true",
                    help="do NOT exclude hierarchical container roots")
    ap.add_argument("boards", nargs="*")
    args = ap.parse_args()
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(args.weights)
    src = DetectorComponentSource(dpi=args.dpi)
    names = args.boards or [
        d.name for d in sorted(BASE.iterdir())
        if d.is_dir() and list(d.glob("*.pdf")) and list(d.glob("*.kicad_sch"))
    ]
    rows = []
    for n in names:
        d = BASE / n
        try:
            sch_p = next(d.glob("*.kicad_sch"))
            r = score(next(d.glob("*.pdf")), sch_p, model, args.dpi, src,
                      Path(args.images_dir), hybrid=args.hybrid, min_frac=args.min_frac)
            if not args.keep_containers and "num_gt" in r and is_container(sch_p, r["num_gt"]):
                r["container"] = True
        except Exception as e:  # noqa: BLE001
            r = {"error": str(e)[:50]}
        rows.append((n, r))
        if "error" in r:
            print(f"{n:28s} ERROR {r['error']}")
        elif r.get("container"):
            print(f"{n:28s} CONTAINER (excluded; sub-sheets scored separately)")
        else:
            d_ = r["det_f1"] - r["geo_f1"]
            flag = "  <== DET WORSE" if d_ < -0.02 else ("  <== DET BETTER" if d_ > 0.02 else "")
            print(f"{n:28s} geo={r['geo_f1']:.3f} det={r['det_f1']:.3f} "
                  f"d={d_:+.3f} (gt={r['num_gt']},pg={r['pages']}){flag}")
    real = [r for _, r in rows if "geo_f1" in r and r["num_gt"] > 0 and not r.get("container")]
    if real:
        g = sum(r["geo_f1"] for r in real) / len(real)
        de = sum(r["det_f1"] for r in real) / len(real)
        print(f"\nboards={len(real)}  geo mean={g:.4f}  det mean={de:.4f}  delta={de - g:+.4f}")


if __name__ == "__main__":
    main()
