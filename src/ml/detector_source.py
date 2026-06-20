"""Turn component-detector predictions into graph components (detector integration).

The trained detector predicts component BOXES + class (not pins). This module
converts those boxes into `ComponentNode`s whose `cluster` holds the page segments
that fall inside the box, so the existing geometry-based pin extraction
(`BipartiteGraphBuilder.select_pins`) and net connection run unchanged — pins come
from REAL geometry inside each true component boundary, not guessed from wires.
This is the resolutive path for the ref->cluster collision (HANDOFF §22/§23): the
boxes give true component boundaries the geometric clusterer could not.

Detector boxes are in image pixels (the dataset was rendered at some dpi); pass the
SAME dpi used for `scripts/build_detector_dataset.py` to map px -> PDF pt
(pt = px * 72 / dpi). The build is testable without any trained weights by passing
synthetic `Detection`s.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.core.graph_builder import ComponentNode
from src.core.pdf_parser import ExtractedPage, PDFSegment
from src.core.text_associator import TextAssociator
from src.ml.clustering import ComponentCluster


@dataclass(frozen=True)
class Detection:
    """A single detector prediction (box in IMAGE pixels)."""

    class_name: str
    bbox_px: tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels
    confidence: float = 1.0
    ref: str | None = None  # optional pre-assigned ref designator


def _seg_midpoint(s: PDFSegment) -> tuple[float, float]:
    return ((s.start[0] + s.end[0]) / 2.0, (s.start[1] + s.end[1]) / 2.0)


def _point_in(box: tuple[float, float, float, float], x: float, y: float) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def _box_area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


class DetectorComponentSource:
    """Build graph components from detector boxes + page geometry."""

    def __init__(
        self,
        dpi: float,
        text_associator: TextAssociator | None = None,
        box_pad_pt: float = 4.0,
    ) -> None:
        self.px_to_pt = 72.0 / dpi
        self.text_associator = text_associator or TextAssociator()
        # Pad each box outward (pt) before assigning segments so a part's
        # connecting STUBS (which exit the body toward the net) are captured. Their
        # free endpoints then land on the wires, so pins reach nets instead of
        # dangling. 0 = strict body only.
        self.box_pad_pt = box_pad_pt

    def components(
        self, detections: list[Detection], page: ExtractedPage
    ) -> list[ComponentNode]:
        boxes_pt = [
            (
                d,
                tuple(c * self.px_to_pt for c in d.bbox_px),  # type: ignore[misc]
            )
            for d in detections
        ]
        # Assign each segment to the SMALLEST detection box (padded) that contains
        # its midpoint OR either endpoint. Smallest wins (tight part over enclosing
        # IC frame); endpoint match captures stubs that exit the body to the net.
        pad = self.box_pad_pt
        seg_groups: list[list[PDFSegment]] = [[] for _ in boxes_pt]
        for seg in page.segments:
            mx, my = _seg_midpoint(seg)
            pts = ((mx, my), seg.start, seg.end)
            best = -1
            best_area = float("inf")
            for i, (_d, box) in enumerate(boxes_pt):
                pbox = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
                if any(_point_in(pbox, px, py) for px, py in pts):
                    area = _box_area(box)
                    if area < best_area:
                        best_area = area
                        best = i
            if best >= 0:
                seg_groups[best].append(seg)

        # Refs available on the page (to name detected components).
        refs, _vals, _labels = self.text_associator.associate(page)

        out: list[ComponentNode] = []
        used_refs: set[str] = set()
        for i, (det, box) in enumerate(boxes_pt):
            segs = seg_groups[i]
            if not segs:
                continue  # no geometry inside the box -> can't derive pins
            x0, y0, x1, y1 = box
            ref = self._pick_ref(refs, box, used_refs) or det.ref or f"DET{i}"
            used_refs.add(ref)
            cluster = ComponentCluster(
                cluster_id=i,
                segments=segs,
                shapes=[],
                text_blocks=[],
                bbox=(x0, y0, x1, y1),
                center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            )
            out.append(
                ComponentNode(
                    node_id=ref,
                    ref=ref,
                    class_name=det.class_name,
                    cluster=cluster,
                    confidence=det.confidence,
                    bbox=(x0, y0, x1, y1),
                )
            )
        return out

    def components_or_fallback(
        self,
        detections: list[Detection],
        page: ExtractedPage,
        min_frac: float = 0.5,
    ) -> list[ComponentNode] | None:
        """Hybrid gate: return detector components only if the detector found at
        least `min_frac` of the page's ref-designators; otherwise return None so
        the caller falls back to the geometric pipeline.

        Rationale (measured): the detector wins big on some boards but on others
        (small / under-represented) finds nothing and REGRESSES vs geometric
        (e.g. pic_sockets geo 0.625 -> det 0.0). Falling back keeps the wins
        without the crashes. min_frac=0.5 -> need to cover half the visible refs.
        """
        comps = self.components(detections, page)
        refs, _v, _l = self.text_associator.associate(page)
        n_ref = len({r.text for r in refs})
        if n_ref and len(comps) < min_frac * n_ref:
            return None
        return comps

    @staticmethod
    def _pick_ref(refs, box, used_refs):  # type: ignore[no-untyped-def]
        """Nearest unused ref whose anchor is inside the box (else None)."""
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        best = None
        best_d = float("inf")
        for r in refs:
            if r.text in used_refs:
                continue
            ax, ay = r.symbol_center
            if not _point_in(box, ax, ay):
                continue
            d = (ax - cx) ** 2 + (ay - cy) ** 2
            if d < best_d:
                best_d = d
                best = r.text
        return best
