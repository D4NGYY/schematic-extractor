from __future__ import annotations

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.pdf_parser import ExtractedPage, PDFSegment
from src.core.text_associator import SymbolAssociation
from src.ml.detector_source import Detection, DetectorComponentSource


def _page(segments: list[PDFSegment]) -> ExtractedPage:
    return ExtractedPage(page_num=0, segments=segments)


def test_components_assigns_segments_and_classes() -> None:
    # dpi=72 -> 1 px == 1 pt, so boxes_px == boxes_pt.
    segs = [
        PDFSegment(start=(5, 5), end=(15, 5), item_type="line"),    # in box A
        PDFSegment(start=(105, 5), end=(115, 5), item_type="line"),  # in box B
        PDFSegment(start=(55, 5), end=(65, 5), item_type="line"),    # between -> unassigned
    ]
    dets = [
        Detection(class_name="resistor", bbox_px=(0, 0, 20, 20)),
        Detection(class_name="capacitor", bbox_px=(100, 0, 120, 20)),
    ]
    comps = DetectorComponentSource(dpi=72).components(dets, _page(segs))
    by_cls = {c.class_name: c for c in comps}
    assert set(by_cls) == {"resistor", "capacitor"}
    assert len(by_cls["resistor"].cluster.segments) == 1
    assert len(by_cls["capacitor"].cluster.segments) == 1


def test_smallest_box_wins_on_overlap() -> None:
    seg = [PDFSegment(start=(45, 45), end=(55, 55), item_type="line")]  # mid (50,50)
    dets = [
        Detection(class_name="ic", bbox_px=(0, 0, 200, 200)),       # big enclosing
        Detection(class_name="resistor", bbox_px=(40, 40, 60, 60)),  # tight
    ]
    comps = DetectorComponentSource(dpi=72).components(dets, _page(seg))
    # only the tight box gets the segment; the big box has none -> skipped
    assert len(comps) == 1
    assert comps[0].class_name == "resistor"


def test_segmentless_box_skipped() -> None:
    seg = [PDFSegment(start=(5, 5), end=(15, 5), item_type="line")]
    dets = [
        Detection(class_name="resistor", bbox_px=(0, 0, 20, 20)),
        Detection(class_name="diode", bbox_px=(500, 500, 520, 520)),  # empty
    ]
    comps = DetectorComponentSource(dpi=72).components(dets, _page(seg))
    assert [c.class_name for c in comps] == ["resistor"]


def test_pick_ref_nearest_inside_box() -> None:
    box = (0.0, 0.0, 20.0, 20.0)
    refs = [
        SymbolAssociation("R1", "ref", (9, 9), (8, 8, 10, 10), (9, 9), 1.0, 1.0),
        SymbolAssociation("R9", "ref", (200, 200), (0, 0, 0, 0), (200, 200), 1.0, 1.0),
    ]
    assert DetectorComponentSource._pick_ref(refs, box, set()) == "R1"
    assert DetectorComponentSource._pick_ref(refs, box, {"R1"}) is None


def test_build_from_page_uses_detector_components() -> None:
    segs = [PDFSegment(start=(5, 5), end=(15, 5), item_type="line")]
    page = _page(segs)
    dets = [Detection(class_name="resistor", bbox_px=(0, 0, 20, 20), ref="R1")]
    comps = DetectorComponentSource(dpi=72).components(dets, page)
    gb = BipartiteGraphBuilder(cluster_eps=None)
    gb.build_from_page(page, detector_components=comps)
    assert "R1" in gb.components
    assert gb.components["R1"].class_name == "resistor"
