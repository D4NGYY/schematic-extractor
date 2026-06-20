from __future__ import annotations

from src.core.pdf_parser import PDFSegment, PDFShape
from src.ml.clustering import ComponentCluster, SpatialClusterer


def _seg(x0: float, y0: float, x1: float, y1: float) -> PDFSegment:
    return PDFSegment(start=(x0, y0), end=(x1, y1), item_type="line")


# ── FASE 3 TESTS: CLASSIFICATION ─────────────────────────────────────────────


def test_classify_cluster_wire_fragment() -> None:
    cluster = ComponentCluster(
        cluster_id=0,
        segments=[_seg(0, 0, 10, 0), _seg(10, 0, 20, 0)],
        shapes=[],
        text_blocks=[],
        bbox=(0, 0, 20, 0),
        center=(10, 0),
    )
    assert SpatialClusterer._classify_cluster_by_shape(cluster) == "wire_fragment"


def test_classify_cluster_wire_path() -> None:
    cluster = ComponentCluster(
        cluster_id=0,
        segments=[_seg(0, 0, 10, 0), _seg(10, 0, 10, 20), _seg(10, 20, 40, 20)],
        shapes=[],
        text_blocks=[],
        bbox=(0, 0, 40, 20),
        center=(20, 10),
    )
    cluster = ComponentCluster(
        cluster_id=0,
        segments=[
            _seg(0, 0, 10, 0),
            _seg(10, 0, 10, 10),
            _seg(10, 10, 40, 10),
            _seg(40, 10, 40, 20),
        ],
        shapes=[],
        text_blocks=[],
        bbox=(0, 0, 40, 10),
        center=(20, 5),
    )
    cluster = ComponentCluster(
        cluster_id=0,
        segments=[_seg(0, 0, 10, 0), _seg(10, 0, 20, 0), _seg(20, 0, 30, 0), _seg(30, 0, 40, 0)],
        shapes=[],
        text_blocks=[],
        bbox=(0, 0, 40, 0),
        center=(20, 0),
    )
    assert SpatialClusterer._classify_cluster_by_shape(cluster) == "wire_path"


def test_classify_cluster_symbol_compact() -> None:
    cluster = ComponentCluster(
        cluster_id=0,
        segments=[_seg(0, 0, 5, 5), _seg(5, 5, 10, 0), _seg(10, 0, 5, -5)],
        shapes=[],
        text_blocks=[],
        bbox=(0, -5, 10, 5),
        center=(5, 0),
    )
    assert SpatialClusterer._classify_cluster_by_shape(cluster) == "symbol"


def test_classify_cluster_symbol_rectangular() -> None:
    cluster = ComponentCluster(
        cluster_id=0,
        segments=[_seg(0, 0, 10, 0), _seg(10, 0, 10, 10), _seg(10, 10, 0, 10), _seg(0, 10, 0, 0)],
        shapes=[],
        text_blocks=[],
        bbox=(0, 0, 10, 10),
        center=(5, 5),
    )
    assert SpatialClusterer._classify_cluster_by_shape(cluster) == "symbol"


def test_cluster_returns_split_lists() -> None:
    sym = [_seg(0, 0, 10, 0), _seg(10, 0, 10, 10), _seg(10, 10, 0, 10), _seg(0, 10, 0, 0)]
    wir = [_seg(100, 100, 110, 100)]
    components, wires = SpatialClusterer(eps=5.0, min_samples=1).cluster(sym + wir, [])
    assert len(components) == 1
    assert len(wires) == 1
    assert components[0].num_segments == 4
    assert wires[0].num_segments == 1


# ── recover_absorbed_wires ───────────────────────────────────────────────────


def test_recover_absorbed_wires_extracts_outside() -> None:
    sh = PDFShape(vertices=[(0, 0), (10, 0), (10, 10), (0, 10)], item_type="rect")
    seg = _seg(5, 5, 50, 5)
    cluster = ComponentCluster(
        cluster_id=0,
        segments=[seg],
        shapes=[sh],
        text_blocks=[],
        bbox=(0, 0, 50, 10),
        center=(25, 5),
    )
    wires = []
    recovered = SpatialClusterer.recover_absorbed_wires(
        [cluster], wires, link_dist=5.0, min_samples=1
    )
    assert len(recovered) == 1
    out_seg = recovered[0]
    assert out_seg.start == (10, 5)
    assert out_seg.end == (50, 5)
    assert len(cluster.segments) == 1
    in_seg = cluster.segments[0]
    assert in_seg.start == (5, 5)
    assert in_seg.end == (10, 5)
    assert len(wires) == 1


def test_recover_absorbed_wires_keeps_internal() -> None:
    sh = PDFShape(vertices=[(0, 0), (20, 0), (20, 20), (0, 20)], item_type="rect")
    seg = _seg(5, 5, 15, 5)
    cluster = ComponentCluster(
        cluster_id=0,
        segments=[seg],
        shapes=[sh],
        text_blocks=[],
        bbox=(0, 0, 20, 20),
        center=(10, 10),
    )
    wires = []
    recovered = SpatialClusterer.recover_absorbed_wires(
        [cluster], wires, link_dist=5.0, min_samples=1
    )
    assert len(recovered) == 0
    assert len(cluster.segments) == 1
    assert cluster.segments[0].start == (5, 5)


def test_recover_absorbed_wires_respects_min_samples() -> None:
    seg = _seg(0, 0, 50, 0)
    cluster = ComponentCluster(
        cluster_id=0, segments=[seg], shapes=[], text_blocks=[], bbox=(0, 0, 50, 0), center=(25, 0)
    )
    wires = []
    recovered = SpatialClusterer.recover_absorbed_wires(
        [cluster], wires, link_dist=5.0, min_samples=2
    )
    assert len(recovered) == 0
    assert len(cluster.segments) == 1
