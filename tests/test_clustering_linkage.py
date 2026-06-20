"""Test per il clustering single-linkage su endpoint (sostituisce midpoint-DBSCAN).

Verifica le proprieta chiave del nuovo SpatialClusterer.cluster():
- gruppi di tratti che si toccano restano uniti;
- gruppi distinti che NON si toccano non vengono incatenati (no blob);
- le shapes vicine vengono assegnate al gruppo-segmenti corretto;
- link_dist adattivo data-derived e positivo.
"""
from __future__ import annotations

from src.core.pdf_parser import PDFSegment, PDFShape
from src.ml.clustering import SpatialClusterer


def _seg(x0: float, y0: float, x1: float, y1: float) -> PDFSegment:
    return PDFSegment(start=(x0, y0), end=(x1, y1), item_type="line")


def test_touching_segments_group_together() -> None:
    # Tre tratti che condividono endpoint (un "simbolo") -> un solo cluster.
    segs = [_seg(0, 0, 10, 0), _seg(10, 0, 10, 10), _seg(10, 10, 0, 10)]
    clusters = SpatialClusterer(eps=2.0, min_samples=2).cluster(segs, [])
    assert len(clusters) == 1
    assert clusters[0].num_segments == 3


def test_distant_groups_not_chained() -> None:
    # Due "simboli" lontani: nessun endpoint entro link_dist -> due cluster, mai uno.
    a = [_seg(0, 0, 10, 0), _seg(10, 0, 10, 10)]
    b = [_seg(500, 500, 510, 500), _seg(510, 500, 510, 510)]
    clusters = SpatialClusterer(eps=5.0, min_samples=2).cluster(a + b, [])
    assert len(clusters) == 2
    widths = sorted(c.bbox[2] - c.bbox[0] for c in clusters)
    # Nessun cluster abbraccia entrambi i gruppi (la distanza tra loro e ~500pt).
    assert widths[-1] < 100


def test_dense_chain_does_not_span_when_gapped() -> None:
    # Catena di tratti corti con un gap > link_dist nel mezzo -> NON un blob unico.
    left = [_seg(i * 3, 0, i * 3 + 2, 1) for i in range(5)]      # x: 0..14
    right = [_seg(100 + i * 3, 0, 100 + i * 3 + 2, 1) for i in range(5)]  # x: 100..114
    clusters = SpatialClusterer(eps=5.0, min_samples=2).cluster(left + right, [])
    assert len(clusters) >= 2
    assert all((c.bbox[2] - c.bbox[0]) < 50 for c in clusters)


def test_shape_assigned_to_nearest_group() -> None:
    a = [_seg(0, 0, 10, 0), _seg(10, 0, 10, 10)]
    b = [_seg(500, 500, 510, 500), _seg(510, 500, 510, 510)]
    junction = PDFShape(vertices=[(11, 1), (12, 1), (12, 2), (11, 2)], item_type="rect")
    clusters = SpatialClusterer(eps=5.0, min_samples=2).cluster(a + b, [junction])
    near = next(c for c in clusters if c.bbox[0] < 100)
    far = next(c for c in clusters if c.bbox[0] >= 100)
    assert len(near.shapes) == 1
    assert len(far.shapes) == 0


def test_orphan_shape_dropped() -> None:
    a = [_seg(0, 0, 10, 0), _seg(10, 0, 10, 10)]
    frame = PDFShape(vertices=[(900, 900), (910, 900), (910, 910), (900, 910)], item_type="rect")
    clusters = SpatialClusterer(eps=5.0, min_samples=2).cluster(a, [frame])
    assert sum(len(c.shapes) for c in clusters) == 0


def test_adaptive_link_dist_positive() -> None:
    segs = [_seg(0, 0, 10, 0), _seg(10, 0, 10, 10), _seg(40, 40, 50, 40)]
    ld = SpatialClusterer._estimate_link_dist(segs)
    assert ld >= 5.0


def test_empty_input_returns_empty() -> None:
    assert SpatialClusterer().cluster([], []) == []


def make_cluster(segs, id=0):
    xs = [s.start[0] for s in segs] + [s.end[0] for s in segs]
    ys = [s.start[1] for s in segs] + [s.end[1] for s in segs]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    from src.ml.clustering import ComponentCluster
    return ComponentCluster(cluster_id=id, segments=segs, shapes=[], text_blocks=[], bbox=bbox, center=center)


def test_merge_two_single_segment_clusters():
    c1 = make_cluster([_seg(0, 0, 10, 0)], id=0)
    c2 = make_cluster([_seg(10, 0, 20, 0)], id=1)
    merged = SpatialClusterer._merge_noise_clusters([c1, c2], link_dist=5.0)
    assert len(merged) == 1
    assert merged[0].num_segments == 2


def test_merge_does_not_merge_distant():
    c1 = make_cluster([_seg(0, 0, 10, 0)], id=0)
    c2 = make_cluster([_seg(100, 0, 110, 0)], id=1)
    merged = SpatialClusterer._merge_noise_clusters([c1, c2], link_dist=5.0)
    assert len(merged) == 2


def test_merge_respects_max_size():
    c1 = make_cluster([_seg(0, 0, 10, 0), _seg(10, 0, 20, 0)], id=0)
    c2 = make_cluster([_seg(20, 0, 30, 0), _seg(30, 0, 40, 0)], id=1)
    c3 = make_cluster([_seg(40, 0, 50, 0), _seg(50, 0, 60, 0)], id=2)
    merged = SpatialClusterer._merge_noise_clusters([c1, c2, c3], link_dist=5.0)
    # c1+c2 merge to 4. c3 is 2. 4+2=6 > max_size (5) -> stops merging.
    assert len(merged) == 2


def test_merge_keeps_substantial_clusters():
    c1_segs = [_seg(i * 10, 0, (i + 1) * 10, 0) for i in range(5)]
    c1 = make_cluster(c1_segs, id=0)
    c2 = make_cluster([_seg(50, 0, 60, 0)], id=1)
    merged = SpatialClusterer._merge_noise_clusters([c1, c2], link_dist=5.0)
    assert len(merged) == 2
