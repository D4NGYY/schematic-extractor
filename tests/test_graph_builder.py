from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.pdf_parser import ExtractedPage, PDFSegment, PDFShape, PDFTextBlock
from src.core.text_associator import SymbolAssociation
from src.ml.classifier import COMPONENT_CLASSES, ComponentClassifier
from src.ml.clustering import ComponentCluster, SpatialClusterer
from src.ml.feature_extractor import FeatureExtractor, FeatureVector


class TestSpatialClusterer:
    def test_cluster_lines(self) -> None:
        """Due segmenti vicini formano un cluster; uno lontano è noise."""
        segments = [
            PDFSegment(start=(0, 0), end=(10, 0), item_type="line"),
            PDFSegment(start=(0, 1), end=(10, 1), item_type="line"),  # vicino al primo
            PDFSegment(start=(100, 100), end=(200, 100), item_type="line"),  # lontano
        ]
        clusterer = SpatialClusterer(eps=20.0, min_samples=2)
        clusters = clusterer.cluster(segments, shapes=[])
        assert len(clusters) >= 1
        # Il cluster con i due segmenti vicini deve esistere
        assert any(len(c.segments) >= 2 for c in clusters)

    def test_empty_input(self) -> None:
        assert SpatialClusterer().cluster([], []) == []

    def test_estimate_eps(self) -> None:
        x = np.array([[0, 0], [1, 0], [2, 0], [100, 100]])
        eps = SpatialClusterer._estimate_eps(x)
        assert eps >= 5.0


class TestFeatureExtractor:
    def test_extract_from_cluster(self) -> None:
        cluster = ComponentCluster(
            cluster_id=0,
            segments=[
                PDFSegment(start=(0, 0), end=(10, 0), item_type="line"),
                PDFSegment(start=(0, 0), end=(0, 10), item_type="line"),
            ],
            shapes=[
                PDFShape(vertices=[(0, 0), (10, 0), (10, 10), (0, 10)], item_type="rect"),
            ],
            text_blocks=[],
            bbox=(0, 0, 10, 10),
            center=(5, 5),
        )
        fe = FeatureExtractor()
        fv = fe.extract(cluster)
        assert isinstance(fv, FeatureVector)
        arr = fv.to_array()
        assert arr.shape == (13,)
        assert arr[0] == pytest.approx(1.0)  # aspect_ratio 10/10 = 1
        assert arr[4] == pytest.approx(1.0)  # solidity ≈ 1 per rettangolo

    def test_empty_cluster(self) -> None:
        cluster = ComponentCluster(
            cluster_id=0,
            segments=[],
            shapes=[],
            text_blocks=[],
            bbox=(0, 0, 0, 0),
            center=(0, 0),
        )
        fe = FeatureExtractor()
        fv = fe.extract(cluster)
        arr = fv.to_array()
        assert arr.shape == (13,)
        assert np.isfinite(arr).all()


class TestComponentClassifier:
    def test_train_and_predict(self) -> None:
        # Mock training data: 2 classi, 2 campioni ciascuna
        features = np.array([
            [1.0, 0.5, 1.0, 1.0, 0.9, 0.5, 0, 0.8, 50, 50, 2, 2, 0.5],  # resistor-like
            [1.0, 0.5, 1.0, 1.0, 0.9, 0.5, 0, 0.8, 50, 50, 2, 2, 0.5],  # resistor-like
            [2.0, 1.0, 2.0, 2.0, 0.8, 1.0, 0, 0.5, 100, 100, 5, 5, 1.0],  # capacitor-like
            [2.0, 1.0, 2.0, 2.0, 0.8, 1.0, 0, 0.5, 100, 100, 5, 5, 1.0],  # capacitor-like
        ])
        y = np.array(["resistor", "resistor", "capacitor", "capacitor"])

        clf = ComponentClassifier()
        clf.fit(features, y)

        # Predici su un cluster simile a resistor
        cluster = ComponentCluster(
            cluster_id=99,
            segments=[PDFSegment(start=(0, 0), end=(10, 0), item_type="line")],
            shapes=[],
            text_blocks=[],
            bbox=(0, 0, 10, 10),
            center=(5, 5),
        )
        # Dobbiamo bypassare il predict con model untrained -> invece il model è addestrato
        class_name, conf = clf.predict(cluster)
        assert class_name in COMPONENT_CLASSES or class_name == "unknown"
        assert 0.0 <= conf <= 1.0

    def test_untrained_raises(self) -> None:
        clf = ComponentClassifier()
        cluster = ComponentCluster(
            cluster_id=0,
            segments=[],
            shapes=[],
            text_blocks=[],
            bbox=(0, 0, 1, 1),
            center=(0.5, 0.5),
        )
        with pytest.raises(RuntimeError, match="Model not trained"):
            clf.predict(cluster)


class TestBipartiteGraphBuilder:
    def test_build_empty(self) -> None:
        page = ExtractedPage(page_num=0)
        builder = BipartiteGraphBuilder()
        g = builder.build_from_page(page)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_segments_intersect(self) -> None:
        a = PDFSegment(start=(0, 0), end=(10, 10), item_type="line")
        b = PDFSegment(start=(0, 10), end=(10, 0), item_type="line")
        assert BipartiteGraphBuilder._segments_intersect(a, b)

    def test_segments_no_intersect(self) -> None:
        a = PDFSegment(start=(0, 0), end=(10, 0), item_type="line")
        b = PDFSegment(start=(0, 10), end=(10, 10), item_type="line")
        assert not BipartiteGraphBuilder._segments_intersect(a, b)

    def test_export_json_structure(self) -> None:
        page = ExtractedPage(page_num=0)
        page.segments.append(PDFSegment(start=(0, 0), end=(10, 0), item_type="line"))
        page.segments.append(PDFSegment(start=(10, 0), end=(10, 10), item_type="line"))
        page.shapes.append(
            PDFShape(vertices=[(-5, -5), (5, -5), (5, 5), (-5, 5)], item_type="rect")
        )
        page.text_blocks.append(PDFTextBlock("R1", (-5, -5, 5, 5)))
        page.text_blocks.append(PDFTextBlock("1k", (15, 5, 25, 15)))

        builder = BipartiteGraphBuilder()
        _ = builder.build_from_page(page)
        data = builder.export_json("data/ground_truth/test_graph.json")

        assert "components" in data
        assert "nets" in data
        assert "edges" in data

    def test_export_spice(self) -> None:
        page = ExtractedPage(page_num=0)
        page.segments.append(PDFSegment(start=(0, 0), end=(10, 0), item_type="line"))
        page.shapes.append(
            PDFShape(vertices=[(0, 0), (5, 0), (5, 5), (0, 5)], item_type="rect")
        )
        page.text_blocks.append(PDFTextBlock("R1", (0, 0, 5, 5)))

        builder = BipartiteGraphBuilder()
        builder.build_from_page(page)
        builder.export_spice("data/ground_truth/test.cir")
        content = Path("data/ground_truth/test.cir").read_text()
        assert ".end" in content
        assert "R1" in content

    def test_export_kicad_netlist(self) -> None:
        page = ExtractedPage(page_num=0)
        page.segments.append(PDFSegment(start=(0, 0), end=(10, 0), item_type="line"))
        page.shapes.append(
            PDFShape(vertices=[(0, 0), (5, 0), (5, 5), (0, 5)], item_type="rect")
        )
        page.text_blocks.append(PDFTextBlock("R1", (0, 0, 5, 5)))

        builder = BipartiteGraphBuilder()
        builder.build_from_page(page)
        builder.export_kicad_netlist("data/ground_truth/test.net")
        content = Path("data/ground_truth/test.net").read_text()
        assert "(export" in content
        assert "R1" in content


class TestEpsEstimationD2:
    """D2: k-NN eps non deve essere enorme su dati bimodali (bug pdist)."""

    def test_knn_eps_local_for_clustered_data(self) -> None:
        """Due cluster densi molto distanti: eps deve riflettere densità locale."""
        rng = np.random.default_rng(42)
        cluster1 = rng.normal([0, 0], 2.0, (50, 2))
        cluster2 = rng.normal([300, 300], 2.0, (50, 2))
        x = np.vstack([cluster1, cluster2])
        eps = SpatialClusterer._estimate_eps(x)
        # pdist darebbe ~212 (mediana inter-cluster), k-NN deve dare << 50
        assert eps < 50.0, f"eps troppo grande ({eps:.1f}): il bug pdist è tornato?"

    def test_eps_minimum_floor(self) -> None:
        """eps non scende mai sotto 5.0."""
        x = np.array([[0.0, 0.0], [0.001, 0.0]])
        assert SpatialClusterer._estimate_eps(x) >= 5.0

    def test_single_point_returns_fallback(self) -> None:
        x = np.array([[5.0, 5.0]])
        assert SpatialClusterer._estimate_eps(x) == 10.0


def _make_cluster(cid: int, cx: float, cy: float) -> ComponentCluster:
    return ComponentCluster(
        cluster_id=cid,
        segments=[],
        shapes=[],
        text_blocks=[],
        bbox=(cx - 5, cy - 5, cx + 5, cy + 5),
        center=(cx, cy),
    )


def _make_assoc(text: str, sx: float, sy: float, dist: float = 1.0) -> SymbolAssociation:
    return SymbolAssociation(
        text=text,
        text_type="ref",
        text_pos=(sx + 20, sy + 20),  # testo lontano dal simbolo (D5 test)
        symbol_bbox=(sx - 5, sy - 5, sx + 5, sy + 5),
        symbol_center=(sx, sy),
        distance=dist,
        confidence=0.9,
    )


class TestD4NoCollision:
    """D4: due ref che puntano allo stesso cluster non si sovrascrivono."""

    def test_both_refs_accumulated(self) -> None:
        cluster = _make_cluster(0, 5.0, 5.0)
        r1 = _make_assoc("R1", 4.9, 5.0, dist=0.1)
        r2 = _make_assoc("R2", 5.1, 5.1, dist=0.2)

        ref_map: dict[int, list[SymbolAssociation]] = {}
        for r in [r1, r2]:
            cid = BipartiteGraphBuilder._nearest_cluster(r, [cluster])
            ref_map.setdefault(cid, []).append(r)

        assert 0 in ref_map
        assert len(ref_map[0]) == 2

    def test_closest_ref_wins(self) -> None:
        """Quando due ref si mappano allo stesso cluster, vince il più vicino."""
        cluster = _make_cluster(0, 5.0, 5.0)
        close = _make_assoc("CLOSE", 5.0, 5.0, dist=0.1)
        far = _make_assoc("FAR", 4.0, 4.0, dist=5.0)

        ref_map: dict[int, list[SymbolAssociation]] = {}
        for r in [far, close]:  # inserisco FAR per primo per testare l'ordinamento
            cid = BipartiteGraphBuilder._nearest_cluster(r, [cluster])
            ref_map.setdefault(cid, []).append(r)

        winner = min(ref_map[0], key=lambda a: a.distance)
        assert winner.text == "CLOSE"


class TestD5SymbolCenter:
    """D5: _nearest_cluster usa symbol_center (non text_pos)."""

    def test_picks_cluster_near_symbol_not_text(self) -> None:
        """symbol_center vicino a cluster 0, text_pos vicino a cluster 1 → cluster 0."""
        c0 = _make_cluster(0, 10.0, 10.0)
        c1 = _make_cluster(1, 200.0, 200.0)
        # symbol_center=(10,10) → vicino c0; text_pos=(199,199) → vicino c1
        assoc = SymbolAssociation(
            text="R1",
            text_type="ref",
            text_pos=(199.0, 199.0),
            symbol_bbox=(5, 5, 15, 15),
            symbol_center=(10.0, 10.0),
            distance=1.0,
            confidence=0.9,
        )
        result = BipartiteGraphBuilder._nearest_cluster(assoc, [c0, c1])
        assert result == 0
