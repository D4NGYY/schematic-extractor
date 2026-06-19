from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.pdf_parser import ExtractedPage, PDFSegment, PDFShape, PDFTextBlock
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
