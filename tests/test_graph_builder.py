from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.pdf_parser import ExtractedPage, PDFSegment, PDFShape, PDFTextBlock
from src.core.text_associator import SymbolAssociation
from src.ml.classifier import COMPONENT_CLASSES, ComponentClassifier, RuleBasedClassifier
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


def _make_small_cluster(cid: int) -> ComponentCluster:
    """Cluster minuscolo (power symbol tipico): bbox 10×10, 2 seg, 0 shape."""
    return ComponentCluster(
        cluster_id=cid,
        segments=[
            PDFSegment(start=(0, 0), end=(5, 0)),
            PDFSegment(start=(5, 0), end=(5, 5)),
        ],
        shapes=[],
        text_blocks=[],
        bbox=(0.0, 0.0, 10.0, 10.0),
        center=(5.0, 5.0),
    )


def _make_large_cluster(cid: int) -> ComponentCluster:
    """Cluster largo (IC-like): bbox 80×20, 10 seg."""
    segs = [PDFSegment(start=(i * 5.0, 0), end=(i * 5.0 + 5, 0)) for i in range(10)]
    return ComponentCluster(
        cluster_id=cid,
        segments=segs,
        shapes=[],
        text_blocks=[],
        bbox=(0.0, 0.0, 80.0, 20.0),
        center=(40.0, 10.0),
    )


class TestRuleBasedClassifierB1:
    """B1: classificatore rule-based prefisso→classe e fallback geometrico."""

    clf = RuleBasedClassifier()

    # ── segnale primario: prefisso 1 lettera ────────────────────────────────

    @pytest.mark.parametrize("ref,expected_class", [
        ("R1",   "resistor"),
        ("C22",  "capacitor"),
        ("L3",   "inductor"),
        ("D7",   "diode"),
        ("Q5",   "transistor"),
        ("U1",   "ic"),
        ("U1A",  "ic"),          # gate suffix non influenza il prefisso
        ("J3",   "connector"),
        ("P1",   "connector"),
        ("Y1",   "crystal"),
        ("X2",   "crystal"),
        ("F1",   "fuse"),
        ("T1",   "transformer"),
        ("K1",   "relay"),
        ("S1",   "switch"),
    ])
    def test_one_letter_prefix(self, ref: str, expected_class: str) -> None:
        cluster = _make_small_cluster(0)
        class_name, confidence = self.clf.classify(ref, cluster)
        assert class_name == expected_class, f"{ref} → {class_name}, atteso {expected_class}"
        assert confidence >= 0.80

    # ── segnale primario: prefisso 2 lettere con mappa dedicata ─────────────

    @pytest.mark.parametrize("ref,expected_class", [
        ("TP2",  "testpoint"),
        ("VR3",  "regulator"),
        ("SW1",  "switch"),
        ("IC1",  "ic"),
        ("RN4",  "resistor"),
        ("FB1",  "inductor"),
        ("TR1",  "transformer"),
    ])
    def test_two_letter_prefix(self, ref: str, expected_class: str) -> None:
        cluster = _make_small_cluster(0)
        class_name, confidence = self.clf.classify(ref, cluster)
        assert class_name == expected_class, f"{ref} → {class_name}, atteso {expected_class}"
        assert confidence >= 0.85

    # ── prefissi 2-lettere non in mappa → prima lettera ─────────────────────

    @pytest.mark.parametrize("ref,expected_class", [
        ("QB1",  "transistor"),  # QB non in mappa → Q → transistor
        ("RB14", "resistor"),    # RB non in mappa → R → resistor
        ("DX7",  "diode"),       # DX non in mappa → D → diode
        ("CB1",  "capacitor"),   # CB non in mappa → C → capacitor
    ])
    def test_two_letter_prefix_fallback_to_first(self, ref: str, expected_class: str) -> None:
        cluster = _make_small_cluster(0)
        class_name, confidence = self.clf.classify(ref, cluster)
        assert class_name == expected_class, f"{ref} → {class_name}, atteso {expected_class}"
        assert confidence >= 0.90

    # ── tutte le classi restituite devono essere in COMPONENT_CLASSES ────────

    def test_all_results_in_component_classes(self) -> None:
        refs = ["R1", "C1", "L1", "D1", "Q1", "U1", "J1", "Y1", "F1",
                "T1", "K1", "TP1", "VR1", "SW1", "IC1"]
        cluster = _make_small_cluster(0)
        for ref in refs:
            class_name, _ = self.clf.classify(ref, cluster)
            assert class_name in COMPONENT_CLASSES, f"{ref} → classe '{class_name}' non in COMPONENT_CLASSES"

    # ── fallback geometrico (nessun ref) ────────────────────────────────────

    def test_no_ref_small_cluster_power_symbol(self) -> None:
        """Cluster piccolo senza ref → power_symbol (euristico)."""
        cluster = ComponentCluster(
            cluster_id=0, segments=[], shapes=[], text_blocks=[],
            bbox=(0.0, 0.0, 10.0, 10.0), center=(5.0, 5.0),
        )
        class_name, confidence = self.clf.classify(None, cluster)
        assert class_name in COMPONENT_CLASSES
        assert confidence < 0.70  # fallback geometrico ha confidence bassa

    def test_no_ref_large_ic_cluster(self) -> None:
        """Cluster largo (AR>3, molti seg) senza ref → ic o unknown."""
        cluster = _make_large_cluster(0)
        class_name, confidence = self.clf.classify(None, cluster)
        assert class_name in COMPONENT_CLASSES
        assert confidence < 0.70

    def test_confidence_high_when_ref_present(self) -> None:
        """Con ref il classificatore deve avere confidence ≥ 0.80."""
        cluster = _make_small_cluster(0)
        _, confidence = self.clf.classify("R1", cluster)
        assert confidence >= 0.80


# ── D6: _estimate_scale ───────────────────────────────────────────────────────

class TestEstimateScaleD6:
    """D6: stima scala caratteristica dai segmenti."""

    def test_empty_returns_one(self) -> None:
        assert BipartiteGraphBuilder._estimate_scale([]) == 1.0

    def test_degenerate_zero_length_ignored(self) -> None:
        """Segmenti di lunghezza nulla non devono influire sulla stima."""
        segs = [PDFSegment(start=(0.0, 0.0), end=(0.0, 0.0), item_type="line")]
        assert BipartiteGraphBuilder._estimate_scale(segs) == 1.0

    def test_single_segment_returns_length(self) -> None:
        seg = PDFSegment(start=(0.0, 0.0), end=(10.0, 0.0), item_type="line")
        scale = BipartiteGraphBuilder._estimate_scale([seg])
        assert scale == pytest.approx(10.0, abs=0.01)

    def test_p10_of_lengths(self) -> None:
        """Con 10 segmenti, p10 deve essere il primo della lista ordinata."""
        segs = [
            PDFSegment(start=(0.0, 0.0), end=(float(i), 0.0), item_type="line")
            for i in range(1, 11)  # lunghezze 1..10
        ]
        scale = BipartiteGraphBuilder._estimate_scale(segs)
        # p10 = lengths[max(0, int(10*0.10)-1)] = lengths[0] = 1.0
        assert scale == pytest.approx(1.0, abs=0.01)

    def test_large_scale_schematic(self) -> None:
        """Segmenti di 50pt → scala ≥ 50."""
        segs = [
            PDFSegment(start=(0.0, 0.0), end=(50.0, 0.0), item_type="line"),
            PDFSegment(start=(0.0, 0.0), end=(100.0, 0.0), item_type="line"),
        ]
        scale = BipartiteGraphBuilder._estimate_scale(segs)
        assert scale >= 50.0


# ── D6: T-junction in _segments_intersect ────────────────────────────────────

class TestTJunctionIntersectionD6:
    """D6: stub che termina esattamente sul corpo del filo deve essere rilevato."""

    def test_t_junction_stub_end_on_wire(self) -> None:
        """Stub orizzontale il cui end tocca il filo verticale (T-junction)."""
        stub = PDFSegment(start=(0.0, 0.0), end=(10.0, 0.0), item_type="line")
        wire = PDFSegment(start=(10.0, -5.0), end=(10.0, 5.0), item_type="line")
        assert BipartiteGraphBuilder._segments_intersect(stub, wire)

    def test_t_junction_stub_start_on_wire(self) -> None:
        """Filo che inizia esattamente dove finisce lo stub (T-junction inversa)."""
        stub = PDFSegment(start=(0.0, 0.0), end=(10.0, 0.0), item_type="line")
        wire = PDFSegment(start=(0.0, -5.0), end=(0.0, 5.0), item_type="line")
        assert BipartiteGraphBuilder._segments_intersect(stub, wire)

    def test_non_touching_parallel_still_false(self) -> None:
        """Segmenti paralleli non sovrapposti non devono intersecarsi."""
        a = PDFSegment(start=(0.0, 0.0), end=(10.0, 0.0), item_type="line")
        b = PDFSegment(start=(0.0, 5.0), end=(10.0, 5.0), item_type="line")
        assert not BipartiteGraphBuilder._segments_intersect(a, b)


# ── Wire/symbol separation ────────────────────────────────────────────────────

class TestSeparateWires:
    """Separazione wire-candidate da symbol-primitive prima del clustering."""

    def test_horizontal_is_axis_aligned(self) -> None:
        seg = PDFSegment(start=(0.0, 0.0), end=(100.0, 0.0), item_type="line")
        assert SpatialClusterer._is_axis_aligned(seg)

    def test_vertical_is_axis_aligned(self) -> None:
        seg = PDFSegment(start=(0.0, 0.0), end=(0.0, 100.0), item_type="line")
        assert SpatialClusterer._is_axis_aligned(seg)

    def test_diagonal_not_axis_aligned(self) -> None:
        seg = PDFSegment(start=(0.0, 0.0), end=(10.0, 10.0), item_type="line")
        assert not SpatialClusterer._is_axis_aligned(seg)

    def test_nearly_horizontal_within_tol(self) -> None:
        """Segmento con inclinazione 3° → axis-aligned (entro 5°)."""
        import math
        dx = math.cos(math.radians(3)) * 10
        dy = math.sin(math.radians(3)) * 10
        seg = PDFSegment(start=(0.0, 0.0), end=(dx, dy), item_type="line")
        assert SpatialClusterer._is_axis_aligned(seg)

    def test_empty_returns_empty(self) -> None:
        sym, wire = SpatialClusterer.separate_wires([])
        assert sym == [] and wire == []

    def test_uniform_short_all_become_symbol(self) -> None:
        """Segmenti tutti corti (test sintetici 10pt) → tutti symbol, zero wire.
        Garantisce che i test sintetici non cambino comportamento.
        """
        segs = [
            PDFSegment(start=(0.0, 0.0), end=(10.0, 0.0), item_type="line"),  # 10pt H
            PDFSegment(start=(0.0, 0.0), end=(0.0, 10.0), item_type="line"),  # 10pt V
        ]
        # p25=10, threshold=max(5,30)=30 → entrambi i 10pt stanno sotto soglia → symbol
        sym, wire = SpatialClusterer.separate_wires(segs)
        assert len(wire) == 0
        assert len(sym) == 2

    def test_long_axis_aligned_becomes_wire(self) -> None:
        short = PDFSegment(start=(0.0, 0.0), end=(5.0, 0.0), item_type="line")   # 5pt
        long_ = PDFSegment(start=(0.0, 0.0), end=(100.0, 0.0), item_type="line") # 100pt
        # p25([5,100]) ≈ 5, threshold=max(5, 5*3)=15 → 100pt >= 15 → wire
        sym, wire = SpatialClusterer.separate_wires([short, long_])
        assert long_ in wire
        assert short in sym

    def test_long_diagonal_stays_symbol(self) -> None:
        """Filo lungo ma diagonale → symbol (non asse-allineato)."""
        diag = PDFSegment(start=(0.0, 0.0), end=(100.0, 100.0), item_type="line")  # 45°
        sym, wire = SpatialClusterer.separate_wires([diag])
        assert diag in sym
        assert diag not in wire

class TestWireMergeW1:
    def test_derive_wire_tol_adaptive(self) -> None:
        segs = [
            PDFSegment(start=(0, 0), end=(10, 0), item_type="line"),
            PDFSegment(start=(5, 0), end=(5, 10), item_type="line"),
        ]
        tol = BipartiteGraphBuilder._derive_wire_tol(segs)
        assert tol < 5.0

    def test_wire_merge_two_collinear_touching(self) -> None:
        builder = BipartiteGraphBuilder()
        builder.components = {}
        segs = [
            PDFSegment(start=(0, 0), end=(10, 0), item_type="line"),
            PDFSegment(start=(10, 0), end=(20, 0), item_type="line"),
        ]
        builder._build_nets(segs, scale=1.0)
        assert len(builder.nets) == 1

    def test_wire_merge_two_collinear_gap(self) -> None:
        builder = BipartiteGraphBuilder()
        builder.components = {}
        segs = [
            PDFSegment(start=(0, 0), end=(10, 0), item_type="line"),
            PDFSegment(start=(10.5, 0), end=(20, 0), item_type="line"),
        ]
        builder._build_nets(segs, scale=1.0)
        assert len(builder.nets) == 1

    def test_wire_merge_no_connection(self) -> None:
        builder = BipartiteGraphBuilder()
        builder.components = {}
        segs = [
            PDFSegment(start=(0, 0), end=(10, 0), item_type="line"),
            PDFSegment(start=(11, 0), end=(12, 0), item_type="line"),
            PDFSegment(start=(20, 0), end=(30, 0), item_type="line"),
        ]
        builder._build_nets(segs, scale=1.0)
        assert len(builder.nets) == 2

    def test_t_junction_recognized(self) -> None:
        """T-junction must be merged into 1 net."""
        builder = BipartiteGraphBuilder()
        builder.components = {}
        segs = [
            PDFSegment(start=(0, 0), end=(10, 0), item_type="line"),  # Wire A
            PDFSegment(start=(5, 0), end=(5, 5), item_type="line"),   # Wire B
        ]
        builder._build_nets(segs, scale=1.0)
        assert len(builder.nets) == 1

    def test_l_junction_recognized(self) -> None:
        """L-junction (shared endpoint) must be merged into 1 net."""
        builder = BipartiteGraphBuilder()
        builder.components = {}
        segs = [
            PDFSegment(start=(0, 0), end=(5, 0), item_type="line"),   # Wire A
            PDFSegment(start=(5, 0), end=(5, 5), item_type="line"),   # Wire B
        ]
        builder._build_nets(segs, scale=1.0)
        assert len(builder.nets) == 1

    def test_crossing_not_junction_if_no_shared_endpoint(self) -> None:
        """Crossing wires without a shared endpoint or T-junction should NOT merge."""
        builder = BipartiteGraphBuilder()
        builder.components = {}
        segs = [
            PDFSegment(start=(0, 5), end=(10, 5), item_type="line"),  # Wire A
            PDFSegment(start=(5, 0), end=(5, 10), item_type="line"),  # Wire B
            # Dummy segments to force p25 of min_dist to be 0 (tol=1.0)
            PDFSegment(start=(10, 5), end=(20, 5), item_type="line"),
            PDFSegment(start=(5, 10), end=(5, 20), item_type="line"),
        ]
        builder._build_nets(segs, scale=1.0)
        # Expected behavior: Wire A and B cross but don't touch ends. 
        # But Wire A merges with dummy 1, and Wire B merges with dummy 2.
        # Total nets = 2
        assert len(builder.nets) == 2

    def test_no_false_merge_distant_wires(self) -> None:
        """Distant wires should be in separate nets."""
        builder = BipartiteGraphBuilder()
        builder.components = {}
        segs = [
            PDFSegment(start=(0, 0), end=(10, 0), item_type="line"),
            PDFSegment(start=(50, 50), end=(60, 50), item_type="line"),
            # Dummy segments to force p25 of min_dist to be 0 (tol=1.0)
            PDFSegment(start=(10, 0), end=(20, 0), item_type="line"),
            PDFSegment(start=(60, 50), end=(70, 50), item_type="line"),
        ]
        builder._build_nets(segs, scale=1.0)
        assert len(builder.nets) == 2

class TestPinSelectionP1:
    def test_select_pins_resistor_zigzag(self) -> None:
        cluster = ComponentCluster(
            cluster_id=0,
            segments=[
                PDFSegment(start=(0, 0), end=(5, 5), item_type="line"),
                PDFSegment(start=(5, 5), end=(10, -5), item_type="line"),
                PDFSegment(start=(10, -5), end=(15, 5), item_type="line"),
                PDFSegment(start=(15, 5), end=(20, 0), item_type="line"),
            ],
            shapes=[],
            text_blocks=[],
            bbox=(0, -5, 20, 5),
            center=(10, 0)
        )
        pins = BipartiteGraphBuilder.select_pins(cluster)
        assert len(pins) == 2
        assert set(pins) == {(0, 0), (20, 0)}

    def test_select_pins_ic_rectangular(self) -> None:
        cluster = ComponentCluster(
            cluster_id=0,
            segments=[
                PDFSegment(start=(0, 10), end=(-5, 10), item_type="line"),
                PDFSegment(start=(0, 20), end=(-5, 20), item_type="line"),
                PDFSegment(start=(30, 10), end=(35, 10), item_type="line"),
                PDFSegment(start=(30, 20), end=(35, 20), item_type="line"),
            ],
            shapes=[
                PDFShape(vertices=[(0, 0), (30, 0), (30, 30), (0, 30)], item_type="rect")
            ],
            text_blocks=[],
            bbox=(-5, 0, 35, 30),
            center=(15, 15)
        )
        pins = BipartiteGraphBuilder.select_pins(cluster)
        assert len(pins) == 4

    def test_select_pins_cluster_with_internal_strokes(self) -> None:
        cluster = ComponentCluster(
            cluster_id=0,
            segments=[
                PDFSegment(start=(-5, 5), end=(15, 5), item_type="line"),
                PDFSegment(start=(5, 4), end=(5, 6), item_type="line"),
            ],
            shapes=[],
            text_blocks=[],
            bbox=(-5, 2, 15, 8),
            center=(5, 5)
        )
        pins = BipartiteGraphBuilder.select_pins(cluster)
        assert len(pins) == 2
        assert set(pins) == {(-5, 5), (15, 5)}

class TestPinConnectP2:
    def test_pin_connects_when_on_wire(self) -> None:
        builder = BipartiteGraphBuilder()
        assert builder._point_to_seg_d2((5, 0), PDFSegment(start=(0, 0), end=(10, 0), item_type="line")) == 0.0

    def test_pin_connects_within_3x_wire_tol(self) -> None:
        builder = BipartiteGraphBuilder()
        d2 = builder._point_to_seg_d2((5, 2.5), PDFSegment(start=(0, 0), end=(10, 0), item_type="line"))
        assert d2 == 6.25
        assert d2 <= (3.0 * 1.0)**2

    def test_pin_not_connected_beyond_tol(self) -> None:
        builder = BipartiteGraphBuilder()
        d2 = builder._point_to_seg_d2((5, 10.0), PDFSegment(start=(0, 0), end=(10, 0), item_type="line"))
        assert d2 == 100.0
        assert d2 > (3.0 * 1.0)**2


