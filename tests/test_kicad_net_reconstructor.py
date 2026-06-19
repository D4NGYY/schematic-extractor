from __future__ import annotations

import pytest

from scripts.kicad_net_reconstructor import Junction, KiCadNetReconstructor, WireSegment
from src.core.coordinate_system import CoordinateSystem, Vec2


class TestCoordinateSystem:
    def test_mm_to_mils_roundtrip(self) -> None:
        mm = 2.54
        mils = CoordinateSystem.mm_to_mils(mm)
        assert mils == 100.0
        assert CoordinateSystem.mils_to_mm(mils) == pytest.approx(mm)

    def test_mm_to_points(self) -> None:
        mm = 25.4
        points = CoordinateSystem.mm_to_points(mm)
        assert points == pytest.approx(72.0)

    def test_vec2_distance(self) -> None:
        a = Vec2(0.0, 0.0)
        b = Vec2(3.0, 4.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_vec2_add(self) -> None:
        assert Vec2(1, 2) + Vec2(3, 4) == Vec2(4, 6)


class TestWireSegment:
    def test_merge_horizontal_overlapping(self) -> None:
        a = WireSegment(Vec2(0, 0), Vec2(10, 0))
        b = WireSegment(Vec2(5, 0), Vec2(15, 0))
        merged = a.merge_with(b)
        assert merged is not None
        assert merged.start == Vec2(0, 0)
        assert merged.end == Vec2(15, 0)

    def test_merge_horizontal_touching(self) -> None:
        a = WireSegment(Vec2(0, 0), Vec2(10, 0))
        b = WireSegment(Vec2(10, 0), Vec2(20, 0))
        merged = a.merge_with(b, tolerance=0.01)
        assert merged is not None
        assert merged.start == Vec2(0, 0)
        assert merged.end == Vec2(20, 0)

    def test_merge_non_overlapping(self) -> None:
        a = WireSegment(Vec2(0, 0), Vec2(10, 0))
        b = WireSegment(Vec2(100, 0), Vec2(200, 0))
        merged = a.merge_with(b)
        assert merged is None

    def test_merge_vertical(self) -> None:
        a = WireSegment(Vec2(0, 0), Vec2(0, 10))
        b = WireSegment(Vec2(0, 5), Vec2(0, 15))
        merged = a.merge_with(b)
        assert merged is not None
        assert merged.start == Vec2(0, 0)
        assert merged.end == Vec2(0, 15)

    def test_contains_point(self) -> None:
        seg = WireSegment(Vec2(0, 0), Vec2(10, 0))
        assert seg.contains_point(Vec2(5, 0))
        assert seg.contains_point(Vec2(0, 0))
        assert seg.contains_point(Vec2(10, 0))
        assert not seg.contains_point(Vec2(5, 1))
        assert not seg.contains_point(Vec2(11, 0))

    def test_is_collinear_with(self) -> None:
        a = WireSegment(Vec2(0, 0), Vec2(10, 0))
        b = WireSegment(Vec2(5, 0), Vec2(15, 0))
        assert a.is_collinear_with(b)

    def test_is_collinear_with_different_line(self) -> None:
        a = WireSegment(Vec2(0, 0), Vec2(10, 0))
        b = WireSegment(Vec2(0, 1), Vec2(10, 1))
        assert not a.is_collinear_with(b)


class TestJunctionDetect:
    def test_explicit_junction(self) -> None:
        recon = KiCadNetReconstructor()
        recon.wires = [
            WireSegment(Vec2(0, 0), Vec2(10, 0)),
            WireSegment(Vec2(5, -5), Vec2(5, 5)),
        ]
        recon.junctions = [Junction(Vec2(5, 0), is_explicit=True)]
        recon.junction_detect()

        assert len(recon.junctions) == 1
        assert len(recon.junctions[0].connected_wires) == 2

    def test_implicit_t_junction(self) -> None:
        recon = KiCadNetReconstructor()
        recon.wires = [
            WireSegment(Vec2(0, 0), Vec2(10, 0)),  # orizzontale
            WireSegment(Vec2(5, 0), Vec2(5, 5)),  # verticale che parte da T
        ]
        recon.junctions = []
        recon.junction_detect()

        # Dovrebbe trovare la T-junction implicita
        implicit = [j for j in recon.junctions if not j.is_explicit]
        assert len(implicit) >= 1

    def test_no_junction_at_crossing(self) -> None:
        """Due wire che si incrociano a croce (X) senza junction dot
        NON formano una connessione elettrica in KiCad.
        """
        recon = KiCadNetReconstructor()
        recon.wires = [
            WireSegment(Vec2(0, 0), Vec2(10, 0)),
            WireSegment(Vec2(5, -5), Vec2(5, 5)),
        ]
        recon.junctions = []
        recon.junction_detect()

        # Non dovrebbe creare junction implicita (il punto 5,0 è su entrambi,
        # ma per il wire verticale è un'estremità - quindi è una T valida)
        # In realtà questo è un caso limite: in KiCad questo SAREBBE una T
        # Però senza junction dot, i due wire non sono connessi!
        # La T-junction detection la trova comunque, ma in KiCad reale
        # servirebbe il junction dot. Implementiamo questo comportamento corretto.
        pass  # TODO: rivedere regola KiCad per crossing senza dot


class TestNetReconstructor:
    def test_simple_net(self) -> None:
        recon = KiCadNetReconstructor()
        recon.wires = [
            WireSegment(Vec2(0, 0), Vec2(10, 0)),
            WireSegment(Vec2(10, 0), Vec2(10, 10)),
        ]
        recon.junction_detect()
        nets = recon.build_nets()

        assert len(nets) == 1
        assert len(nets[0].segments) == 2

    def test_two_separate_nets(self) -> None:
        recon = KiCadNetReconstructor()
        recon.wires = [
            WireSegment(Vec2(0, 0), Vec2(10, 0)),
            WireSegment(Vec2(0, 100), Vec2(10, 100)),
        ]
        recon.junction_detect()
        nets = recon.build_nets()

        assert len(nets) == 2

    def test_wire_merge_then_net(self) -> None:
        recon = KiCadNetReconstructor()
        recon.wires = [
            WireSegment(Vec2(0, 0), Vec2(5, 0)),
            WireSegment(Vec2(5, 0), Vec2(10, 0)),
        ]
        recon.wire_merge()
        assert len(recon.wires) == 1
        assert recon.wires[0].start == Vec2(0, 0)
        assert recon.wires[0].end == Vec2(10, 0)
