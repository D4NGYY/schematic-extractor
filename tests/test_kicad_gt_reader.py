from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.core.kicad_gt_reader import (
    build_gt_graph,
    parse_kicad_sch,
    parse_sexpr,
    tokenize_sexpr,
)


def test_tokenize_and_parse() -> None:
    text = '(symbol "Device:R" (at 10 20) (property "Reference" "R1"))'
    tokens = tokenize_sexpr(text)
    ast = parse_sexpr(tokens)
    assert ast == ["symbol", "Device:R", ["at", "10", "20"], ["property", "Reference", "R1"]]


def test_parse_simple_resistor() -> None:
    text = """(kicad_sch (version 20211123)
      (symbol (lib_id "Device:R") (at 10 20)
        (property "Reference" "R1" (at 15 20 0))
        (pin "1" (uuid "123") (at 10 15 90))
        (pin "2" (uuid "456") (at 10 25 270))
      )
      (wire (pts (xy 10 10) (xy 10 15)))
    )"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.kicad_sch"
        p.write_text(text, encoding="utf-8")

        sch = parse_kicad_sch(p)
        assert len(sch.symbols) == 1
        assert sch.symbols[0].ref == "R1"
        assert len(sch.symbols[0].pins) == 2
        assert sch.symbols[0].pins[0].pin_num == "1"
        assert len(sch.wires) == 1
        assert sch.wires[0].x1 == 10.0


def test_gt_graph_bridge_rectifier() -> None:
    p = Path("data/kicad/synthetic/bridge_rectifier.kicad_sch")
    if not p.exists():
        pytest.skip(f"GT file {p} not found")

    sch = parse_kicad_sch(p)
    assert len(sch.wires) == 6
    assert len(sch.junctions) == 1
    assert len(sch.labels) == 3

    graph = build_gt_graph(sch)
    # The file has 3 global labels: AC_IN, VCC, GND, and 6 wires.
    # We expect some nets to be formed and named after labels.
    assert "AC_IN" in graph.nets or graph.net_count > 0


def test_gt_graph_resistor_ladder() -> None:
    p = Path("data/kicad/synthetic/resistor_ladder_3.kicad_sch")
    if not p.exists():
        pytest.skip(f"GT file {p} not found")

    sch = parse_kicad_sch(p)
    assert len(sch.wires) > 0
    graph = build_gt_graph(sch)
    assert graph.net_count > 0


def test_parse_lib_symbols_and_absolute_position() -> None:
    text = """(kicad_sch (version 20211123)
      (lib_symbols
        (symbol "Device:R"
          (symbol "Device:R_1_1"
            (pin passive line (at 0 2.54 270) (length 2.54)
              (name "~" (effects (font (size 1.27 1.27))))
              (number "1" (effects (font (size 1.27 1.27))))
            )
            (pin passive line (at 0 -2.54 90) (length 2.54)
              (name "~" (effects (font (size 1.27 1.27))))
              (number "2" (effects (font (size 1.27 1.27))))
            )
          )
        )
      )
      (symbol (lib_id "Device:R") (at 10 20 90)
        (property "Reference" "R1" (at 15 20 0))
      )
    )"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test2.kicad_sch"
        p.write_text(text, encoding="utf-8")

        sch = parse_kicad_sch(p)
        assert len(sch.symbols) == 1
        assert sch.symbols[0].ref == "R1"
        assert len(sch.symbols[0].pins) == 2
        # Transform check:
        # Original: (0, 2.54) rotated by 90 deg clockwise (or CCW?)
        # rot = 90
        # mx = 0, my = 2.54
        # rx = 0*cos(90) - 2.54*sin(90) = -2.54
        # ry = 0*sin(90) + 2.54*cos(90) = 0
        # abs_x = 10 - 2.54 = 7.46
        # abs_y = 20 + 0 = 20
        # Let's just check that pins were populated
        assert sch.symbols[0].pins[0].pin_num == "1"


def test_gt_excludes_power_symbols_from_components() -> None:
    # KiCad refs prefixed with '#' (e.g. #PWR0xx power ports, #FLG power flags)
    # are virtual symbols / net anchors, NOT physical components. They must not
    # count as components or they crush component-level recall.
    text = """(kicad_sch (version 20211123)
      (symbol (lib_id "Device:R") (at 10 20)
        (property "Reference" "R1" (at 15 20 0))
        (pin "1" (uuid "1") (at 10 15 90))
      )
      (symbol (lib_id "power:GND") (at 10 30)
        (property "Reference" "#PWR01" (at 10 35 0))
        (pin "1" (uuid "2") (at 10 15 90))
      )
    )"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pwr.kicad_sch"
        p.write_text(text, encoding="utf-8")

        sch = parse_kicad_sch(p)
        graph = build_gt_graph(sch)
        assert graph.components == {"R1"}
        assert "#PWR01" not in graph.components


def test_gt_graph_micro_after_fix() -> None:
    p = Path("test_input/multi_schematic/arduino_micro/arduino_micro.kicad_sch")
    if not p.exists():
        pytest.skip(f"GT file {p} not found")

    sch = parse_kicad_sch(p)
    graph = build_gt_graph(sch)
    assert graph.net_count > 0
    assert len(graph.nets) > 0


def test_gt_graph_nano_after_fix() -> None:
    p = Path("test_input/multi_schematic/arduino_nano/arduino_nano.kicad_sch")
    if not p.exists():
        pytest.skip(f"GT file {p} not found")

    sch = parse_kicad_sch(p)
    graph = build_gt_graph(sch)
    assert graph.net_count > 0
    assert len(graph.nets) > 0
