from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.core.kicad_gt_reader import (
    KicadJunction,
    KicadLabel,
    KicadSymbol,
    KicadWire,
    build_gt_graph,
    parse_kicad_sch,
    tokenize_sexpr,
    parse_sexpr,
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
