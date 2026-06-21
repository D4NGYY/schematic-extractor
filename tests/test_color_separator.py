from __future__ import annotations

from src.core.pdf_parser import PDFSegment
from src.ml.color_separator import (
    color_dist,
    dominant_color,
    same_color,
    wire_color_model,
)


def _seg(color, x1=0.0, y1=0.0, x2=10.0, y2=0.0):
    return PDFSegment(start=(x1, y1), end=(x2, y2), item_type="line", color=color)


def test_dominant_color_ignores_none() -> None:
    segs = [_seg((0, 100, 100)), _seg((0, 100, 100)), _seg(None), _seg((132, 0, 0))]
    assert dominant_color(segs) == (0, 100, 100)


def test_color_dist_and_same() -> None:
    assert color_dist((0, 0, 0), (0, 0, 0)) == 0.0
    assert same_color((0, 100, 100), (2, 98, 101))
    assert not same_color((0, 100, 100), (132, 0, 0))
    assert not same_color(None, (0, 0, 0))


def test_wire_color_model_informative() -> None:
    # teal wires distinct from red bodies -> informative
    wires = [_seg((0, 100, 100)) for _ in range(6)]
    bodies = [_seg((132, 0, 0)) for _ in range(6)]
    assert wire_color_model(wires, bodies) == (0, 100, 100)


def test_wire_color_model_monochrome_falls_back() -> None:
    # wire color == body color (monochrome) -> None (use geometry)
    wires = [_seg((20, 20, 20)) for _ in range(6)]
    bodies = [_seg((20, 20, 20)) for _ in range(6)]
    assert wire_color_model(wires, bodies) is None


def test_wire_color_model_too_few_colored() -> None:
    assert wire_color_model([_seg((0, 100, 100))], [_seg((132, 0, 0))]) is None
