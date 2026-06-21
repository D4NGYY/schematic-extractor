"""Color-aware wire/symbol separation (KiCad-style stroke encoding).

KiCad PDF exports color-code strokes: wires teal, symbol bodies dark-red, labels
blue/purple (measured). The geometric `separate_wires` gate (axis-aligned AND
length>=p25*3) drops SHORT wires between close components -> dangling pins
(HANDOFF §16/§19). This module recovers them WITHOUT hardcoding RGB and WITHOUT
risk of pulling in symbol bodies:

  1. learn the wire color from segments geometry already confidently calls wires
     (axis-aligned + long) -> `wire_color`;
  2. only declare color "informative" if that color is DISTINCT from the dominant
     symbol-body color (monochrome legacy PDFs -> not informative -> caller keeps
     pure geometry, so Bryston etc. are unchanged);
  3. the caller then reclaims SHORT axis-aligned segments of the wire color.

Purely additive: it can only move short same-colored axis-aligned strokes from
symbols to wires, never the reverse.
"""
from __future__ import annotations

from collections import Counter

from src.core.pdf_parser import PDFSegment

Color = tuple[int, int, int]


def dominant_color(segments: list[PDFSegment]) -> Color | None:
    cnt = Counter(s.color for s in segments if s.color is not None)
    return cnt.most_common(1)[0][0] if cnt else None


def color_dist(a: Color | None, b: Color | None) -> float:
    if a is None or b is None:
        return float("inf")
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=False)) ** 0.5


def same_color(a: Color | None, b: Color | None, tol: float = 45.0) -> bool:
    return a is not None and b is not None and color_dist(a, b) <= tol


def wire_color_model(
    confident_wires: list[PDFSegment],
    symbol_segs: list[PDFSegment],
    min_colored_wires: int = 4,
    distinct_tol: float = 60.0,
) -> Color | None:
    """Wire color if the page's stroke colors are informative, else None.

    Returns None (-> caller uses pure geometry) when there are too few colored
    confident wires, or when the wire color is not distinct from the dominant
    symbol-body color (monochrome / non-KiCad rendering).
    """
    wc = dominant_color(confident_wires)
    if wc is None:
        return None
    n_colored = sum(1 for s in confident_wires if s.color is not None)
    if n_colored < min_colored_wires:
        return None
    sc = dominant_color(symbol_segs)
    if sc is not None and color_dist(wc, sc) < distinct_tol:
        return None
    return wc
