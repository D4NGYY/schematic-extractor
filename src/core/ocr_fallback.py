"""OCR fallback for text-less PDFs (CAD exports with outlined fonts).

Some schematic PDFs (e.g. ``arduino_nano``) ship with no text layer: every
glyph is drawn as vector outlines, so PyMuPDF's ``get_text`` returns nothing and
the whole text-association stage (ref/value/net-label → component) starves.

This module recovers text by rasterizing a page and running RapidOCR, the same
engine already used in the librechat ingestion project. The crucial difference
from that RAG-oriented version is that we **keep the bounding boxes**: RapidOCR
returns ``[box, text, confidence]`` rows and the box is what lets us place each
recovered ref/label back onto the schematic so it can be associated with a
component by coordinate. We therefore emit :class:`PDFTextBlock` objects in PDF
point space (pixels ÷ render scale), identical to the native extraction path —
the rest of the pipeline is unchanged.

The OCR engine is injectable so the conversion logic is unit-testable without
loading the ONNX runtime. ``rapidocr-onnxruntime`` is an optional dependency
(``pip install '.[ocr]'``); it is imported lazily only when actually invoked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .pdf_parser import PDFTextBlock

# An OCR callable maps a rendered page image (H×W×3 uint8 array) to RapidOCR's
# result: a list of ``[box, text, confidence]`` rows, or None when nothing was
# found. ``box`` is four [x, y] corner points in pixel space.
OcrCallable = Callable[[np.ndarray], Any]

# Drop OCR rows below this confidence so misreads don't pollute association.
# RapidOCR scores are typically >0.8 for clean glyphs; 0.5 trims the noise tail.
DEFAULT_MIN_CONFIDENCE = 0.5
# Native text-layer character count below which a page is treated as text-less
# and the OCR fallback kicks in. Mirrors the librechat module's threshold.
DEFAULT_MIN_CHARS = 16


def text_char_count(blocks: Sequence[PDFTextBlock]) -> int:
    """Count non-whitespace characters across ``blocks``.

    Used to decide whether a page has a usable text layer or needs OCR.
    """
    return sum(
        1 for block in blocks for char in block.text if not char.isspace()
    )


def ocr_rows_to_text_blocks(
    rows: Any,
    scale: float,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[PDFTextBlock]:
    """Convert RapidOCR rows into :class:`PDFTextBlock` in PDF point space.

    :param rows: RapidOCR output — a list of ``[box, text, confidence]`` rows
        (or ``None``). ``box`` is four ``[x, y]`` corner points in pixels.
    :param scale: the render scale used to rasterize the page (``dpi / 72``);
        pixel coordinates are divided by it to recover PDF points.
    :param min_confidence: rows scoring below this are dropped.
    """
    if not rows:
        return []

    blocks: list[PDFTextBlock] = []
    for row in rows:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2):
            continue
        box, text = row[0], row[1]
        text = str(text).strip()
        if not text:
            continue

        confidence = 1.0
        if len(row) >= 3:
            try:
                confidence = float(row[2])
            except (TypeError, ValueError):
                confidence = 1.0
        if confidence < min_confidence:
            continue

        try:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
        except (TypeError, ValueError, IndexError):
            continue
        if not xs or not ys:
            continue

        x0, y0 = min(xs) / scale, min(ys) / scale
        x1, y1 = max(xs) / scale, max(ys) / scale
        blocks.append(
            PDFTextBlock(
                text=text,
                bbox=(x0, y0, x1, y1),
                font_size=y1 - y0,  # box height in points ≈ glyph size
                font_name="",
                is_bold=False,
            )
        )
    return blocks


def render_page(page: Any, dpi: int) -> np.ndarray:
    """Rasterize a PyMuPDF page to an RGB ``H×W×3`` uint8 array."""
    import fitz

    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:  # RGBA → RGB
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def _build_default_ocr() -> OcrCallable:
    """Construct the default RapidOCR-backed callable (lazy ONNX import)."""
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()

    def run(image: np.ndarray) -> Any:
        result, _elapsed = engine(image)
        return result

    return run


def ocr_text_blocks(
    page: Any,
    *,
    dpi: int = 300,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ocr: OcrCallable | None = None,
) -> list[PDFTextBlock]:
    """Render ``page`` and recover text blocks via OCR, in PDF point space.

    :param page: a PyMuPDF page (must support ``get_pixmap``).
    :param dpi: render resolution; higher recovers smaller refs but costs more.
    :param min_confidence: drop OCR rows below this score.
    :param ocr: optional injected engine (defaults to lazy RapidOCR). The seam
        keeps this unit-testable without the ONNX runtime.
    """
    if ocr is None:
        ocr = _build_default_ocr()
    image = render_page(page, dpi)
    rows = ocr(image)
    return ocr_rows_to_text_blocks(rows, scale=dpi / 72.0, min_confidence=min_confidence)
