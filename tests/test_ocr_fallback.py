from __future__ import annotations

import numpy as np

from src.core.ocr_fallback import (
    ocr_rows_to_text_blocks,
    text_char_count,
)
from src.core.pdf_parser import PDFTextBlock


def _row(text: str, x0: float, y0: float, x1: float, y1: float, conf: float):
    """Build a rapidocr-style row: ([4 corner points in pixels], text, score)."""
    box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return [box, text, conf]


class TestRowsToTextBlocks:
    """rapidocr returns pixel-space boxes; we must keep them (the librechat OCR
    module discarded box[0]) and convert pixel -> PDF points by dividing by the
    render scale, so OCR text aligns with the vector geometry."""

    def test_pixel_box_converted_to_pdf_coords(self) -> None:
        # render scale = dpi/72; at dpi=144 scale=2.0 -> pixel/2 = pdf points.
        rows = [_row("R1", 200.0, 100.0, 240.0, 120.0, 0.99)]
        blocks = ocr_rows_to_text_blocks(rows, scale=2.0, min_confidence=0.0)
        assert len(blocks) == 1
        b = blocks[0]
        assert isinstance(b, PDFTextBlock)
        assert b.text == "R1"
        assert b.bbox == (100.0, 50.0, 120.0, 60.0)

    def test_low_confidence_rows_dropped(self) -> None:
        rows = [
            _row("R1", 0, 0, 10, 10, 0.95),
            _row("noise", 0, 0, 10, 10, 0.10),
        ]
        blocks = ocr_rows_to_text_blocks(rows, scale=1.0, min_confidence=0.5)
        assert [b.text for b in blocks] == ["R1"]

    def test_empty_or_none_result_yields_no_blocks(self) -> None:
        assert ocr_rows_to_text_blocks(None, scale=1.0, min_confidence=0.0) == []
        assert ocr_rows_to_text_blocks([], scale=1.0, min_confidence=0.0) == []

    def test_blank_text_skipped(self) -> None:
        rows = [_row("   ", 0, 0, 10, 10, 0.99), _row("C2", 0, 0, 10, 10, 0.99)]
        blocks = ocr_rows_to_text_blocks(rows, scale=1.0, min_confidence=0.0)
        assert [b.text for b in blocks] == ["C2"]

    def test_font_size_derived_from_box_height(self) -> None:
        # box height 20px at scale 2.0 -> 10 pt font size.
        rows = [_row("U1", 0, 0, 40, 20, 0.99)]
        blocks = ocr_rows_to_text_blocks(rows, scale=2.0, min_confidence=0.0)
        assert blocks[0].font_size == 10.0

    def test_recovered_block_is_classified_as_ref(self) -> None:
        rows = [_row("C5", 0, 0, 10, 10, 0.99)]
        blocks = ocr_rows_to_text_blocks(rows, scale=1.0, min_confidence=0.0)
        assert blocks[0].is_ref_designator


class TestTextCharCount:
    def test_counts_non_whitespace(self) -> None:
        blocks = [
            PDFTextBlock(text="R1", bbox=(0, 0, 1, 1)),
            PDFTextBlock(text="  ", bbox=(0, 0, 1, 1)),
            PDFTextBlock(text="GND", bbox=(0, 0, 1, 1)),
        ]
        assert text_char_count(blocks) == 5  # R1 + GND

    def test_empty_is_zero(self) -> None:
        assert text_char_count([]) == 0


class TestOcrTextBlocksInjectable:
    """ocr_text_blocks must accept an injected engine so the integration is
    testable without loading the ONNX model."""

    def test_uses_injected_engine(self) -> None:
        captured = {}

        def fake_engine(image: np.ndarray):
            captured["shape"] = image.shape
            return [_row("R7", 72.0, 0.0, 144.0, 36.0, 0.9)]

        # a fake fitz-like page that renders a dummy raster
        class FakePix:
            def __init__(self) -> None:
                self.width, self.height, self.n = 8, 8, 3
                self.samples = bytes(8 * 8 * 3)

        class FakePage:
            def get_pixmap(self, matrix=None):  # noqa: ANN001
                return FakePix()

        from src.core.ocr_fallback import ocr_text_blocks

        blocks = ocr_text_blocks(
            FakePage(), dpi=72, min_confidence=0.0, ocr=fake_engine
        )
        assert captured["shape"] == (8, 8, 3)
        assert [b.text for b in blocks] == ["R7"]
        assert blocks[0].bbox == (72.0, 0.0, 144.0, 36.0)  # scale=1.0 at dpi=72


class _FakePix:
    def __init__(self) -> None:
        self.width, self.height, self.n = 8, 8, 3
        self.samples = bytes(8 * 8 * 3)


class _FakePage:
    def get_pixmap(self, matrix=None):  # noqa: ANN001
        return _FakePix()


class TestVectorExtractorHook:
    """VectorExtractor must OCR only when the native text layer is empty, and
    leave text-rich pages untouched (no needless rasterization)."""

    def test_ocr_fires_when_text_layer_empty(self) -> None:
        from src.core.pdf_parser import VectorExtractor

        def engine(image):  # noqa: ANN001
            return [_row("J3", 0.0, 0.0, 72.0, 36.0, 0.9)]

        ext = VectorExtractor(dpi=72, ocr_engine=engine)
        blocks = ext._maybe_ocr(_FakePage(), [])
        assert [b.text for b in blocks] == ["J3"]

    def test_ocr_skipped_when_text_present(self) -> None:
        from src.core.pdf_parser import VectorExtractor

        def engine(image):  # noqa: ANN001
            raise AssertionError("OCR must not run when a text layer exists")

        ext = VectorExtractor(dpi=72, ocr_engine=engine)
        native = [
            PDFTextBlock(text="ATMEGA328P-AU", bbox=(0, 0, 1, 1)),
            PDFTextBlock(text="R1 C2 GND VCC", bbox=(0, 0, 1, 1)),
        ]
        blocks = ext._maybe_ocr(_FakePage(), native)
        assert blocks == native
