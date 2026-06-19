from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.pdf_parser import (
    PDFSegment,
    PDFShape,
    PDFTextBlock,
    VectorExtractor,
)
from src.core.text_associator import TextAssociator


class TestPDFSegment:
    def test_length(self) -> None:
        seg = PDFSegment(start=(0, 0), end=(3, 4))
        assert seg.length == pytest.approx(5.0)

    def test_midpoint(self) -> None:
        seg = PDFSegment(start=(0, 0), end=(10, 10))
        assert seg.midpoint() == (5.0, 5.0)


class TestPDFTextBlock:
    def test_is_ref_designator(self) -> None:
        assert PDFTextBlock("R1", (0, 0, 1, 1)).is_ref_designator
        assert PDFTextBlock("C105", (0, 0, 1, 1)).is_ref_designator
        assert not PDFTextBlock("1k", (0, 0, 1, 1)).is_ref_designator
        assert not PDFTextBlock("GND", (0, 0, 1, 1)).is_ref_designator

    def test_is_value(self) -> None:
        assert PDFTextBlock("1k", (0, 0, 1, 1)).is_value
        assert PDFTextBlock("10uF", (0, 0, 1, 1)).is_value
        assert not PDFTextBlock("R1", (0, 0, 1, 1)).is_value

    def test_is_net_label(self) -> None:
        assert PDFTextBlock("VCC", (0, 0, 1, 1)).is_net_label
        assert PDFTextBlock("SIG_OUT", (0, 0, 1, 1)).is_net_label
        assert not PDFTextBlock("R1", (0, 0, 1, 1)).is_net_label


class TestPDFShape:
    def test_bbox_and_area(self) -> None:
        shape = PDFShape(
            vertices=[(0, 0), (10, 0), (10, 10), (0, 10)],
            item_type="rect",
        )
        assert shape.bbox == (0, 0, 10, 10)
        assert shape.area == pytest.approx(100.0)

    def test_circle_area(self) -> None:
        shape = PDFShape(
            vertices=[(0, 0), (10, 0)],  # dummy per bbox
            item_type="circle",
        )
        assert shape.area > 0

    def test_is_filled_circle_junction(self) -> None:
        # Cerchio piccolo pieno = junction dot candidate
        shape = PDFShape(
            vertices=[(0, 0), (2, 2)],
            item_type="circle",
            fill_color=(0, 0, 0),
        )
        assert shape.is_filled_circle

        # Troppo grande = no
        shape2 = PDFShape(
            vertices=[(0, 0), (20, 20)],
            item_type="circle",
            fill_color=(0, 0, 0),
        )
        assert not shape2.is_filled_circle


class TestVectorExtractorMerge:
    def test_merge_collinear(self) -> None:
        a = PDFSegment(start=(0, 0), end=(10, 0), item_type="line")
        b = PDFSegment(start=(5, 0), end=(15, 0), item_type="line")
        merged = VectorExtractor._try_merge(a, b, tolerance=0.5)
        assert merged is not None
        assert merged.start == (0, 0)
        assert merged.end == (15, 0)

    def test_no_merge_different_line(self) -> None:
        a = PDFSegment(start=(0, 0), end=(10, 0), item_type="line")
        b = PDFSegment(start=(0, 100), end=(10, 100), item_type="line")
        merged = VectorExtractor._try_merge(a, b, tolerance=0.5)
        assert merged is None

    def test_merge_non_overlapping(self) -> None:
        a = PDFSegment(start=(0, 0), end=(10, 0), item_type="line")
        b = PDFSegment(start=(100, 0), end=(200, 0), item_type="line")
        merged = VectorExtractor._try_merge(a, b, tolerance=0.5)
        assert merged is None


class TestTextAssociator:
    def test_directional_priority(self) -> None:
        from src.core.pdf_parser import ExtractedPage

        page = ExtractedPage(page_num=0)
        # Simbolo al centro (0,0)
        page.shapes.append(
            PDFShape(
                vertices=[(-5, -5), (5, -5), (5, 5), (-5, 5)],
                item_type="rect",
            )
        )
        # Ref a sinistra del simbolo
        page.text_blocks.append(PDFTextBlock("R1", (-20, -2, -10, 2)))
        # Value a destra del simbolo
        page.text_blocks.append(PDFTextBlock("1k", (10, -2, 20, 2)))

        assoc = TextAssociator()
        refs, values, nets = assoc.associate(page)

        assert len(refs) == 1
        assert refs[0].text == "R1"
        assert len(values) == 1
        assert values[0].text == "1k"
        assert len(nets) == 0

    def test_net_label_to_wire(self) -> None:
        from src.core.pdf_parser import ExtractedPage

        page = ExtractedPage(page_num=0)
        # Wire orizzontale
        page.segments.append(PDFSegment(start=(0, 0), end=(50, 0), item_type="line"))
        # Net label sopra il wire
        page.text_blocks.append(PDFTextBlock("VCC", (20, 5, 30, 10)))

        assoc = TextAssociator()
        refs, values, nets = assoc.associate(page)

        assert len(nets) == 1
        assert nets[0].text == "VCC"
        assert nets[0].distance < 10.0

    def test_no_match_if_too_far(self) -> None:
        from src.core.pdf_parser import ExtractedPage

        page = ExtractedPage(page_num=0)
        page.shapes.append(
            PDFShape(
                vertices=[(0, 0), (10, 0), (10, 10), (0, 10)],
                item_type="rect",
            )
        )
        # Ref molto lontano (> max_distance=50)
        page.text_blocks.append(PDFTextBlock("R99", (1000, 1000, 1010, 1010)))

        assoc = TextAssociator()
        refs, _, _ = assoc.associate(page)
        assert len(refs) == 0


class TestRefDesignatorRegexD1:
    """D1: nuova regex ^(?:[A-Z][0-9]{1,4}|[A-Z]{2}[0-9]{1,2})[A-Z]?$"""

    def test_single_letter_prefix(self) -> None:
        for ref in ("R1", "C105", "Q3", "D1", "L2", "J3"):
            assert PDFTextBlock(ref, (0, 0, 1, 1)).is_ref_designator, ref

    def test_two_letter_prefix(self) -> None:
        for ref in ("QB1", "RB14", "U1A", "U12B", "RN1", "DB3", "CB1", "SW1", "TP1"):
            assert PDFTextBlock(ref, (0, 0, 1, 1)).is_ref_designator, ref

    def test_not_ref_part_numbers(self) -> None:
        # 2-letter prefix + 3+ digits = part number, non ref designator
        for val in ("BC547", "TL071", "MJE15030", "LM358"):
            assert not PDFTextBlock(val, (0, 0, 1, 1)).is_ref_designator, val

    def test_not_ref_net_labels(self) -> None:
        for label in ("GND", "VCC", "SIG_OUT", "VBAT"):
            assert not PDFTextBlock(label, (0, 0, 1, 1)).is_ref_designator, label


class TestValueRegexB4:
    """B4: regex E96, standard, part number, tensioni."""

    def test_e96_r_notation(self) -> None:
        for val in ("49R9", "100R", "0R1", "22R1", "0R", "150R"):
            assert PDFTextBlock(val, (0, 0, 1, 1)).is_value, val

    def test_e96_k_m_notation(self) -> None:
        for val in ("4k7", "10K0", "4K99", "3K48", "1K0", "475K", "1MEG"):
            assert PDFTextBlock(val, (0, 0, 1, 1)).is_value, val

    def test_standard_values(self) -> None:
        for val in ("1k", "10uF", "4.7nH", "100N", "220p", "2uH", "68p"):
            assert PDFTextBlock(val, (0, 0, 1, 1)).is_value, val

    def test_transistor_part_numbers(self) -> None:
        for val in ("2N2222", "BC547", "MJE15030", "MJL21193", "TL064"):
            assert PDFTextBlock(val, (0, 0, 1, 1)).is_value, val

    def test_voltages(self) -> None:
        for val in ("+5V", "-24V", "12V", "3V3", "+48V", "-48V"):
            assert PDFTextBlock(val, (0, 0, 1, 1)).is_value, val

    def test_not_value_net_labels(self) -> None:
        for label in ("GND", "VCC", "SIG_OUT", "VBAT", "VDD"):
            assert not PDFTextBlock(label, (0, 0, 1, 1)).is_value, label

    def test_not_value_ref_designators(self) -> None:
        # I ref designator hanno priorità su is_value
        for ref in ("R1", "QB1", "RB14", "U1A", "CB1"):
            assert not PDFTextBlock(ref, (0, 0, 1, 1)).is_value, ref


class TestSpanMergingB3:
    """B3: _extract_text_blocks merge span frammentati."""

    def _make_page(self, spans: list[dict[str, Any]]) -> Any:
        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [{"type": 0, "lines": [{"spans": spans}]}]
        }
        return page

    def test_merge_adjacent_spans(self) -> None:
        """Span "R"+"1" con gap < soglia devono fondersi in "R1"."""
        page = self._make_page([
            {"text": "R", "bbox": (0.0, 0.0, 5.0, 10.0), "size": 8.0, "font": "Arial", "flags": 0},
            {"text": "1", "bbox": (5.5, 0.0, 10.0, 10.0), "size": 8.0, "font": "Arial", "flags": 0},
        ])
        blocks = VectorExtractor()._extract_text_blocks(page)
        assert len(blocks) == 1
        assert blocks[0].text == "R1"

    def test_merge_value_spans(self) -> None:
        """Span "10"+"k" devono fondersi in "10k"."""
        page = self._make_page([
            {"text": "10", "bbox": (0.0, 0.0, 8.0, 10.0), "size": 8.0, "font": "Arial", "flags": 0},
            {"text": "k", "bbox": (8.3, 0.0, 12.0, 10.0), "size": 8.0, "font": "Arial", "flags": 0},
        ])
        blocks = VectorExtractor()._extract_text_blocks(page)
        assert len(blocks) == 1
        assert blocks[0].text == "10k"

    def test_separate_far_spans(self) -> None:
        """Span molto distanti (gap > soglia) restano token separati."""
        page = self._make_page([
            {"text": "R1", "bbox": (0.0, 0.0, 20.0, 10.0), "size": 8.0, "font": "Arial", "flags": 0},
            {"text": "1k", "bbox": (100.0, 0.0, 120.0, 10.0), "size": 8.0, "font": "Arial", "flags": 0},
        ])
        blocks = VectorExtractor()._extract_text_blocks(page)
        assert len(blocks) == 2
        assert {b.text for b in blocks} == {"R1", "1k"}

    def test_empty_spans_skipped(self) -> None:
        """Span vuoti non generano PDFTextBlock."""
        page = self._make_page([
            {"text": "", "bbox": (0.0, 0.0, 5.0, 10.0), "size": 8.0, "font": "Arial", "flags": 0},
            {"text": "VCC", "bbox": (10.0, 0.0, 30.0, 10.0), "size": 8.0, "font": "Arial", "flags": 0},
        ])
        blocks = VectorExtractor()._extract_text_blocks(page)
        assert len(blocks) == 1
        assert blocks[0].text == "VCC"

    def test_image_blocks_skipped(self) -> None:
        """Blocchi tipo=1 (immagini) non generano PDFTextBlock."""
        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [{"type": 1, "lines": []}]  # immagine
        }
        blocks = VectorExtractor()._extract_text_blocks(page)
        assert blocks == []


class TestJunctionDetectionB2:
    """B2: _try_extract_circle rileva cerchi pieni dai drawing dict di PyMuPDF."""

    def _make_rect(self, x0: float, y0: float, x1: float, y1: float) -> Any:
        rect = MagicMock()
        rect.x0 = x0
        rect.y0 = y0
        rect.x1 = x1
        rect.y1 = y1
        rect.width = x1 - x0
        rect.height = y1 - y0
        return rect

    def _make_circle_drawing(
        self, x0: float, y0: float, x1: float, y1: float,
        fill: Any = (0.0, 0.0, 0.0),
    ) -> dict[str, Any]:
        return {
            "fill": fill,
            "rect": self._make_rect(x0, y0, x1, y1),
            "items": [("c", None, None, None, None)] * 4,
        }

    def test_small_filled_circle_detected(self) -> None:
        """Cerchio 2×2 pt riempito → PDFShape circle."""
        drawing = self._make_circle_drawing(0.0, 0.0, 2.0, 2.0)
        shape = VectorExtractor()._try_extract_circle(drawing)
        assert shape is not None
        assert shape.item_type == "circle"
        assert shape.fill_color is not None

    def test_is_junction_candidate(self) -> None:
        """Cerchio 2×2 pieno → is_filled_circle True."""
        drawing = self._make_circle_drawing(0.0, 0.0, 2.0, 2.0)
        shape = VectorExtractor()._try_extract_circle(drawing)
        assert shape is not None
        assert shape.is_filled_circle

    def test_large_circle_not_junction(self) -> None:
        """Cerchio grande (10×10) → shape creato ma is_filled_circle False."""
        drawing = self._make_circle_drawing(0.0, 0.0, 10.0, 10.0)
        shape = VectorExtractor()._try_extract_circle(drawing)
        assert shape is not None
        assert not shape.is_filled_circle

    def test_non_square_bbox_ignored(self) -> None:
        """Bbox non quadrata (ellisse distorta) → None."""
        drawing = self._make_circle_drawing(0.0, 0.0, 10.0, 2.0)
        shape = VectorExtractor()._try_extract_circle(drawing)
        assert shape is None

    def test_unfilled_drawing_ignored(self) -> None:
        """Drawing senza fill → None."""
        drawing = self._make_circle_drawing(0.0, 0.0, 2.0, 2.0, fill=None)
        shape = VectorExtractor()._try_extract_circle(drawing)
        assert shape is None

    def test_non_bezier_items_ignored(self) -> None:
        """Drawing con linee (non Bezier) → None."""
        drawing = {
            "fill": (0.0, 0.0, 0.0),
            "rect": self._make_rect(0.0, 0.0, 2.0, 2.0),
            "items": [("l", None, None), ("l", None, None)],
        }
        shape = VectorExtractor()._try_extract_circle(drawing)
        assert shape is None

    def test_grayscale_fill_accepted(self) -> None:
        """Fill scalare (grayscale) → PDFShape valido."""
        drawing = self._make_circle_drawing(0.0, 0.0, 2.0, 2.0, fill=0.0)
        shape = VectorExtractor()._try_extract_circle(drawing)
        assert shape is not None
        assert shape.fill_color == (0, 0, 0)
