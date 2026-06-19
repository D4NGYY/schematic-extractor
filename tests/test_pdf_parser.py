from __future__ import annotations

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
