from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from src.core.pdf_parser import ExtractedPage, PDFSegment, PDFTextBlock

logger = structlog.get_logger("text_associator")


@dataclass
class SymbolAssociation:
    """Associazione tra un testo (Ref/Value/NetLabel) e un simbolo geometrico."""
    text: str
    text_type: str  # "ref", "value", "net_label"
    text_pos: tuple[float, float]
    symbol_bbox: tuple[float, float, float, float]
    symbol_center: tuple[float, float]
    distance: float
    confidence: float


class TextAssociator:
    """Associa blocchi di testo (Ref, Value, NetLabel) ai simboli geometrici più vicini.

    Euristiche:
    - Ref Designator: cerca il simbolo più vicino con priorità direzionale
      (Ref a sinistra o sopra il simbolo, come convenzione EDA)
    - Value: cerca il simbolo più vicino con priorità direzionale
      (Value a destra o sotto il simbolo)
    - Net Label: cerca il wire/segmento più vicino (non un simbolo)
    """

    def __init__(
        self,
        ref_direction_priority: str = "left_above",
        value_direction_priority: str = "right_below",
        max_distance: float = 50.0,  # punti PDF ≈ 17mm
    ) -> None:
        self.ref_direction = ref_direction_priority
        self.value_direction = value_direction_priority
        self.max_distance = max_distance

    def associate(
        self, page: ExtractedPage
    ) -> tuple[list[SymbolAssociation], list[SymbolAssociation], list[SymbolAssociation]]:
        """Associa Ref, Value e NetLabel ai simboli/segmenti.

        Ritorna: (ref_associations, value_associations, net_label_associations)
        """
        # Crea candidate simboli dai segmenti (cluster di segmenti = simbolo)
        # Per MVP: usiamo le shapes estratte e i segmenti come proxy
        symbol_candidates = self._build_symbol_candidates(page)

        refs: list[SymbolAssociation] = []
        values: list[SymbolAssociation] = []
        nets: list[SymbolAssociation] = []

        for block in page.ref_blocks():
            assoc = self._find_best_symbol(
                block, symbol_candidates, direction="left_above"
            )
            if assoc is not None:
                refs.append(assoc)

        for block in page.value_blocks():
            assoc = self._find_best_symbol(
                block, symbol_candidates, direction="right_below"
            )
            if assoc is not None:
                values.append(assoc)

        for block in page.net_label_blocks():
            assoc = self._find_nearest_wire(block, page)
            if assoc is not None:
                nets.append(assoc)

        logger.info(
            "Text association complete",
            refs=len(refs),
            values=len(values),
            nets=len(nets),
        )
        return refs, values, nets

    def _build_symbol_candidates(
        self, page: ExtractedPage
    ) -> list[dict[str, Any]]:
        """Costruisce candidate simboli da shapes e segmenti.

        Semplice: ogni shape chiusa è un simbolo candidate.
        In futuro: clustering spaziale dei segmenti.
        """
        candidates = []
        for shape in page.shapes:
            bbox = shape.bbox
            candidates.append({
                "bbox": bbox,
                "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                "type": shape.item_type,
                "area": shape.area,
            })

        # Se non ci sono shape, usa i segmenti come proxy (raggruppa per prossimità)
        if not candidates and page.segments:
            # Fallback: ogni segmento è un candidate (molto grezzo)
            for seg in page.segments:
                mid = seg.midpoint()
                candidates.append({
                    "bbox": (seg.start[0], seg.start[1], seg.end[0], seg.end[1]),
                    "center": mid,
                    "type": "segment",
                    "area": 0.0,
                })

        return candidates

    def _find_best_symbol(
        self,
        block: PDFTextBlock,
        candidates: list[dict[str, Any]],
        direction: str,
    ) -> SymbolAssociation | None:
        """Trova il simbolo candidate più vicino con priorità direzionale."""
        best = None
        best_score = float("inf")
        tx, ty = block.center

        for cand in candidates:
            cx, cy = cand["center"]
            dx = tx - cx
            dy = ty - cy
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > self.max_distance:
                continue

            # Score = distanza * fattore direzionale
            # Preferisce: left_above → dx>0 (testo a destra), dy<0 (testo sopra)
            # Preferisce: right_below → dx<0 (testo a sinistra), dy>0 (testo sotto)
            direction_penalty = 1.0
            if direction == "left_above":
                # Ref dovrebbe essere a sinistra/sopra del simbolo
                # Quindi tx < cx (dx < 0) e ty < cy (dy < 0) è preferito
                if dx > 0:
                    direction_penalty *= 1.5
                if dy > 0:
                    direction_penalty *= 1.5
            elif direction == "right_below":
                # Value dovrebbe essere a destra/sotto del simbolo
                if dx < 0:
                    direction_penalty *= 1.5
                if dy < 0:
                    direction_penalty *= 1.5

            score = dist * direction_penalty
            if score < best_score:
                best_score = score
                best = cand

        if best is None:
            return None

        cx, cy = best["center"]
        dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
        confidence = max(0.0, 1.0 - dist / self.max_distance)

        return SymbolAssociation(
            text=block.text,
            text_type="ref" if direction == "left_above" else "value",
            text_pos=(tx, ty),
            symbol_bbox=best["bbox"],
            symbol_center=(cx, cy),
            distance=dist,
            confidence=confidence,
        )

    def _find_nearest_wire(
        self, block: PDFTextBlock, page: ExtractedPage
    ) -> SymbolAssociation | None:
        """Trova il segmento di wire più vicino a un net label."""
        best = None
        best_dist = float("inf")
        tx, ty = block.center

        for seg in page.all_wire_segments():
            # Distanza punto-segmento
            dist = self._point_to_segment_distance(tx, ty, seg)
            if dist < best_dist and dist < self.max_distance:
                best_dist = dist
                best = seg

        if best is None:
            return None

        # Crea una "bbox" fittizia per il segmento
        bbox = (
            min(best.start[0], best.end[0]),
            min(best.start[1], best.end[1]),
            max(best.start[0], best.end[0]),
            max(best.start[1], best.end[1]),
        )
        mid = best.midpoint()
        confidence = max(0.0, 1.0 - best_dist / self.max_distance)

        return SymbolAssociation(
            text=block.text,
            text_type="net_label",
            text_pos=(tx, ty),
            symbol_bbox=bbox,
            symbol_center=mid,
            distance=best_dist,
            confidence=confidence,
        )

    @staticmethod
    def _point_to_segment_distance(
        px: float, py: float, seg: PDFSegment
    ) -> float:
        """Distanza minima da un punto a un segmento."""
        x1: float = seg.start[0]
        y1: float = seg.start[1]
        x2: float = seg.end[0]
        y2: float = seg.end[1]
        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-12:
            return float(((px - x1) ** 2 + (py - y1) ** 2) ** 0.5)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return float(((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5)
