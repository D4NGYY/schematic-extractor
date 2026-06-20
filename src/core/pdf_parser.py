from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger("pdf_parser")

# B4: regex per valori EDA reali (E96, standard, part number, tensioni)
_VALUE_RE = re.compile(
    r"^(?:"
    r"[0-9]+[RrKkMmGgNnPpFf][0-9]*"                           # E96/Enotation: 49R9, 100R, 4k7, 22p, 0R1
    r"|[0-9]+(?:[.][0-9]+)?\s*[kKMGmnupfTμΩµ]{1,2}[FfHhΩzZ]?"  # standard: 1k, 10uF, 4.7nH
    r"|[+-]?[0-9]+(?:[.][0-9]+)?[Vv]"                         # tensioni: +5V, -24V, 12V
    r"|[0-9]+[Vv][0-9]+"                                        # frazionali: 3V3
    r"|[0-9][A-Z][A-Z0-9]{2,}"                                 # JEDEC: 2N2222, 1N4148, 2SC1815
    r"|[A-Z]{2,4}[0-9]{2,}"                                    # part number: BC547, TL071, MJE3055
    r")$"
)


class PDFSourceFormat(Enum):
    """Formato di origine dello schema PDF."""
    KICAD = "kicad"
    ALTIUM = "altium"
    EAGLE = "eagle"
    GENERIC = "generic"
    UNKNOWN = "unknown"


class PDFFormatClassifier:
    """Classifica il formato di origine di uno schema PDF tramite euristiche.

    Euristiche usate (in ordine di affidabilità):
    1. Metadati XMP / Producer (es. "KiCad", "Altium")
    2. Font embedding (font EDA specifici)
    3. Pattern di testo (ref des, net labels, pagine specifiche)
    4. Struttura pagina (dimensioni, griglia, margine)
    """

    _PRODUCER_PATTERNS: dict[str, PDFSourceFormat] = {
        "kicad": PDFSourceFormat.KICAD,
        "altium": PDFSourceFormat.ALTIUM,
        "eagle": PDFSourceFormat.EAGLE,
    }

    def classify(self, pdf_path: Path | str) -> PDFSourceFormat:
        """Classifica il PDF e ritorna il formato di origine."""
        pdf_path = Path(pdf_path)
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed, falling back to generic")
            return PDFSourceFormat.GENERIC

        doc = fitz.open(str(pdf_path))
        fmt = self._classify_doc(doc)
        doc.close()
        logger.info("PDF classified", file=str(pdf_path), format=fmt.value)
        return fmt

    def _classify_doc(self, doc: Any) -> PDFSourceFormat:
        # 1. Metadati Producer / Creator
        metadata = doc.metadata or {}
        producer = (metadata.get("producer", "") + metadata.get("creator", "")).lower()
        for pattern, fmt in self._PRODUCER_PATTERNS.items():
            if pattern in producer:
                return fmt

        # 2. Pattern di testo nella prima pagina
        if len(doc) > 0:
            text = doc[0].get_text()
            if "Sheet" in text and "Date:" in text and re.search(r"[A-Z]\d{1,4}", text):
                return PDFSourceFormat.ALTIUM
            if "KICAD" in text.upper() or ".kicad_sch" in text:
                return PDFSourceFormat.KICAD

        # 3. Dimensione pagina (KiCad default A4 = 595x842 pt)
        if len(doc) > 0:
            rect = doc[0].rect
            if abs(rect.width - 595) < 5 and abs(rect.height - 842) < 5:
                # Potrebbe essere KiCad o generico A4
                pass

        return PDFSourceFormat.UNKNOWN


@dataclass
class PDFSegment:
    """Segmento vettoriale estratto da PDF (linea o arco)."""
    start: tuple[float, float]
    end: tuple[float, float]
    stroke_width: float = 0.0
    color: tuple[int, int, int] | None = None
    item_type: str = "line"  # "line", "arc", "curve", "rect"

    @property
    def length(self) -> float:
        return math.hypot(
            self.end[0] - self.start[0], self.end[1] - self.start[1]
        )

    def midpoint(self) -> tuple[float, float]:
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2,
        )


@dataclass
class PDFTextBlock:
    """Blocco di testo estratto da PDF con metadati geometrici."""
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def is_ref_designator(self) -> bool:
        """D1: 1-lettera con fino a 4 cifre (R1, C105) o 2-lettere con 1-2 cifre (QB1, RB14, U1A)."""
        return bool(re.match(r"^(?:[A-Z][0-9]{1,4}|[A-Z]{2}[0-9]{1,2})[A-Z]?$", self.text.strip()))

    @property
    def is_value(self) -> bool:
        """B4: valore componente EDA (49R9, 4k7, 10uF, 2N2222, BC547, +5V, 3V3).
        I ref designator hanno priorità: se is_ref_designator=True, is_value=False.
        """
        if self.is_ref_designator:
            return False
        return bool(_VALUE_RE.match(self.text.strip()))

    @property
    def is_net_label(self) -> bool:
        """Label di net (non ref, non value, non vuota)."""
        text = self.text.strip()
        if not text or len(text) > 50:
            return False
        # Esclude ref e value, include nomi tipo GND, VCC, SIG, ecc.
        if self.is_ref_designator or self.is_value:
            return False
        return bool(re.match(r"^[A-Za-z0-9_+/\-]+$", text))


@dataclass
class PDFShape:
    """Forma chiusa estratta da PDF (circolo, rettangolo, poligono)."""
    vertices: list[tuple[float, float]] = field(default_factory=list)
    item_type: str = "polygon"  # "polygon", "circle", "rect"
    fill_color: tuple[int, int, int] | None = None
    stroke_color: tuple[int, int, int] | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def area(self) -> float:
        if self.item_type == "circle":
            # Approssimazione: diametro dalla bbox
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            r = (w + h) / 4
            return math.pi * r * r
        if self.item_type == "rect":
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            return w * h
        # Poligono: shoelace formula
        return self._polygon_area()

    def _polygon_area(self) -> float:
        area = 0.0
        n = len(self.vertices)
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    @property
    def is_filled_circle(self) -> bool:
        """Verifica se è un cerchio pieno (junction dot candidate)."""
        if self.item_type != "circle":
            return False
        if self.fill_color is None:
            return False
        # Diametro tipico junction dot: 0.3-1.0 mm ≈ 1-3 pt @ 72 DPI
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return 0.5 < w < 5.0 and 0.5 < h < 5.0 and abs(w - h) < 1.0


@dataclass
class ExtractedPage:
    """Risultato dell'estrazione di una singola pagina PDF."""
    page_num: int
    segments: list[PDFSegment] = field(default_factory=list)
    text_blocks: list[PDFTextBlock] = field(default_factory=list)
    shapes: list[PDFShape] = field(default_factory=list)
    raw_rect: tuple[float, float, float, float] = (0, 0, 0, 0)

    def all_wire_segments(self) -> list[PDFSegment]:
        """Filtra solo i segmenti che sembrano wire (linee rette, non archi)."""
        return [s for s in self.segments if s.item_type == "line"]

    def ref_blocks(self) -> list[PDFTextBlock]:
        return [t for t in self.text_blocks if t.is_ref_designator]

    def value_blocks(self) -> list[PDFTextBlock]:
        return [t for t in self.text_blocks if t.is_value]

    def net_label_blocks(self) -> list[PDFTextBlock]:
        return [t for t in self.text_blocks if t.is_net_label]

    def junction_candidates(self) -> list[PDFShape]:
        return [s for s in self.shapes if s.is_filled_circle]


class VectorExtractor:
    """Estrae elementi vettoriali (segmenti, testo, forme) da PDF tramite PyMuPDF.

    Pipeline per pagina:
    1. `get_drawings()` → segmenti, archi, forme
    2. `get_text("blocks")` → blocchi di testo con bbox
    3. Unisce span frammentati basandosi *esclusivamente su geometria* (no font name)
    """

    def __init__(
        self,
        dpi: int = 300,
        ocr_fallback: bool = True,
        ocr_engine: Any | None = None,
        ocr_min_chars: int = 16,
        ocr_min_confidence: float = 0.5,
    ) -> None:
        self.dpi = dpi
        self.scale = dpi / 72.0  # PDF native è 72 DPI
        # OCR fallback for text-less PDFs (CAD exports with outlined fonts): when
        # the native text layer is empty we rasterize and OCR the page so the
        # association stage still gets refs/labels. Optional dep ('[ocr]' extra);
        # ocr_engine is injectable for testing.
        self.ocr_fallback = ocr_fallback
        self.ocr_engine = ocr_engine
        self.ocr_min_chars = ocr_min_chars
        self.ocr_min_confidence = ocr_min_confidence

    def extract(self, pdf_path: Path | str) -> list[ExtractedPage]:
        """Estrae tutte le pagine del PDF."""
        pdf_path = Path(pdf_path)
        import fitz

        doc = fitz.open(str(pdf_path))
        pages: list[ExtractedPage] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            extracted = self._extract_page(page, page_num)
            pages.append(extracted)
            logger.info(
                "Extracted page",
                page=page_num,
                segments=len(extracted.segments),
                texts=len(extracted.text_blocks),
                shapes=len(extracted.shapes),
            )

        doc.close()
        logger.info(
            "PDF extraction complete",
            file=str(pdf_path),
            pages=len(pages),
        )
        return pages

    def _extract_page(self, page: Any, page_num: int) -> ExtractedPage:
        result = ExtractedPage(page_num=page_num, raw_rect=page.rect)

        # 1. Disegni vettoriali (linee, archi, rettangoli, cerchi)
        drawings = page.get_drawings()
        for drawing in drawings:
            # B2: rileva cerchi pieni (junction dot) a livello di drawing dict
            circle = self._try_extract_circle(drawing)
            if circle is not None:
                result.shapes.append(circle)
                continue  # gli items sono i Bezier del cerchio, non processarli
            items = drawing.get("items", [])
            for item in items:
                parsed = self._parse_drawing_item(item)
                if parsed is not None:
                    if isinstance(parsed, PDFSegment):
                        result.segments.append(parsed)
                    elif isinstance(parsed, PDFShape):
                        result.shapes.append(parsed)

        # 2. Testo: merge reale degli span via get_text("dict") (fix B3)
        result.text_blocks = self._extract_text_blocks(page)

        # 2b. OCR fallback per PDF senza layer testo (font outlined / scansioni):
        # se l'estrazione nativa è sotto soglia, rasterizza e fai OCR (box incluse).
        if self.ocr_fallback:
            result.text_blocks = self._maybe_ocr(page, result.text_blocks)

        # 3. Post-processing: merge segmenti collineari (come wire_merge)
        result.segments = self._merge_collinear_segments(result.segments)

        return result

    def _maybe_ocr(
        self, page: Any, text_blocks: list[PDFTextBlock]
    ) -> list[PDFTextBlock]:
        """Sostituisce i text block con quelli OCR se il layer testo è vuoto.

        Importa il modulo OCR in modo lazy: se l'extra '[ocr]' non è installato
        si degrada silenziosamente lasciando i (pochi/zero) blocchi nativi.
        """
        from .ocr_fallback import ocr_text_blocks, text_char_count

        if text_char_count(text_blocks) >= self.ocr_min_chars:
            return text_blocks
        try:
            ocr_blocks = ocr_text_blocks(
                page,
                dpi=self.dpi,
                min_confidence=self.ocr_min_confidence,
                ocr=self.ocr_engine,
            )
        except ImportError:
            logger.warning(
                "OCR fallback skipped: rapidocr not installed (install extra 'ocr')"
            )
            return text_blocks
        logger.info("OCR fallback applied", recovered_blocks=len(ocr_blocks))
        return ocr_blocks

    def _parse_drawing_item(self, item: tuple[Any, ...]) -> PDFSegment | PDFShape | None:
        """Parse di un singolo item da page.get_drawings().

        PyMuPDF get_drawings() ritorna items come:
        - ('l', p1, p2) → linea
        - ('c', p1, p2, p3, p4) → curva di Bézier
        - ('re', rect) → rettangolo
        - ('qu', p1, p2, p3) → quadratic Bezier
        """
        kind = item[0]

        if kind == "l":
            # Linea: ('l', (x1,y1), (x2,y2))
            p1, p2 = item[1], item[2]
            return PDFSegment(
                start=(float(p1[0]), float(p1[1])),
                end=(float(p2[0]), float(p2[1])),
                item_type="line",
            )

        if kind == "c":
            # Cubic Bezier: approssima con segmento retto tra start e end
            p1, p4 = item[1], item[4]
            return PDFSegment(
                start=(float(p1[0]), float(p1[1])),
                end=(float(p4[0]), float(p4[1])),
                item_type="curve",
            )

        if kind == "re":
            # Rettangolo: ('re', rect)
            rect = item[1]
            return PDFShape(
                vertices=[
                    (rect.x0, rect.y0),
                    (rect.x1, rect.y0),
                    (rect.x1, rect.y1),
                    (rect.x0, rect.y1),
                ],
                item_type="rect",
            )

        if kind == "qu":
            # Quadrilatero (fitz.Quad): crea PDFShape dai 4 vertici
            quad = item[1]
            ul, ur, lr, ll = quad.ul, quad.ur, quad.lr, quad.ll
            return PDFShape(
                vertices=[
                    (float(ul[0]), float(ul[1])),
                    (float(ur[0]), float(ur[1])),
                    (float(lr[0]), float(lr[1])),
                    (float(ll[0]), float(ll[1])),
                ],
                item_type="polygon",
            )

        # Ignora altri tipi (immagini, clip, etc.)
        return None

    def _try_extract_circle(self, drawing: dict[str, Any]) -> PDFShape | None:
        """B2: individua cerchi pieni (junction dot candidate) dal drawing dict.

        Criteri: tutti gli items sono Bezier ('c'), bbox circa quadrata, fill presente.
        """
        fill: Any = drawing.get("fill")
        rect: Any = drawing.get("rect")
        items: list[Any] = drawing.get("items", [])

        if fill is None or rect is None or not items:
            return None
        # fill può essere un float grayscale o una sequenza RGB
        if isinstance(fill, (int, float)):
            g = int(float(fill) * 255)
            fill_color: tuple[int, int, int] = (g, g, g)
        elif isinstance(fill, (list, tuple)) and len(fill) >= 3:
            fill_color = (
                int(float(fill[0]) * 255),
                int(float(fill[1]) * 255),
                int(float(fill[2]) * 255),
            )
        else:
            return None
        # Un cerchio è approssimato solo da Bezier cubici
        if not all(item[0] == "c" for item in items):
            return None
        w = float(rect.width)
        h = float(rect.height)
        if w <= 0 or h <= 0:
            return None
        # Tolleranza 20% per distinguere cerchi da ellissi distorte
        if abs(w - h) > max(w, h) * 0.2:
            return None
        return PDFShape(
            vertices=[(float(rect.x0), float(rect.y0)), (float(rect.x1), float(rect.y1))],
            item_type="circle",
            fill_color=fill_color,
        )

    def _extract_text_blocks(self, page: Any) -> list[PDFTextBlock]:
        """B3: estrae testo con merge reale degli span tramite get_text("dict").

        Per ogni riga, raggruppa gli span con gap orizzontale < 60% del font-size:
        "R"+"1" → "R1", "10"+"k" → "10k". Emette un PDFTextBlock per gruppo.
        """
        page_dict = page.get_text("dict")
        result: list[PDFTextBlock] = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = testo, 1 = immagine
                continue
            for line in block.get("lines", []):
                raw_spans: list[Any] = line.get("spans", [])
                if not raw_spans:
                    continue
                spans: list[Any] = sorted(raw_spans, key=lambda s: s["bbox"][0])

                # Soglia gap: 60% del font-size medio della riga
                sizes = [float(s["size"]) for s in spans if s.get("size", 0) > 0]
                avg_size = sum(sizes) / len(sizes) if sizes else 10.0
                gap_threshold = avg_size * 0.6

                # Raggruppa span contigui (gap < soglia = stesso token)
                groups: list[list[Any]] = [[spans[0]]]
                for sp in spans[1:]:
                    prev_x1 = float(groups[-1][-1]["bbox"][2])
                    if float(sp["bbox"][0]) - prev_x1 < gap_threshold:
                        groups[-1].append(sp)
                    else:
                        groups.append([sp])

                for group in groups:
                    text = "".join(s.get("text", "") for s in group).strip()
                    if not text:
                        continue
                    x0 = float(group[0]["bbox"][0])
                    y0 = float(min(s["bbox"][1] for s in group))
                    x1 = float(group[-1]["bbox"][2])
                    y1 = float(max(s["bbox"][3] for s in group))
                    first = group[0]
                    result.append(PDFTextBlock(
                        text=text,
                        bbox=(x0, y0, x1, y1),
                        font_size=float(first.get("size", 0.0)),
                        font_name=str(first.get("font", "")),
                        is_bold=bool(first.get("flags", 0) & 16),
                    ))

        return result

    def _merge_collinear_segments(
        self, segments: list[PDFSegment], tolerance: float = 0.5
    ) -> list[PDFSegment]:
        """Unisce segmenti collineari e sovrapposti (versione PDF di wire_merge).

        Algoritmo semplificato: per ogni coppia di segmenti linea, se sono
        collineari e sovrapposti, li unisce.
        """
        if not segments:
            return segments

        lines = [s for s in segments if s.item_type == "line"]
        others = [s for s in segments if s.item_type != "line"]

        changed = True
        max_iter = len(lines) * 2
        iteration = 0

        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            new_lines: list[PDFSegment] = []

            for seg in lines:
                merged = False
                for i, existing in enumerate(new_lines):
                    merged_seg = self._try_merge(seg, existing, tolerance)
                    if merged_seg is not None:
                        new_lines[i] = merged_seg
                        merged = True
                        changed = True
                        break
                if not merged:
                    new_lines.append(seg)

            lines = new_lines

        return lines + others

    @staticmethod
    def _try_merge(
        a: PDFSegment, b: PDFSegment, tolerance: float
    ) -> PDFSegment | None:
        """Tenta di unire due segmenti collineari."""
        # Verifica collinearità (cross product ≈ 0)
        ax, ay = a.end[0] - a.start[0], a.end[1] - a.start[1]
        bx, by = b.end[0] - b.start[0], b.end[1] - b.start[1]
        cross = ax * by - ay * bx
        if abs(cross) > tolerance * max(math.hypot(ax, ay), math.hypot(bx, by)):
            return None

        # Verifica sovrapposizione: 4 punti sulla stessa linea, span <= somma lunghezze
        points = [a.start, a.end, b.start, b.end]

        # Ordina proiettando sulla direzione dominante
        if abs(ax) >= abs(ay):
            points.sort(key=lambda p: p[0])
        else:
            points.sort(key=lambda p: p[1])

        span = math.hypot(
            points[-1][0] - points[0][0], points[-1][1] - points[0][1]
        )
        len_a = a.length
        len_b = b.length
        if span > len_a + len_b + tolerance:
            return None

        return PDFSegment(
            start=points[0],
            end=points[-1],
            item_type="line",
        )
