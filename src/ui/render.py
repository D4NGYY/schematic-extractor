"""Render headless di una pagina schematic con overlay diagnostico.

Banco di prova visivo per tarare il clustering (link_dist) e ispezionare la
connettivita (pin->net / D3): renderizza il PDF e sovrappone le bounding box dei
componenti (verde = connesso a >=1 net, rosso = isolato), i loro ref e i
segmenti-filo delle net. Usato sia headless (salva PNG) sia dalla UI Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.pdf_parser import VectorExtractor

_CONNECTED = (40, 180, 80, 255)   # verde
_ISOLATED = (220, 50, 50, 255)    # rosso
_NET = (30, 120, 220, 170)        # blu


@dataclass
class OverlayResult:
    image: Image.Image
    components: int
    nets: int
    edges: int
    isolated: int
    link_dist: float | None


def render_page_png(pdf_path: str, page_num: int = 0, dpi: int = 200) -> tuple[Image.Image, float]:
    """Renderizza una pagina PDF in un'immagine PIL. Restituisce (img, zoom)."""
    zoom = dpi / 72.0
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img.convert("RGBA"), zoom


def build_overlay(
    pdf_path: str,
    page_num: int = 0,
    dpi: int = 200,
    link_dist: float | None = None,
) -> OverlayResult:
    """Costruisce il grafo (con link_dist scelto) e disegna l'overlay diagnostico."""
    page = VectorExtractor().extract(pdf_path)[page_num]
    builder = BipartiteGraphBuilder(cluster_eps=link_dist)
    graph = builder.build_from_page(page)

    img, zoom = render_page_png(pdf_path, page_num, dpi)
    draw = ImageDraw.Draw(img, "RGBA")

    # Segmenti-filo delle net (blu).
    for net in builder.nets.values():
        for seg in net.segments:
            draw.line(
                [seg.start[0] * zoom, seg.start[1] * zoom, seg.end[0] * zoom, seg.end[1] * zoom],
                fill=_NET,
                width=2,
            )

    # Bounding box componenti: verde se connesso, rosso se isolato.
    isolated = 0
    for node_id, comp in builder.components.items():
        if comp.bbox is None:
            continue
        connected = graph.degree(node_id) > 0
        if not connected:
            isolated += 1
        color = _CONNECTED if connected else _ISOLATED
        x0, y0, x1, y1 = (v * zoom for v in comp.bbox)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, max(0.0, y0 - 11)), comp.ref, fill=color)

    return OverlayResult(
        image=img,
        components=len(builder.components),
        nets=len(builder.nets),
        edges=graph.number_of_edges(),
        isolated=isolated,
        link_dist=link_dist,
    )


def save_overlay(pdf_path: str, out_path: str, **kw: Any) -> OverlayResult:
    """Genera l'overlay e lo salva su file. Restituisce le metriche."""
    res = build_overlay(pdf_path, **kw)
    res.image.convert("RGB").save(out_path)
    return res
