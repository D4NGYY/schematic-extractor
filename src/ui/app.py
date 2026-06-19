"""UI Streamlit minimale: banco di prova visivo per clustering e connettivita.

Avvio:
    pip install streamlit
    streamlit run src/ui/app.py

Permette di: scegliere il PDF, regolare DPI e link_dist (override del clustering;
0 = adattivo data-derived), e vedere live l'overlay con bbox componenti
(verde=connesso / rosso=isolato), net (blu) e le metriche del grafo. Serve a
tarare link_dist e ad attaccare il pin->net matching (D3) guardando, non alla cieca.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ui.render import build_overlay

_PDF_DIRS = ["test_input", "data"]


def _find_pdfs() -> list[Path]:
    found: list[Path] = []
    for d in _PDF_DIRS:
        found.extend(sorted(Path(d).glob("**/*.pdf")))
    return found


def main() -> None:
    st.set_page_config(page_title="Schematic Extractor — debug", layout="wide")
    st.title("Schematic Extractor — debug visivo")

    pdfs = _find_pdfs()
    if not pdfs:
        st.error("Nessun PDF trovato in test_input/ o data/.")
        return

    pdf = st.sidebar.selectbox("PDF", pdfs, format_func=lambda p: p.name)
    dpi = st.sidebar.slider("DPI render", 100, 300, 200, 50)
    ld_raw = st.sidebar.slider(
        "link_dist (0 = adattivo)", 0.0, 30.0, 0.0, 0.5,
        help="Override della distanza di linkage del clustering. 0 usa il p60 adattivo data-derived.",
    )
    link_dist = None if ld_raw == 0.0 else ld_raw

    with st.spinner("Costruzione grafo e overlay..."):
        res = build_overlay(str(pdf), dpi=dpi, link_dist=link_dist)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Componenti", res.components)
    c2.metric("Net", res.nets)
    c3.metric("Edges", res.edges)
    c4.metric("Isolati", res.isolated, delta=-res.isolated, delta_color="inverse")

    st.caption(
        f"link_dist = {res.link_dist if res.link_dist is not None else 'adattivo (p60)'} · "
        "verde = componente connesso a >=1 net · rosso = isolato · blu = segmenti net"
    )
    st.image(res.image, use_container_width=True)


if __name__ == "__main__":
    main()
