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
    show_pins = st.sidebar.checkbox("Mostra pin candidati (giallo)", value=True)

    tab_overlay, tab_chat = st.tabs(["Overlay", "Chat"])
    
    with tab_overlay:
        with st.spinner("Costruzione grafo e overlay..."):
            res = build_overlay(str(pdf), dpi=dpi, link_dist=link_dist, show_pins=show_pins)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Componenti", res.components)
        c2.metric("Net", res.nets)
        c3.metric("Edges", res.edges)
        c4.metric("Isolati", res.isolated, delta=-res.isolated, delta_color="inverse")
        c5.metric("Pin candidati", res.pins)

        st.caption(
            f"link_dist = {res.link_dist if res.link_dist is not None else 'adattivo (p60)'} · "
            "verde = connesso a >=1 net · rosso = isolato · blu = segmenti net · giallo = pin candidati (free-endpoint)"
        )
        st.image(res.image, use_container_width=True)
        
    with tab_chat:
        st.subheader("LLM Schematic Query")
        
        # Chat Sidebar settings
        st.sidebar.markdown("---")
        st.sidebar.subheader("Impostazioni Chat")
        model = st.sidebar.selectbox("Modello", ["llama3.1:8b-instruct-q4_K_M", "llama3:8b", "gpt-4o"])
        max_iters = st.sidebar.slider("Max iterazioni", 1, 20, 10)
        mock_mode = st.sidebar.checkbox("Mock Mode (No Ollama)", value=True)
        
        # Session state management
        if "current_pdf" not in st.session_state or st.session_state.current_pdf != str(pdf):
            st.session_state.current_pdf = str(pdf)
            st.session_state.chat_history = []
            st.session_state.agent = None
            st.session_state.graph = None
            
        # Demo buttons
        st.markdown("**Query rapide:**")
        cols = st.columns(4)
        quick_query = None
        if cols[0].button("Quali componenti sono isolati?"): quick_query = "Quali componenti sono isolati?"
        if cols[1].button("Elenca componenti su GND"): quick_query = "Elenca componenti su GND"
        if cols[2].button("Trova path tra R1 e R5"): quick_query = "Trova path tra R1 e R5"
        if cols[3].button("Cerca resistori da 10k"): quick_query = "Cerca resistori da 10k"
            
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "tool_trace" in msg and msg["tool_trace"]:
                    with st.expander("Tool Trace"):
                        st.code(msg["tool_trace"])
                        
        user_input = st.chat_input("Fai una domanda sullo schema...") or quick_query
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.chat_message("assistant"):
                with st.spinner("Analisi in corso..."):
                    import asyncio
                    from src.core.pdf_parser import VectorExtractor
                    from src.core.graph_builder import BipartiteGraphBuilder
                    from src.core.text_associator import TextAssociator
                    from src.llm.tools import GraphContext
                    from src.llm.agent import SchematicAgent, OllamaClient, MockClient
                    
                    if st.session_state.agent is None:
                        # Build graph for agent
                        parser = VectorExtractor()
                        pages = parser.extract(str(pdf))
                        builder = BipartiteGraphBuilder(cluster_eps=link_dist, text_associator=TextAssociator())
                        graph = builder.build_from_page(pages[0])
                        st.session_state.graph = graph
                        
                        context = GraphContext(graph)
                        client = MockClient() if mock_mode else OllamaClient(model=model)
                        st.session_state.agent = SchematicAgent(context, client, max_iterations=max_iters)
                    
                    # Intercept stdout to capture tool trace or we can just fetch the history from agent
                    # For simplicity, we just run the query.
                    # We create an event loop if not present
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                    response = loop.run_until_complete(st.session_state.agent.query(user_input))
                    
                    st.markdown(response)
                    # We could dump the agent's internal message trace as tool trace
                    trace = "\n".join([f"{m.get('role')}: {str(m)[:200]}" for m in st.session_state.agent.client_history[-5:]])
                    with st.expander("Tool Trace"):
                        st.code(trace)
                        
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "tool_trace": trace
                    })


if __name__ == "__main__":
    main()
