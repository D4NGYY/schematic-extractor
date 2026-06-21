"""UI Streamlit: sistema finito per leggere schemi PDF e parlarci con un LLM.

Avvio:
    streamlit run src/ui/app.py

Permette di:
- caricare un PDF schematico arbitrario (o usare un demo sintetico license-clean);
- scegliere il path componenti (detector YOLO ibrido se i pesi esistono, con
  fallback geometrico automatico);
- vedere l'overlay diagnostico + lo scope gate (densità nel range supportato?);
- chattare con il grafo estratto tramite LLM (Ollama locale o mock mode).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap: assicura che la ROOT del progetto sia nel path, così gli import
# `src.*` funzionano qualunque sia la directory da cui si lancia Streamlit
# (Streamlit mette in sys.path la directory dello script, non la root).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.ui.render import build_overlay

# Demo schematics license-clean (NO Bryston: license non verificata).
_DEMO_DIR = Path("data/kicad/synthetic")


def _find_demo_pdfs() -> list[Path]:
    if not _DEMO_DIR.exists():
        return []
    return sorted(_DEMO_DIR.glob("**/*.pdf"))


def main() -> None:
    st.set_page_config(page_title="Schematic AI Reasoner", layout="wide")
    st.title("Schematic AI Reasoner")
    st.caption("Carica uno schema in PDF → estrai il grafo → parlane con l'LLM.")

    # ---------- Sidebar: source + settings ----------
    st.sidebar.subheader("Sorgente PDF")
    demo_pdfs = _find_demo_pdfs()
    use_upload = st.sidebar.radio(
        "Sorgente",
        ["Carica PDF", "Demo"] if demo_pdfs else ["Carica PDF"],
        horizontal=True,
    )

    pdf_path: str | None = None
    if use_upload == "Carica PDF":
        uploaded = st.sidebar.file_uploader("Schema PDF", type=["pdf"])
        if uploaded is not None:
            cache = Path(".cache/uploads")
            cache.mkdir(parents=True, exist_ok=True)
            out = cache / uploaded.name
            out.write_bytes(uploaded.getvalue())
            pdf_path = str(out)
    else:
        sel = st.sidebar.selectbox("Demo", demo_pdfs, format_func=lambda p: p.name)
        if sel:
            pdf_path = str(sel)

    if pdf_path is None:
        st.info("👆 Carica un PDF o scegli un demo dalla sidebar per iniziare.")
        st.stop()

    dpi = st.sidebar.slider("DPI render", 100, 300, 200, 50)
    ld_raw = st.sidebar.slider(
        "link_dist (0 = adattivo)", 0.0, 30.0, 0.0, 0.5,
        help="Override della distanza di linkage del clustering. 0 = p60 data-derived.",
    )
    link_dist = None if ld_raw == 0.0 else ld_raw
    show_pins = st.sidebar.checkbox("Mostra pin candidati (giallo)", value=True)

    # Detector path (auto se i pesi esistono).
    from src.ml.detector_runner import DetectorRunner, is_available

    det_available = is_available()
    if det_available:
        use_det = st.sidebar.checkbox(
            "Detector YOLO ibrido", value=True,
            help="Usa il detector addestrato con fallback geometrico.",
        )
    else:
        use_det = False
        st.sidebar.caption("_(detector non disponibile: pesi mancanti — geometrico)_")

    # ---------- Estrazione (con cache di sessione) ----------
    cache_key = (pdf_path, dpi, link_dist, use_det)
    if st.session_state.get("_build_key") != cache_key:
        st.session_state._build_key = cache_key
        st.session_state.pop("_built", None)

    if "_built" not in st.session_state:
        with st.spinner("Estrazione del grafo..."):
            detector_comps = None
            if use_det:
                from src.core.pdf_parser import VectorExtractor

                pages = VectorExtractor().extract(pdf_path)
                runner = DetectorRunner()
                detector_comps, _label = runner.run_detector_or_none(pdf_path, pages[0])
            res = build_overlay(
                pdf_path, dpi=dpi, link_dist=link_dist,
                show_pins=show_pins, detector_components=detector_comps,
            )
            scope = assess_scope_counts(res.components, res.nets)
            st.session_state._built = {
                "overlay": res,
                "scope": scope,
                "detector_active": detector_comps is not None,
            }
            # Invalida la chat (grafo cambiato).
            st.session_state.pop("agent", None)
            st.session_state.pop("graph", None)

    built = st.session_state._built
    res = built["overlay"]
    scope = built["scope"]

    tab_overlay, tab_chat = st.tabs(["Overlay", "Chat"])

    with tab_overlay:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Componenti", res.components)
        c2.metric("Net", res.nets)
        c3.metric("Edges", res.edges)
        c4.metric("Isolati", res.isolated, delta=-res.isolated, delta_color="inverse")
        c5.metric("Pin candidati", res.pins)

        if scope.in_scope:
            st.success(f"**In scope** — densità supportata ({scope.nets_per_component} nets/comp).")
        else:
            st.warning(
                f"**Out of scope** — {scope.reason}. "
                "Il risultato è a bassa confidenza (densità oltre il range supportato)."
            )

        path_tag = "Detector YOLO ibrido" if built["detector_active"] else "Geometrico"
        st.caption(
            f"Path: **{path_tag}** · link_dist = "
            f"{res.link_dist if res.link_dist is not None else 'adattivo (p60)'} · "
            "verde = connesso · rosso = isolato · blu = net · giallo = pin"
        )
        st.image(res.image, use_container_width=True)

    with tab_chat:
        _render_chat(pdf_path, link_dist, use_det)


def assess_scope_counts(n_components: int, n_nets: int):
    """Wrapper per evitare import top-level (scope serve solo dopo la build)."""
    from src.core.scope import assess_scope

    return assess_scope(n_components, n_nets)


def _render_chat(pdf_path: str, link_dist: float | None, use_det: bool) -> None:
    st.subheader("LLM Schematic Query")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Impostazioni Chat")
    from src.llm.agent import DEFAULT_MODEL

    _models = [DEFAULT_MODEL, "llama3.1:8b-instruct-q4_K_M", "mistral:7b-instruct-v0.3-q4_K_M"]
    model = st.sidebar.selectbox("Modello", list(dict.fromkeys(_models)))
    max_iters = st.sidebar.slider("Max iterazioni", 1, 20, 10)
    mock_mode = st.sidebar.checkbox("Mock Mode (No Ollama)", value=False)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Quick query dinamici dai ref estratti (no più R1/R5 hardcoded).
    refs = _extracted_refs()
    if refs:
        st.markdown("**Query rapide:**")
        cols = st.columns(4)
        quick = None
        if cols[0].button("Componenti isolati?"):
            quick = "Quali componenti sono isolati?"
        if cols[1].button(f"Vicini di {refs[0]}") and len(refs) >= 1:
            quick = f"Quali componenti sono collegati a {refs[0]}?"
        if cols[2].button("Net con ≥2 componenti"):
            quick = "Quali net collegano almeno 2 componenti?"
        if cols[3].button("Sintesi topologia"):
            quick = "Descrivi brevemente la topologia di questo schema."
    else:
        quick = None

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_trace"):
                with st.expander("Tool Trace"):
                    st.code(msg["tool_trace"])

    user_input = st.chat_input("Fai una domanda sullo schema...") or quick
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"), st.spinner("Analisi in corso..."):
        import asyncio

        from src.core.graph_builder import BipartiteGraphBuilder
        from src.core.pdf_parser import VectorExtractor
        from src.core.text_associator import TextAssociator
        from src.llm.agent import MockClient, OllamaClient, SchematicAgent
        from src.llm.tools import GraphContext
        from src.ml.detector_runner import DetectorRunner

        if st.session_state.get("agent") is None:
            parser = VectorExtractor()
            pages = parser.extract(pdf_path)
            detector_comps = None
            if use_det:
                runner = DetectorRunner()
                detector_comps, _ = runner.run_detector_or_none(pdf_path, pages[0])
            builder = BipartiteGraphBuilder(
                cluster_eps=link_dist, text_associator=TextAssociator()
            )
            graph = builder.build_from_page(pages[0], detector_components=detector_comps)
            st.session_state.graph = graph

            context = GraphContext(graph)
            client = MockClient() if mock_mode else OllamaClient(model=model)
            st.session_state.agent = SchematicAgent(context, client, max_iterations=max_iters)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(st.session_state.agent.query(user_input))

        st.markdown(response)
        # Tool trace: l'agent espone gli ultimi messaggi (system escluso) con
        # i tool eseguiti. Vedi SchematicAgent.last_messages.
        msgs = getattr(st.session_state.agent, "last_messages", [])
        trace_lines = []
        for m in msgs:
            role = m.get("role", "?")
            if role == "system":
                continue
            if role == "tool":
                name = m.get("name", "tool")
                content = str(m.get("content", ""))[:150]
                trace_lines.append(f"tool[{name}] -> {content}")
            elif role == "assistant" and m.get("tool_calls"):
                calls = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
                trace_lines.append(f"assistant calls: {calls}")
            elif role == "assistant":
                trace_lines.append(f"assistant: {str(m.get('content',''))[:150]}")
            else:
                trace_lines.append(f"{role}: {str(m.get('content',''))[:150]}")
        trace = "\n".join(trace_lines[-8:]) if trace_lines else "(nessun tool call)"

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response, "tool_trace": trace}
        )


def _extracted_refs() -> list[str]:
    """Refs del grafo corrente (per le quick query dinamiche). Vuoto se non buildato."""
    graph = st.session_state.get("graph")
    if graph is None:
        return []
    out = []
    for node, data in graph.nodes(data=True):
        if data.get("bipartite") == 0 and isinstance(node, str):
            out.append(node)
    return sorted(out)[:20]


if __name__ == "__main__":
    main()
