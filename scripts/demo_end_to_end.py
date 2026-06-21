"""End-to-end demo: schematic PDF -> Components<->Nets graph -> topology Q&A.

Runs the FULL pipeline on an in-scope board and answers questions about the
circuit using the same GraphContext tools the LLM agent calls. The tool-based
answers are deterministic and need no Ollama, so the demo runs anywhere; pass
--llm to additionally route natural-language questions through the LLM agent
(requires a local Ollama).

  python scripts/demo_end_to_end.py                 # default: sallen_key
  python scripts/demo_end_to_end.py --board rectifier
  python scripts/demo_end_to_end.py --pdf path/to/schematic.pdf
  python scripts/demo_end_to_end.py --llm           # + real LLM Q&A via Ollama
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.logging_config import configure_logging
from src.core.pdf_parser import VectorExtractor
from src.core.scope import assess_scope
from src.llm.tools import GraphContext

BASE = Path("test_input/multi_schematic")


def banner(t: str) -> None:
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="sallen_key")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--weights", default=None,
                    help="YOLO weights -> use the detector path (clean components). "
                         "Without it, the geometric baseline runs (noisier).")
    ap.add_argument("--dpi", type=float, default=150.0, help="must match the model's training dpi")
    ap.add_argument("--images-dir", default="data/detector/images")
    ap.add_argument("--llm", action="store_true", help="also run the Ollama LLM agent")
    args = ap.parse_args()
    configure_logging(log_level="CRITICAL")

    pdf = Path(args.pdf) if args.pdf else next((BASE / args.board).glob("*.pdf"))
    name = pdf.stem

    banner(f"1. EXTRACT  —  {name}")
    pages = VectorExtractor().extract(str(pdf))
    builder = BipartiteGraphBuilder()
    mode = "geometric"
    detector_comps = None
    if args.weights:
        import fitz  # type: ignore
        from ultralytics import YOLO  # noqa: PLC0415

        from src.ml.detector_source import Detection, DetectorComponentSource
        img = Path(args.images_dir) / f"{name}.png"
        if not img.exists():
            img = Path(f"/tmp/_demo_{int(args.dpi)}_{name}.png")
            fitz.open(str(pdf))[0].get_pixmap(
                matrix=fitz.Matrix(args.dpi / 72, args.dpi / 72)
            ).save(str(img))
        r = YOLO(args.weights)(str(img), verbose=False)[0]
        dets = [
            Detection(class_name=r.names[int(c)], bbox_px=tuple(xy), confidence=float(p))
            for xy, c, p in zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), strict=False
            )
        ]
        src = DetectorComponentSource(dpi=args.dpi)
        detector_comps = src.components_or_fallback(dets, pages[0])
        mode = "detector" if detector_comps is not None else "detector->geometric fallback"
    graph = builder.build_from_page(pages[0], detector_components=detector_comps)
    print(f"mode: {mode}")
    ncomp, nnet = len(builder.components), len(builder.nets)
    print(f"PDF pages: {len(pages)}   graph: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges  ({ncomp} components, {nnet} nets)")

    banner("2. SCOPE / CONFIDENCE")
    sc = assess_scope(ncomp, nnet)
    print(f"in_scope={sc.in_scope}  confidence={sc.confidence.upper()}  "
          f"nets/comp={sc.nets_per_component}")
    print(f"-> {sc.reason}")

    ctx = GraphContext(graph)
    comps = sorted(ctx.components)
    print(f"\ncomponents: {', '.join(comps)}")

    banner("3. TOPOLOGY Q&A  (GraphContext tools — same the LLM calls)")
    qa = []

    r = ctx.get_nets_summary(min_components=2)
    nets = r.get("nets", r)
    qa.append(("Which nets connect 2+ components?",
               f"{r.get('count', len(nets) if isinstance(nets, list) else '?')} nets "
               f"(e.g. {str(nets)[:160]})"))

    r = ctx.find_isolated()
    qa.append(("Any isolated components?", str(r)[:200]))

    target = "U1" if "U1" in ctx.components else comps[0]
    qa.append((f"What is {target}?", str(ctx.get_component_info(target))[:200]))
    qa.append((f"What connects to {target}?", str(ctx.get_neighbors(target))[:240]))

    r = ctx.search_by_value("1k")
    qa.append(("Find components valued '1k'", str(r)[:160]))

    for q, a in qa:
        print(f"\nQ: {q}\nA: {a}")

    if args.llm:
        banner("4. NATURAL-LANGUAGE Q&A  (LLM agent via Ollama)")
        import asyncio

        from src.llm.agent import DEFAULT_MODEL, OllamaClient, SchematicAgent
        agent = SchematicAgent(graph_context=ctx, llm_client=OllamaClient(model=DEFAULT_MODEL))
        for q in (
            f"Quali componenti sono collegati a {target}?",
            "Quali componenti sono isolati?",
        ):
            print(f"\nQ: {q}")
            print("A:", asyncio.run(agent.query(q)))

    banner("DONE")
    print("Full pipeline ran end-to-end. Re-run with --llm for natural-language Q&A.")


if __name__ == "__main__":
    main()
