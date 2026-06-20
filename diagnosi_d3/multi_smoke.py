import json
import logging
from pathlib import Path

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.kicad_gt_reader import build_gt_graph, parse_kicad_sch
from src.core.pdf_parser import VectorExtractor

logger = logging.getLogger("multi_smoke")
logging.basicConfig(level=logging.INFO)

def calculate_pair_f1(ext_nets: list[list[tuple[str, str]]], gt_nets: list[list[tuple[str, str]]]) -> float:
    # Convert to sets of pairs (combinations)
    ext_pairs = set()
    for net in ext_nets:
        # Sort to ensure consistent pair ordering
        pins = sorted(net)
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                ext_pairs.add((pins[i], pins[j]))

    gt_pairs = set()
    for net in gt_nets:
        pins = sorted(net)
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                gt_pairs.add((pins[i], pins[j]))

    if not gt_pairs and not ext_pairs:
        return 1.0
    if not gt_pairs or not ext_pairs:
        return 0.0

    tp = len(ext_pairs & gt_pairs)
    fp = len(ext_pairs - gt_pairs)
    fn = len(gt_pairs - ext_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def main():
    base_dir = Path("test_input/multi_schematic")
    output_file = Path("diagnosi_d3/multi_smoke_report.json")

    # We add Bryston as baseline
    boards = [
        {"name": "Bryston", "pdf": Path("test_input/bryston_schematic.pdf"), "sch": None}
    ]

    if base_dir.exists():
        for board_dir in base_dir.iterdir():
            if board_dir.is_dir():
                pdfs = list(board_dir.glob("*.pdf"))
                schs = list(board_dir.glob("*.kicad_sch"))
                if pdfs:
                    boards.append({
                        "name": board_dir.name,
                        "pdf": pdfs[0],
                        "sch": schs[0] if schs else None
                    })

    results = []

    for board in boards:
        name = board["name"]
        pdf_path = board["pdf"]
        sch_path = board["sch"]

        logger.info(f"Processing {name}")

        try:
            extractor = VectorExtractor()
            pages = extractor.extract(str(pdf_path))
            if not pages:
                continue

            page = pages[0]

            # Using adaptive link_dist=None
            builder = BipartiteGraphBuilder(cluster_eps=None)
            builder.build_from_page(page)

            components = list(builder.components.values())
            nets = list(builder.nets.values())

            isolated = sum(1 for c in components if all(p.connected_net is None for p in c.pins))
            g3_plus = sum(1 for c in components if len([p for p in c.pins if p.connected_net]) >= 3)
            # wait, g3+ in the prompt probably meant endpoint_grado_3+ from intersections, but we can report components with >=3 pins
            # or just report components for simplicity

            f1_gt = None
            if sch_path and sch_path.exists():
                gt_sch = parse_kicad_sch(sch_path)
                gt_graph = build_gt_graph(gt_sch)

                # Extracted nets
                ext_net_lists = []
                for net in nets:
                    net_refs = set()
                    for ref, c in builder.components.items():
                        for p in c.pins:
                            if p.connected_net == net.net_id:
                                net_refs.add(ref)
                    if len(net_refs) > 1:
                        ext_net_lists.append(list(net_refs))

                # GT nets
                gt_net_lists = []
                for pins in gt_graph.nets.values():
                    net_refs = set(ref for ref, pin in pins)
                    if len(net_refs) > 1:
                        gt_net_lists.append(list(net_refs))

                f1_gt = calculate_pair_f1(ext_net_lists, gt_net_lists)

            results.append({
                "board": name,
                "components": len(components),
                "nets": len(nets),
                "edges": builder.graph.number_of_edges(),
                "isolated": isolated,
                "g3+": g3_plus,
                "f1_gt": f1_gt
            })

        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            results.append({
                "board": name,
                "components": "ERROR",
                "nets": "ERROR",
                "edges": "ERROR",
                "isolated": "ERROR",
                "g3+": "ERROR",
                "f1_gt": "ERROR"
            })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    print(f"| {'Board':<15} | {'components':<10} | {'nets':<6} | {'edges':<6} | {'isolated':<8} | {'g3+':<5} | {'F1_GT':<6} |")
    print(f"|{'-'*17}|{'-'*12}|{'-'*8}|{'-'*8}|{'-'*10}|{'-'*7}|{'-'*8}|")
    for r in results:
        f1_str = f"{r['f1_gt']:.2f}" if r["f1_gt"] is not None and isinstance(r["f1_gt"], float) else str(r["f1_gt"])
        print(f"| {r['board']:<15} | {r['components']:<10} | {r['nets']:<6} | {r['edges']:<6} | {r['isolated']:<8} | {r['g3+']:<5} | {f1_str:<6} |")

if __name__ == "__main__":
    main()
