import json
from pathlib import Path

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.kicad_gt_reader import build_gt_graph, parse_kicad_sch
from src.core.pdf_parser import VectorExtractor


def main():
    pdf_path = Path("test_input/multi_schematic/arduino_micro/arduino_micro.pdf")
    sch_path = Path("test_input/multi_schematic/arduino_micro/arduino_micro.kicad_sch")

    gt_sch = parse_kicad_sch(sch_path)
    gt_graph = build_gt_graph(gt_sch)

    extractor = VectorExtractor()
    pages = extractor.extract(str(pdf_path))
    builder = BipartiteGraphBuilder(cluster_eps=0)
    builder.build_from_page(pages[0])

    num_comp_gt = len(gt_graph.components)
    num_nets_gt = len(gt_graph.nets)
    refs_gt = list(gt_graph.components)
    nets_gt = list(gt_graph.nets.keys())

    components_ext = builder.components
    num_comp_ext = len(components_ext)
    num_nets_ext = len(builder.nets)
    refs_ext = [c.ref for c in components_ext.values()]

    # Overlap refs
    overlap_refs = set(refs_gt).intersection(set(refs_ext))
    overlap_count = len(overlap_refs)

    # Clustering distribution
    cluster_sizes = {"1-2": 0, "3-5": 0, "6+": 0}
    for c in components_ext.values():
        if c.cluster:
            n = len(c.cluster.segments)
            if n <= 2:
                cluster_sizes["1-2"] += 1
            elif n <= 5:
                cluster_sizes["3-5"] += 1
            else:
                cluster_sizes["6+"] += 1

    # Sample components
    samples = []
    for c in list(components_ext.values())[:10]:
        samples.append({
            "ref": c.ref,
            "bbox": c.bbox,
            "num_segments": len(c.cluster.segments) if c.cluster else 0
        })

    output = {
        "GT": {
            "num_components": num_comp_gt,
            "num_nets": num_nets_gt,
            "refs": refs_gt[:20] + ["..."] if len(refs_gt)>20 else refs_gt,
            "nets": nets_gt[:20] + ["..."] if len(nets_gt)>20 else nets_gt
        },
        "Extracted": {
            "num_components": num_comp_ext,
            "num_nets": num_nets_ext,
            "refs": refs_ext[:20] + ["..."] if len(refs_ext)>20 else refs_ext,
            "samples": samples
        },
        "Diagnosis": {
            "overlap_refs_count": overlap_count,
            "total_gt_refs": len(refs_gt),
            "total_ext_refs": len(refs_ext),
            "cluster_size_distribution": cluster_sizes
        }
    }

    with open("diagnosi_d3/arduino_micro_diagnosis.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
