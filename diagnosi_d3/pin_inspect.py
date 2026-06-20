import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.pdf_parser import VectorExtractor
from src.ml.clustering import SpatialClusterer


def main():
    pdf_path = "test_input/bryston_schematic.pdf"

    extractor = VectorExtractor()
    pages = extractor.extract(pdf_path)
    page = pages[0]

    builder = BipartiteGraphBuilder(cluster_eps=12)
    builder.build_from_page(page)

    target_refs = ["R45", "U25", "R28"]

    all_wire_segs = []
    for n in builder.nets.values():
        all_wire_segs.extend(n.segments)

    with open("diagnosi_d3/pin_inspect.txt", "w") as f:
        for ref in target_refs:
            comp = None
            for c in builder.components.values():
                if c.ref == ref:
                    comp = c
                    break

            if not comp or not comp.cluster:
                f.write(f"[{ref}] non trovato o senza cluster\n\n")
                continue

            endpoints = SpatialClusterer.free_endpoints(comp.cluster.segments)
            cx, cy = comp.cluster.center
            f.write(f"[{ref}]\n")
            f.write(f"- bbox: {comp.bbox}\n")
            f.write(f"- center: ({cx:.2f}, {cy:.2f})\n")
            f.write(f"- {len(endpoints)} free_endpoints:\n")

            best_overall_dist = float("inf")
            best_overall_pin = -1

            dist_bins = {"<5": 0, "5-15": 0, "15-30": 0, ">30": 0}

            for i, ep in enumerate(endpoints):
                best_d2 = float("inf")
                best_seg = None
                for seg in all_wire_segs:
                    d2 = builder._point_to_seg_d2(ep, seg)
                    if d2 < best_d2:
                        best_d2 = d2
                        best_seg = seg

                d = math.sqrt(best_d2) if best_seg else float("inf")

                if d < best_overall_dist:
                    best_overall_dist = d
                    best_overall_pin = i

                if d < 5:
                    dist_bins["<5"] += 1
                elif d < 15:
                    dist_bins["5-15"] += 1
                elif d <= 30:
                    dist_bins["15-30"] += 1
                else:
                    dist_bins[">30"] += 1

                if best_seg:
                    f.write(
                        f"  - pin {i}: ({ep[0]:.2f}, {ep[1]:.2f}) -> nearest_seg dist={d:.2f}, seg_start=({best_seg.start[0]:.2f}, {best_seg.start[1]:.2f}), seg_end=({best_seg.end[0]:.2f}, {best_seg.end[1]:.2f}), len={best_seg.length:.2f}\n"
                    )
                else:
                    f.write(f"  - pin {i}: ({ep[0]:.2f}, {ep[1]:.2f}) -> no wires found\n")

            f.write(f"- Nearest overall: pin {best_overall_pin}, dist={best_overall_dist:.2f}pt\n")
            f.write(
                f"- Distribuzione: {dist_bins['<5']} pin <5pt, {dist_bins['5-15']} pin 5-15pt, {dist_bins['15-30']} pin 15-30pt, {dist_bins['>30']} pin >30pt\n\n"
            )


if __name__ == "__main__":
    main()
