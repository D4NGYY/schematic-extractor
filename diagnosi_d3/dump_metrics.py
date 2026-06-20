import csv
import json
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

    components = list(builder.components.values())
    nets = list(builder.nets.values())

    num_components = len(components)
    num_nets = len(nets)
    num_edges = builder.graph.number_of_edges()

    # Calculate isolated components
    isolated_components = []
    for c in components:
        if all(p.connected_net is None for p in c.pins):
            isolated_components.append(c)

    num_isolated = len(isolated_components)

    nets_data = []
    for net in nets:
        total_length = sum(seg.length for seg in net.segments)
        if net.segments:
            xmin = min(min(seg.start[0], seg.end[0]) for seg in net.segments)
            ymin = min(min(seg.start[1], seg.end[1]) for seg in net.segments)
            xmax = max(max(seg.start[0], seg.end[0]) for seg in net.segments)
            ymax = max(max(seg.start[1], seg.end[1]) for seg in net.segments)
            bbox = (xmin, ymin, xmax, ymax)
        else:
            bbox = None
        nets_data.append(
            {
                "id": net.net_id,
                "num_segments": len(net.segments),
                "total_length_pt": total_length,
                "bbox": bbox,
            }
        )

    all_wire_segs = []
    for n in nets:
        all_wire_segs.extend(n.segments)

    def nearest_wire_dist(pt):
        best_d2 = float("inf")
        for seg in all_wire_segs:
            d2 = builder._point_to_seg_d2(pt, seg)
            if d2 < best_d2:
                best_d2 = d2
        return math.sqrt(best_d2) if best_d2 != float("inf") else None

    all_endpoints_dists = []
    isolated_data = []

    for c in components:
        if c.cluster is None:
            continue
        endpoints = SpatialClusterer.free_endpoints(c.cluster.segments)
        c_min_dist = float("inf")
        for ep in endpoints:
            d = nearest_wire_dist(ep)
            if d is not None:
                all_endpoints_dists.append(d)
                c_min_dist = min(c_min_dist, d)

        if c in isolated_components:
            isolated_data.append(
                {
                    "id": c.node_id,
                    "bbox": c.bbox,
                    "nearest_wire_distance_pt": c_min_dist if c_min_dist != float("inf") else None,
                    "num_free_endpoints": len(endpoints),
                    "ref": c.ref,
                    "endpoints": endpoints,
                }
            )

    all_endpoints_dists.sort()

    def p(q):
        if not all_endpoints_dists:
            return None
        idx = min(len(all_endpoints_dists) - 1, int(len(all_endpoints_dists) * q))
        return all_endpoints_dists[idx]

    dist_stats = {
        "min": all_endpoints_dists[0] if all_endpoints_dists else None,
        "p25": p(0.25),
        "p50": p(0.50),
        "p75": p(0.75),
        "max": all_endpoints_dists[-1] if all_endpoints_dists else None,
    }

    metrics = {
        "num_components": num_components,
        "num_nets": num_nets,
        "num_edges": num_edges,
        "num_isolated": num_isolated,
        "nets": nets_data,
        "isolated": isolated_data,
        "pin_distances": dist_stats,
    }

    with open("diagnosi_d3/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("diagnosi_d3/topology.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "component_id",
                "ref",
                "bbox_xmin",
                "bbox_ymin",
                "bbox_xmax",
                "bbox_ymax",
                "num_free_endpoints",
                "connected",
                "nearest_net_id",
                "nearest_net_distance_pt",
                "is_isolated",
            ]
        )
        for c in components:
            if c.cluster is None:
                continue
            endpoints = SpatialClusterer.free_endpoints(c.cluster.segments)
            connected = any(p.connected_net is not None for p in c.pins)
            is_iso = not connected

            best_dist = float("inf")
            best_net_id = None
            for ep in endpoints:
                for n in nets:
                    for seg in n.segments:
                        d2 = builder._point_to_seg_d2(ep, seg)
                        if d2 < best_dist:
                            best_dist = d2
                            best_net_id = n.net_id

            writer.writerow(
                [
                    c.node_id,
                    c.ref,
                    c.bbox[0] if c.bbox else "",
                    c.bbox[1] if c.bbox else "",
                    c.bbox[2] if c.bbox else "",
                    c.bbox[3] if c.bbox else "",
                    len(endpoints),
                    connected,
                    best_net_id,
                    math.sqrt(best_dist) if best_dist != float("inf") else "",
                    is_iso,
                ]
            )

    isolated_data.sort(
        key=lambda x: (
            x["nearest_wire_distance_pt"]
            if x["nearest_wire_distance_pt"] is not None
            else float("inf")
        )
    )
    top_3 = isolated_data[:3]

    with open("diagnosi_d3/close_isolated.txt", "w") as f:
        for iso in top_3:
            f.write(
                f"ID={iso['id']}, ref={iso['ref']}, dist={iso['nearest_wire_distance_pt']:.2f}, num_free_endpoints={iso['num_free_endpoints']}\n"
            )
            f.write(f"  bbox={iso['bbox']}\n")
            f.write(f"  free_endpoints={iso['endpoints']}\n")
            segs_dist = []
            for ep in iso["endpoints"]:
                for seg in all_wire_segs:
                    d = math.sqrt(builder._point_to_seg_d2(ep, seg))
                    segs_dist.append((d, seg))
            segs_dist.sort(key=lambda x: x[0])
            for i, (d, seg) in enumerate(segs_dist[:3]):
                f.write(
                    f"  nearest_wires={i + 1}: dist={d:.2f}, start={seg.start}, end={seg.end}, len={seg.length:.2f}\n"
                )
            f.write("\n")


if __name__ == "__main__":
    main()
