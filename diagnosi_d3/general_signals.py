import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.graph_builder import BipartiteGraphBuilder
from src.core.pdf_parser import VectorExtractor


def main():
    pdf_path = "test_input/bryston_schematic.pdf"

    extractor = VectorExtractor()
    pages = extractor.extract(pdf_path)
    page = pages[0]

    builder = BipartiteGraphBuilder(cluster_eps=12)
    builder.build_from_page(page)

    components = list(builder.components.values())
    nets = list(builder.nets.values())

    all_wire_segs = []
    for n in nets:
        all_wire_segs.extend(n.segments)

    def point_to_seg_d2(pt, seg):
        return builder._point_to_seg_d2(pt, seg)

    # --- A) DISTRIBUZIONE pin->wire distance ---
    all_pin_dists = []
    for c in components:
        if not c.cluster:
            continue
        endpoints = BipartiteGraphBuilder.select_pins(c.cluster)
        for ep in endpoints:
            best_d2 = float("inf")
            for seg in all_wire_segs:
                d2 = point_to_seg_d2(ep, seg)
                if d2 < best_d2:
                    best_d2 = d2
            if best_d2 != float("inf"):
                all_pin_dists.append(math.sqrt(best_d2))

    all_pin_dists.sort()

    def get_percentile(arr, p):
        if not arr:
            return 0.0
        idx = min(len(arr) - 1, int(len(arr) * p))
        return arr[idx]

    def get_hist(arr, num_bins=10):
        if not arr:
            return []
        vmin = arr[0]
        vmax = arr[-1]
        if vmax == vmin:
            return [len(arr)] + [0] * (num_bins - 1)
        bins = [0] * num_bins
        for v in arr:
            b = int((v - vmin) / (vmax - vmin) * num_bins)
            if b == num_bins:
                b -= 1
            bins[b] += 1
        return bins

    def is_bimodal(hist):
        peaks = 0
        for i in range(1, len(hist) - 1):
            if (
                hist[i] > hist[i - 1]
                and hist[i] > hist[i + 1]
                and hist[i] > sum(hist) / len(hist) * 0.5
            ):
                peaks += 1
        if hist[0] > hist[1] and hist[0] > sum(hist) / len(hist) * 0.5:
            peaks += 1
        if hist[-1] > hist[-2] and hist[-1] > sum(hist) / len(hist) * 0.5:
            peaks += 1
        return peaks >= 2

    # --- B) DISTRIBUZIONE lunghezza wire_segs ---
    wire_lengths = sorted([s.length for s in all_wire_segs])

    # --- C) TOPOLOGIA ENDPOINT DEI WIRE ---
    endpoints_map = defaultdict(int)

    def round_pt(pt):
        return (round(pt[0], 1), round(pt[1], 1))

    for seg in all_wire_segs:
        endpoints_map[round_pt(seg.start)] += 1
        endpoints_map[round_pt(seg.end)] += 1

    degree_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for _pt, deg in endpoints_map.items():
        if deg >= 5:
            degree_counts[5] += 1
        else:
            degree_counts[deg] += 1

    # --- D) PAIRS COLLINEARI ADIACENTI ---
    mergeable_pairs = 0
    segs_with_partner = set()
    for i, s1 in enumerate(all_wire_segs):
        dx1, dy1 = s1.end[0] - s1.start[0], s1.end[1] - s1.start[1]
        l1 = math.hypot(dx1, dy1)
        if l1 < 1e-5:
            continue
        dx1, dy1 = dx1 / l1, dy1 / l1
        for j, s2 in enumerate(all_wire_segs):
            if i >= j:
                continue
            dx2, dy2 = s2.end[0] - s2.start[0], s2.end[1] - s2.start[1]
            l2 = math.hypot(dx2, dy2)
            if l2 < 1e-5:
                continue
            dx2, dy2 = dx2 / l2, dy2 / l2
            dot = abs(dx1 * dx2 + dy1 * dy2)
            if dot > 0.9998:
                min_dist = min(
                    [
                        math.hypot(s1.start[0] - s2.start[0], s1.start[1] - s2.start[1]),
                        math.hypot(s1.start[0] - s2.end[0], s1.start[1] - s2.end[1]),
                        math.hypot(s1.end[0] - s2.start[0], s1.end[1] - s2.start[1]),
                        math.hypot(s1.end[0] - s2.end[0], s1.end[1] - s2.end[1]),
                    ]
                )
                if min_dist < 2.0:
                    mergeable_pairs += 1
                    segs_with_partner.add(i)
                    segs_with_partner.add(j)

    len_with_partner = [s.length for i, s in enumerate(all_wire_segs) if i in segs_with_partner]
    len_without_partner = [
        s.length for i, s in enumerate(all_wire_segs) if i not in segs_with_partner
    ]
    avg_len_with = sum(len_with_partner) / max(1, len(len_with_partner))
    avg_len_without = sum(len_without_partner) / max(1, len(len_without_partner))

    # --- E) FREE_ENDPOINTS PER CLUSTER ---
    ep_dist_to_border = []
    cluster_counts = []
    for c in components:
        if not c.cluster:
            continue
        endpoints = BipartiteGraphBuilder.select_pins(c.cluster)

        xmin, ymin, xmax, ymax = c.bbox
        dists = []
        for ep in endpoints:
            dx = min(abs(ep[0] - xmin), abs(ep[0] - xmax))
            dy = min(abs(ep[1] - ymin), abs(ep[1] - ymax))
            dists.append(min(dx, dy))

        ep_dist_to_border.extend(dists)
        cluster_counts.append({"ref": c.ref, "num_ep": len(endpoints), "dists": dists})

    num_ep_lt_5 = sum(1 for d in ep_dist_to_border if d < 5)
    num_ep_5_20 = sum(1 for d in ep_dist_to_border if 5 <= d <= 20)
    num_ep_gt_20 = sum(1 for d in ep_dist_to_border if d > 20)

    gt_8 = [(c["ref"], c["num_ep"]) for c in cluster_counts if c["num_ep"] > 8]
    gt_50_far = [
        c["ref"]
        for c in cluster_counts
        if sum(1 for d in c["dists"] if d > 20) > len(c["dists"]) * 0.5 and len(c["dists"]) > 0
    ]

    # --- Output JSON ---
    signals = {
        "A": {
            "num_pins": len(all_pin_dists),
            "dist": {
                "min": get_percentile(all_pin_dists, 0),
                "p10": get_percentile(all_pin_dists, 0.10),
                "p25": get_percentile(all_pin_dists, 0.25),
                "p50": get_percentile(all_pin_dists, 0.50),
                "p75": get_percentile(all_pin_dists, 0.75),
                "p90": get_percentile(all_pin_dists, 0.90),
                "p95": get_percentile(all_pin_dists, 0.95),
                "max": get_percentile(all_pin_dists, 1.0),
            },
            "hist": get_hist(all_pin_dists),
            "bimodal": is_bimodal(get_hist(all_pin_dists)),
            "pct_0": sum(1 for d in all_pin_dists if d < 0.1) / max(1, len(all_pin_dists)) * 100,
            "pct_5": sum(1 for d in all_pin_dists if d < 5) / max(1, len(all_pin_dists)) * 100,
            "pct_15": sum(1 for d in all_pin_dists if d < 15) / max(1, len(all_pin_dists)) * 100,
        },
        "B": {
            "num_segs": len(wire_lengths),
            "dist": {
                "min": get_percentile(wire_lengths, 0),
                "p25": get_percentile(wire_lengths, 0.25),
                "p50": get_percentile(wire_lengths, 0.50),
                "p75": get_percentile(wire_lengths, 0.75),
                "p90": get_percentile(wire_lengths, 0.90),
                "max": get_percentile(wire_lengths, 1.0),
            },
            "hist": get_hist(wire_lengths),
            "bimodal": is_bimodal(get_hist(wire_lengths)),
            "pct_10": sum(1 for length in wire_lengths if length < 10) / max(1, len(wire_lengths)) * 100,
            "pct_20": sum(1 for length in wire_lengths if length < 20) / max(1, len(wire_lengths)) * 100,
            "pct_50": sum(1 for length in wire_lengths if length < 50) / max(1, len(wire_lengths)) * 100,
        },
        "C": {
            "unique_endpoints": len(endpoints_map),
            "degrees": degree_counts,
            "pct_g1": degree_counts[1] / max(1, len(endpoints_map)) * 100,
            "pct_g3_plus": (degree_counts[3] + degree_counts[4] + degree_counts[5])
            / max(1, len(endpoints_map))
            * 100,
        },
        "D": {
            "mergeable_pairs": mergeable_pairs,
            "avg_len_with_partner": avg_len_with,
            "avg_len_without_partner": avg_len_without,
        },
        "E": {
            "num_clusters": len(components),
            "gt_8": gt_8,
            "dists": {"lt5": num_ep_lt_5, "5to20": num_ep_5_20, "gt20": num_ep_gt_20},
            "gt_50_far": gt_50_far,
        },
    }

    with open("diagnosi_d3/signals_after.json", "w") as f:
        json.dump(signals, f, indent=2)


if __name__ == "__main__":
    main()
