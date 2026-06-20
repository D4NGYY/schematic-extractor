import json
import math

from src.core.pdf_parser import PDFSegment, VectorExtractor
from src.ml.clustering import SpatialClusterer


def is_axis_aligned(seg: PDFSegment, tol: float = 1.0) -> bool:
    if seg.item_type == "curve":
        return False
    dx = abs(seg.end[0] - seg.start[0])
    dy = abs(seg.end[1] - seg.start[1])
    return dx < tol or dy < tol


def main():
    extractor = VectorExtractor()
    page = extractor.extract("test_input/bryston_schematic.pdf")[0]

    total_segs = page.segments

    axis_aligned_count = sum(1 for s in total_segs if is_axis_aligned(s))
    curve_count = sum(1 for s in total_segs if s.item_type == "curve")

    pct_axis_aligned = (axis_aligned_count / max(1, len(total_segs))) * 100
    pct_curve = (curve_count / max(1, len(total_segs))) * 100

    # Simulate current separate_wires (which is length-based > 10pt)
    # The actual implementation in SpatialClusterer:
    # min_wire_length = 10.0
    symbol_segs, wire_segs = SpatialClusterer.separate_wires(total_segs)

    # Calculate p50 length of all segments
    lengths = []
    for s in total_segs:
        if s.item_type == "line":
            length = math.hypot(s.end[0] - s.start[0], s.end[1] - s.start[1])
            lengths.append(length)
    lengths.sort()
    p50_len = lengths[len(lengths) // 2] if lengths else 0.0

    # Find axis-aligned AND short in symbol_segs
    short_aa_symbol_segs = []
    for s in symbol_segs:
        if is_axis_aligned(s):
            length = math.hypot(s.end[0] - s.start[0], s.end[1] - s.start[1])
            if length < p50_len:
                short_aa_symbol_segs.append(s)

    # Run clustering
    clusterer = SpatialClusterer(eps=12.0, min_samples=2)
    clusters = clusterer.cluster(symbol_segs, page.shapes)

    # Find absorbed segments that extend beyond cluster bbox by > 2*link_dist (24pt)
    absorbed_count = 0
    link_dist = 12.0
    x_tol = 2 * link_dist

    absorbed_details = []
    for c in clusters:
        xmin, ymin, xmax, ymax = c.bbox
        for s in c.segments:
            if is_axis_aligned(s):
                # check how far it extends outside
                # line is (x1,y1) to (x2,y2)
                # min_x, max_x of segment
                sx_min = min(s.start[0], s.end[0])
                sx_max = max(s.start[0], s.end[0])
                sy_min = min(s.start[1], s.end[1])
                sy_max = max(s.start[1], s.end[1])

                # outside distance
                out_left = max(0, xmin - sx_min)
                out_right = max(0, sx_max - xmax)
                out_top = max(0, ymin - sy_min)
                out_bottom = max(0, sy_max - ymax)

                max_out = max(out_left, out_right, out_top, out_bottom)
                if max_out > x_tol:
                    absorbed_count += 1
                    absorbed_details.append(
                        {
                            "cluster_id": c.cluster_id,
                            "segment_len": math.hypot(s.end[0] - s.start[0], s.end[1] - s.start[1]),
                            "max_out": max_out,
                        }
                    )

    results = {
        "pre_clustering": {
            "total_segments": len(total_segs),
            "pct_axis_aligned": pct_axis_aligned,
            "pct_curve": pct_curve,
            "p50_length": p50_len,
        },
        "post_separate_wires": {
            "wire_segs_count": len(wire_segs),
            "symbol_segs_count": len(symbol_segs),
            "symbol_segs_axis_aligned_and_short": len(short_aa_symbol_segs),
        },
        "post_clustering": {
            "absorbed_segments_count": absorbed_count,
            "absorbed_details": absorbed_details,
        },
    }

    with open("diagnosi_d3/separate_wires_audit.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
