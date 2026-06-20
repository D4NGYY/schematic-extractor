from dataclasses import dataclass


@dataclass
class PDFSegment:
    start: tuple[float, float]
    end: tuple[float, float]

@dataclass
class ComponentCluster:
    bbox: tuple[float, float, float, float]
    segments: list[PDFSegment]

def free_endpoints(segments):
    from collections import defaultdict
    counts = defaultdict(int)
    for s in segments:
        counts[s.start] += 1
        counts[s.end] += 1
    return [pt for pt, c in counts.items() if c == 1]

def _derive_pin_border_tol(cluster):
    free = free_endpoints(cluster.segments)
    x0, y0, x1, y1 = cluster.bbox
    import numpy as np
    dists = []
    for px, py in free:
        dl = px - x0
        dr = x1 - px
        dt = py - y0
        db = y1 - py
        dists.append(min(dl, dr, dt, db))
    print(f"dists: {dists}")
    return float(np.median(dists))

def select_pins(cluster):
    free = free_endpoints(cluster.segments)
    tol = _derive_pin_border_tol(cluster)
    x0, y0, x1, y1 = cluster.bbox
    pins = []
    for px, py in free:
        dl = px - x0
        dr = x1 - px
        dt = py - y0
        db = y1 - py
        dist = min(dl, dr, dt, db)
        print(f"pt: {(px, py)}, dist: {dist}, tol: {tol}")
        if dist <= tol + 1e-3:
            pins.append((px, py))
    return pins

c = ComponentCluster(
    bbox=(-5, 2, 15, 8),
    segments=[
        PDFSegment(start=(0, 5), end=(-5, 5)),
        PDFSegment(start=(10, 5), end=(15, 5)),
        PDFSegment(start=(5, 4), end=(5, 6)),
    ]
)
print(select_pins(c))
