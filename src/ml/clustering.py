from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from sklearn.cluster import DBSCAN

from src.core.pdf_parser import PDFSegment, PDFShape, PDFTextBlock

logger = structlog.get_logger("clustering")


@dataclass
class ComponentCluster:
    """Cluster di segmenti/forme che rappresenta un componente candidate."""
    cluster_id: int
    segments: list[PDFSegment]
    shapes: list[PDFShape]
    text_blocks: list[PDFTextBlock]
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    center: tuple[float, float]

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def total_segment_length(self) -> float:
        return sum(s.length for s in self.segments)

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


class SpatialClusterer:
    """Clustering spaziale di segmenti e forme PDF usando DBSCAN con epsilon adattivo.

    Euristiche:
    - Epsilon adattivo basato su mediana delle distanze inter-segmento
    - Min_samples = 2 (componenti hanno almeno 2 pin/wire)
    - Features per clustering: punto medio di ogni segmento
    """

    def __init__(self, eps: float | None = None, min_samples: int = 2) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, segments: list[PDFSegment], shapes: list[PDFShape]) -> list[ComponentCluster]:
        """Clusterizza segmenti e forme in componenti candidate."""
        if not segments and not shapes:
            return []

        # Punti per clustering: midpoint di ogni segmento + centro di ogni shape
        points: list[tuple[float, float]] = []
        point_to_obj: list[tuple[str, int]] = []  # ("seg", idx) o ("shape", idx)

        for i, seg in enumerate(segments):
            points.append(seg.midpoint())
            point_to_obj.append(("seg", i))

        for i, shape in enumerate(shapes):
            cx = (shape.bbox[0] + shape.bbox[2]) / 2
            cy = (shape.bbox[1] + shape.bbox[3]) / 2
            points.append((cx, cy))
            point_to_obj.append(("shape", i))

        x = np.array(points)
        eps = self.eps if self.eps is not None else self._estimate_eps(x)

        db = DBSCAN(eps=eps, min_samples=self.min_samples).fit(x)
        labels = db.labels_

        # Raggruppa per label
        clusters: dict[int, ComponentCluster] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue  # noise
            if label not in clusters:
                clusters[label] = ComponentCluster(
                    cluster_id=int(label),
                    segments=[],
                    shapes=[],
                    text_blocks=[],
                    bbox=(float("inf"), float("inf"), float("-inf"), float("-inf")),
                    center=(0.0, 0.0),
                )
            obj_type, obj_idx = point_to_obj[idx]
            if obj_type == "seg":
                clusters[label].segments.append(segments[obj_idx])
            else:
                clusters[label].shapes.append(shapes[obj_idx])

        # Calcola bbox e center per ogni cluster
        for cluster in clusters.values():
            xs = [s.start[0] for s in cluster.segments] + [s.end[0] for s in cluster.segments]
            ys = [s.start[1] for s in cluster.segments] + [s.end[1] for s in cluster.segments]
            for sh in cluster.shapes:
                xs.extend([sh.bbox[0], sh.bbox[2]])
                ys.extend([sh.bbox[1], sh.bbox[3]])
            if xs and ys:
                cluster.bbox = (min(xs), min(ys), max(xs), max(ys))
                cluster.center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

        logger.info(
            "Clustering complete",
            clusters=len(clusters),
            noise=sum(1 for lbl in labels if lbl == -1),
            eps=eps,
        )
        return list(clusters.values())

    @staticmethod
    def _estimate_eps(x: np.ndarray) -> float:
        """D2: stima epsilon adattivo via k-NN (4° vicino, percentile 90 × 1.5).

        pdist (mediana all-pairs) dava eps enormi su schemi grandi → un unico cluster.
        k-NN usa la densità locale: eps ≈ distanza intra-cluster tipica.
        """
        if len(x) < 2:
            return 10.0
        from sklearn.neighbors import NearestNeighbors
        k = min(4, len(x) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(x)
        dists, _ = nbrs.kneighbors(x)
        kth_dists = dists[:, -1]  # distanza al k-esimo vicino per ogni punto
        eps = float(np.percentile(kth_dists, 90)) * 1.5
        return max(eps, 5.0)  # minimo 5pt
