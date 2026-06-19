from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull

from src.ml.clustering import ComponentCluster


@dataclass
class FeatureVector:
    """Feature vector di 13 dimensioni per classificazione componenti."""
    aspect_ratio: float          # 1  bbox w/h
    num_segments: float          # 2  normalizzato
    bb_area: float               # 3  bounding box area
    convex_hull_area: float      # 4  area convex hull
    solidity: float              # 5  convex_hull_area / bb_area
    total_segment_length: float  # 6  somma lunghezze segmenti
    num_shapes: float            # 7  numero di forme
    shape_area_ratio: float      # 8  area forme / bb_area
    centroid_x: float            # 9  posizione normalizzata
    centroid_y: float          # 10 posizione normalizzata
    std_x: float                 # 11 deviazione std x segmenti
    std_y: float                 # 12 deviazione std y segmenti
    max_segment_length: float   # 13 lunghezza max segmento

    def to_array(self) -> npt.NDArray[np.float64]:
        return np.array([
            self.aspect_ratio,
            self.num_segments,
            self.bb_area,
            self.convex_hull_area,
            self.solidity,
            self.total_segment_length,
            self.num_shapes,
            self.shape_area_ratio,
            self.centroid_x,
            self.centroid_y,
            self.std_x,
            self.std_y,
            self.max_segment_length,
        ], dtype=np.float64)


class FeatureExtractor:
    """Estrae il feature vector 13D da un ComponentCluster per Random Forest."""

    def extract(self, cluster: ComponentCluster) -> FeatureVector:
        bbox = cluster.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect_ratio = w / h if h > 0 else 1.0
        bb_area = w * h

        # Convex hull dai punti dei segmenti
        points = self._cluster_points(cluster)
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                convex_hull_area = hull.volume  # in 2D volume = area
            except Exception:
                convex_hull_area = bb_area
        else:
            convex_hull_area = bb_area

        solidity = convex_hull_area / bb_area if bb_area > 0 else 1.0
        total_len = sum(s.length for s in cluster.segments)
        max_len = max((s.length for s in cluster.segments), default=0.0)
        num_shapes = len(cluster.shapes)
        shape_area = sum(s.area for s in cluster.shapes)
        shape_area_ratio = shape_area / bb_area if bb_area > 0 else 0.0

        # Statistiche posizione
        all_x = [p[0] for p in points]
        all_y = [p[1] for p in points]
        centroid_x = float(np.mean(all_x) if all_x else 0.0)
        centroid_y = float(np.mean(all_y) if all_y else 0.0)
        std_x = float(np.std(all_x) if len(all_x) > 1 else 0.0)
        std_y = float(np.std(all_y) if len(all_y) > 1 else 0.0)

        # Normalizzazione per stabilità (usare log per aree/lunghezze)
        return FeatureVector(
            aspect_ratio=aspect_ratio,
            num_segments=math.log1p(len(cluster.segments)),
            bb_area=math.log1p(bb_area),
            convex_hull_area=math.log1p(convex_hull_area),
            solidity=solidity,
            total_segment_length=math.log1p(total_len),
            num_shapes=float(num_shapes),
            shape_area_ratio=shape_area_ratio,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            std_x=std_x,
            std_y=std_y,
            max_segment_length=math.log1p(max_len),
        )

    @staticmethod
    def _cluster_points(cluster: ComponentCluster) -> list[list[float]]:
        points: list[list[float]] = []
        for seg in cluster.segments:
            points.append([seg.start[0], seg.start[1]])
            points.append([seg.end[0], seg.end[1]])
        for sh in cluster.shapes:
            for v in sh.vertices:
                points.append([v[0], v[1]])
        return points
