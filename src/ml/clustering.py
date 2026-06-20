from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import structlog

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
    """Clustering spaziale di segmenti e forme PDF.

    Single-linkage sulla prossimità degli ENDPOINT dei segmenti (union-find), non
    DBSCAN sui midpoint: con un eps globale i midpoint dei tratti dei simboli
    (densamente impacchettati) si incatenano in un unico blob a livello di pagina.
    Clusterizzare sugli endpoint condivisi rispecchia come un simbolo è disegnato
    — tratti che si toccano — e separa simboli distinti (i fili che li
    collegherebbero sono gia rimossi da `separate_wires`).
    """

    def __init__(self, eps: float | None = None, min_samples: int = 2) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, segments: list[PDFSegment], shapes: list[PDFShape], text_blocks: list[PDFTextBlock] | None = None) -> list[ComponentCluster]:
        """Clusterizza segmenti e forme in componenti candidate.

        `link_dist`: `self.eps` se fornito esplicitamente (override / test),
        altrimenti adattivo data-derived (p60 della distanza al piu vicino endpoint
        di un ALTRO segmento). Le shapes vengono assegnate al gruppo-segmenti il cui
        endpoint piu vicino dista <= link_dist dal centro shape; le shapes orfane
        (es. cornici pagina lontane da ogni tratto) sono scartate.
        Gruppi con meno di `min_samples` membri (segmenti + shapes) sono noise.
        """
        if not segments and not shapes:
            return []

        link_dist = self.eps if self.eps is not None else self._estimate_link_dist(segments)
        groups = self._link_segments(segments, link_dist)
        seg_to_group = {seg_idx: gid for gid, idxs in enumerate(groups) for seg_idx in idxs}

        # Assegna ogni shape al gruppo-segmenti con l'endpoint piu vicino al centro shape.
        shape_assign: dict[int, list[int]] = defaultdict(list)
        for sh_idx, shape in enumerate(shapes):
            cx = (shape.bbox[0] + shape.bbox[2]) / 2
            cy = (shape.bbox[1] + shape.bbox[3]) / 2
            best_gid, best_d2 = -1, link_dist * link_dist
            for seg_idx, seg in enumerate(segments):
                for px, py in (seg.start, seg.end):
                    d2 = (cx - px) ** 2 + (cy - py) ** 2
                    if d2 <= best_d2:
                        best_d2, best_gid = d2, seg_to_group[seg_idx]
            if best_gid != -1:
                shape_assign[best_gid].append(sh_idx)

        clusters: list[ComponentCluster] = []
        for gid, seg_idxs in enumerate(groups):
            sh_idxs = shape_assign.get(gid, [])
            if len(seg_idxs) + len(sh_idxs) < self.min_samples:
                continue  # noise
            cl_segs = [segments[i] for i in seg_idxs]
            cl_shapes = [shapes[i] for i in sh_idxs]
            xs = [s.start[0] for s in cl_segs] + [s.end[0] for s in cl_segs]
            ys = [s.start[1] for s in cl_segs] + [s.end[1] for s in cl_segs]
            for sh in cl_shapes:
                xs.extend([sh.bbox[0], sh.bbox[2]])
                ys.extend([sh.bbox[1], sh.bbox[3]])
            bbox = (min(xs), min(ys), max(xs), max(ys))
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            clusters.append(
                ComponentCluster(
                    cluster_id=len(clusters),
                    segments=cl_segs,
                    shapes=cl_shapes,
                    text_blocks=[],
                    bbox=bbox,
                    center=center,
                )
            )

        clusters = self._merge_noise_clusters(clusters, link_dist)
        if text_blocks:
            clusters = self._text_guided_merge(clusters, text_blocks, link_dist)

        logger.info(
            "Clustering complete",
            clusters=len(clusters),
            raw_groups=len(groups),
            link_dist=link_dist,
        )
        return clusters

    @staticmethod
    def _text_guided_merge(clusters: list[ComponentCluster], text_blocks: list[PDFTextBlock], link_dist: float) -> list[ComponentCluster]:
        import re
        REF_PATTERN = re.compile(r"^[A-Za-z]+\d+[A-Za-z]*$")
        refs = [t for t in text_blocks if REF_PATTERN.match(t.text.strip())]
        
        if not refs or not clusters:
            return clusters
            
        n = len(clusters)
        parent = list(range(n))
        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for ref in refs:
            x0, y0, x1, y1 = ref.bbox
            w = max(x1 - x0, 1.0)
            h = max(y1 - y0, 1.0)
            # Expand gravity zone reasonably
            pad_x = max(20.0, w * 1.5)
            pad_y = max(20.0, h * 1.5)
            gz = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
            
            intersecting_idx = []
            for i, c in enumerate(clusters):
                if not (c.bbox[2] < gz[0] or c.bbox[0] > gz[2] or c.bbox[3] < gz[1] or c.bbox[1] > gz[3]):
                    intersecting_idx.append(i)
            
            # Prevent merging multiple LARGE clusters (which means we are merging distinct symbols)
            # A typical symbol has < 30 segments. If we find more than one cluster with > 5 segments,
            # we might be merging distinct components.
            large_clusters = sum(1 for i in intersecting_idx if clusters[i].num_segments > 5)
            
            if large_clusters <= 1:
                # Safe to merge: at most one large cluster, the rest are small fragments
                for i in range(1, len(intersecting_idx)):
                    union(intersecting_idx[0], intersecting_idx[i])
                
        merged_groups = defaultdict(list)
        for i in range(n):
            merged_groups[find(i)].append(i)
            
        merged_clusters = []
        for gid, indices in merged_groups.items():
            if len(indices) == 1:
                merged_clusters.append(clusters[indices[0]])
            else:
                new_segs = []
                new_shapes = []
                new_texts = []
                for idx in indices:
                    c = clusters[idx]
                    new_segs.extend(c.segments)
                    new_shapes.extend(c.shapes)
                    new_texts.extend(c.text_blocks)
                
                xs = [s.start[0] for s in new_segs] + [s.end[0] for s in new_segs]
                ys = [s.start[1] for s in new_segs] + [s.end[1] for s in new_segs]
                for sh in new_shapes:
                    xs.extend([sh.bbox[0], sh.bbox[2]])
                    ys.extend([sh.bbox[1], sh.bbox[3]])
                new_bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else (0,0,0,0)
                new_center = ((new_bbox[0] + new_bbox[2]) / 2, (new_bbox[1] + new_bbox[3]) / 2) if xs else (0,0)
                
                merged_clusters.append(
                    ComponentCluster(
                        cluster_id=0,
                        segments=new_segs,
                        shapes=new_shapes,
                        text_blocks=new_texts,
                        bbox=new_bbox,
                        center=new_center
                    )
                )
                
        for i, c in enumerate(merged_clusters):
            c.cluster_id = i
            
        return merged_clusters

    @staticmethod
    def _merge_noise_clusters(clusters: list[ComponentCluster], link_dist: float) -> list[ComponentCluster]:
        """Micro-clusters of 1-2 segments that share an endpoint within link_dist are likely fragments of the same symbol stroke, not separate components. Merge them to reduce over-segmentation."""
        if not clusters:
            return []
            
        # Calculate p90 of cluster sizes (number of segments) to avoid creating giant clusters
        sizes = [c.num_segments for c in clusters]
        p90_size = np.percentile(sizes, 90) if sizes else 0
        max_size = max(5, p90_size)
        
        merged = []
        skip = set()
        
        # Simple iterative merge (could be optimized with spatial index if slow)
        for i, c1 in enumerate(clusters):
            if i in skip:
                continue
            
            if c1.num_segments > 2:
                merged.append(c1)
                continue
                
            current_cluster = c1
            merged_this_round = True
            while merged_this_round:
                merged_this_round = False
                for j in range(i + 1, len(clusters)):
                    if j in skip:
                        continue
                    c2 = clusters[j]
                    if c2.num_segments > 2:
                        continue
                        
                    if current_cluster.num_segments + c2.num_segments > max_size:
                        continue
                        
                    # Check bbox distance
                    bb1 = current_cluster.bbox
                    bb2 = c2.bbox
                    # Distance between bboxes
                    dx = max(0, max(bb1[0] - bb2[2], bb2[0] - bb1[2]))
                    dy = max(0, max(bb1[1] - bb2[3], bb2[1] - bb1[3]))
                    if math.hypot(dx, dy) > 1.5 * link_dist:
                        continue
                        
                    # Check shared endpoint
                    ep1 = SpatialClusterer.free_endpoints(current_cluster.segments)
                    ep2 = SpatialClusterer.free_endpoints(c2.segments)
                    shared = False
                    for p1 in ep1:
                        for p2 in ep2:
                            if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= link_dist:
                                shared = True
                                break
                        if shared:
                            break
                            
                    if shared:
                        # Merge c2 into current_cluster
                        new_segs = current_cluster.segments + c2.segments
                        new_shapes = current_cluster.shapes + c2.shapes
                        xs = [s.start[0] for s in new_segs] + [s.end[0] for s in new_segs]
                        ys = [s.start[1] for s in new_segs] + [s.end[1] for s in new_segs]
                        for sh in new_shapes:
                            xs.extend([sh.bbox[0], sh.bbox[2]])
                            ys.extend([sh.bbox[1], sh.bbox[3]])
                        new_bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else (0,0,0,0)
                        new_center = ((new_bbox[0] + new_bbox[2]) / 2, (new_bbox[1] + new_bbox[3]) / 2) if xs else (0,0)
                        current_cluster = ComponentCluster(
                            cluster_id=current_cluster.cluster_id, # keep first id
                            segments=new_segs,
                            shapes=new_shapes,
                            text_blocks=current_cluster.text_blocks + c2.text_blocks,
                            bbox=new_bbox,
                            center=new_center
                        )
                        skip.add(j)
                        merged_this_round = True
                        # Need to restart inner loop since current_cluster changed, but a simple greedy is fine
                        break 
                        
            merged.append(current_cluster)
            
        # Reassign cluster IDs
        for i, c in enumerate(merged):
            c.cluster_id = i
            
        return merged

    @staticmethod
    def _link_segments(segments: list[PDFSegment], link_dist: float) -> list[list[int]]:
        """Single-linkage union-find: unisce segmenti con endpoint entro link_dist.

        Usa una griglia spaziale (cella = link_dist) per query dei vicini in ~O(n).
        Restituisce la lista dei gruppi (ognuno = lista di indici di segmento).
        """
        n = len(segments)
        if n == 0:
            return []
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        cell = max(link_dist, 1e-6)
        grid: dict[tuple[int, int], list[tuple[int, float, float]]] = defaultdict(list)
        for i, seg in enumerate(segments):
            for x, y in (seg.start, seg.end):
                grid[(int(x // cell), int(y // cell))].append((i, x, y))

        d2_max = link_dist * link_dist
        for i, seg in enumerate(segments):
            for x, y in (seg.start, seg.end):
                gx, gy = int(x // cell), int(y // cell)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for j, xx, yy in grid[(gx + dx, gy + dy)]:
                            if j > i and (x - xx) ** 2 + (y - yy) ** 2 <= d2_max:
                                union(i, j)

        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        return list(groups.values())

    @staticmethod
    def _estimate_link_dist(segments: list[PDFSegment]) -> float:
        """Stima adattiva della distanza di linkage (data-derived, niente costanti assolute).

        p60 della distanza dal piu vicino endpoint appartenente a un ALTRO segmento:
        cattura il gap intra-simbolo tra tratti, restando sotto la spaziatura
        inter-componente. Su scala PDF reale (Bryston) ~ 8pt.
        """
        if len(segments) < 2:
            return 10.0
        from sklearn.neighbors import NearestNeighbors

        pts: list[tuple[float, float]] = []
        owner: list[int] = []
        for i, seg in enumerate(segments):
            pts.append(seg.start)
            owner.append(i)
            pts.append(seg.end)
            owner.append(i)
        x = np.array(pts)
        k = min(6, len(pts))
        nbrs = NearestNeighbors(n_neighbors=k).fit(x)
        dists, idxs = nbrs.kneighbors(x)
        nn_other: list[float] = []
        for r in range(len(pts)):
            for c in range(1, k):
                if owner[idxs[r, c]] != owner[r]:
                    nn_other.append(float(dists[r, c]))
                    break
        if not nn_other:
            return 10.0
        return max(float(np.percentile(nn_other, 60)), 5.0)

    @staticmethod
    def _is_axis_aligned(seg: PDFSegment, tol_deg: float = 5.0) -> bool:
        """Vero se il segmento e orizzontale o verticale entro tol_deg gradi.

        Nei PDF schematics EDA, i fili sono sempre asse-allineati; le linee
        diagonali o curve appartengono ai corpi dei simboli componente.
        """
        dx = abs(seg.end[0] - seg.start[0])
        dy = abs(seg.end[1] - seg.start[1])
        if dx + dy < 0.1:
            return False
        angle = math.degrees(math.atan2(dy, dx))  # [0, 90]
        return angle <= tol_deg or angle >= (90.0 - tol_deg)

    @staticmethod
    def free_endpoints(segments: list[PDFSegment]) -> list[tuple[float, float]]:
        counts: dict[tuple[float, float], int] = defaultdict(int)
        for s in segments:
            counts[s.start] += 1
            counts[s.end] += 1
        return [pt for pt, c in counts.items() if c == 1]

    @staticmethod
    def recover_stub_wires(
        clusters: list[ComponentCluster],
        wire_segs: list[PDFSegment],
        min_samples: int = 2,
    ) -> list[PDFSegment]:
        recovered: list[PDFSegment] = []
        for cluster in clusters:
            free = set(SpatialClusterer.free_endpoints(cluster.segments))
            if not free:
                continue
            to_remove = []
            for s in cluster.segments:
                if SpatialClusterer._is_axis_aligned(s) and (s.start in free or s.end in free):
                    to_remove.append(s)

            if len(cluster.segments) - len(to_remove) + len(cluster.shapes) >= min_samples:
                for s in to_remove:
                    cluster.segments.remove(s)
                    recovered.append(s)
        wire_segs.extend(recovered)
        print(f"DEBUG: recover_stub_wires recovered {len(recovered)} segments")
        return recovered

    @staticmethod
    def separate_wires(
        segments: list[PDFSegment],
        factor: float = 3.0,
    ) -> tuple[list[PDFSegment], list[PDFSegment]]:
        """Separa wire-candidate da symbol-primitive prima del clustering.

        Passo 1 - filtro curva: i segmenti item_type='curve' sono approssimazioni
        lineari di archi Bezier nei corpi dei simboli (spirali induttori, archi
        transistor). Non contribuiscono alla connettivita e, con eps~34pt, formano
        catene dense che creano cluster giganti. Vengono esclusi da entrambe le
        liste output.

        Passo 2 - wire vs symbol: sui soli segmenti 'line' rimanenti:
          Wire: asse-allineato (<=5) AND lunghezza >= factor x p25(lunghezze).
          Il fattore 3x separa fili lunghi dagli stroke corti dei simboli componente
          senza costanti assolute (scala automaticamente con le coordinate PDF).

        Restituisce:
            (symbol_segs, wire_segs)
            - curve omesse da entrambi, disponibili come page.segments per feature
              extraction futura se necessario.

        Garantisce comportamento invariato su test sintetici con item_type='line'
        e segmenti tutti corti (threshold alta -> tutti in symbol_segs).
        """
        if not segments:
            return [], []

        # Passo 1: rimuovi frammenti Bezier (non contribuiscono a topologia ne a DBSCAN)
        non_curve = [s for s in segments if s.item_type != "curve"]
        if not non_curve:
            return [], []

        lengths = [s.length for s in non_curve if s.length > 0.1]
        if not lengths:
            return non_curve, []

        p25 = float(np.percentile(lengths, 25))
        threshold = max(5.0, p25 * factor)

        wire_segs: list[PDFSegment] = []
        symbol_segs: list[PDFSegment] = []
        for seg in non_curve:
            if SpatialClusterer._is_axis_aligned(seg) and seg.length >= threshold:
                wire_segs.append(seg)
            else:
                symbol_segs.append(seg)
        return symbol_segs, wire_segs

    @staticmethod
    def _estimate_eps(x: npt.NDArray[np.float64]) -> float:
        """Stima epsilon adattivo via k-NN (4 vicino, percentile 90 x 1.5).

        Mantenuto per retro-compatibilita (test) e come euristica di scala; il
        clustering principale usa ora il single-linkage su endpoint.
        """
        if len(x) < 2:
            return 10.0
        from sklearn.neighbors import NearestNeighbors
        k = min(4, len(x) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(x)
        dists, _ = nbrs.kneighbors(x)
        kth_dists = dists[:, -1]
        eps = float(np.percentile(kth_dists, 90)) * 1.5
        return max(eps, 5.0)
