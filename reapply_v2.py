
def update_clustering():
    with open("src/ml/clustering.py") as f:
        lines = f.readlines()

    insert_idx = -1
    for i, line in enumerate(lines):
        if "def separate_wires(" in line:
            insert_idx = i - 1  # @staticmethod is on the line before
            break

    code = """    @staticmethod
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
                if SpatialClusterer._is_axis_aligned(s):
                    if s.start in free or s.end in free:
                        to_remove.append(s)

            if len(cluster.segments) - len(to_remove) + len(cluster.shapes) >= min_samples:
                for s in to_remove:
                    cluster.segments.remove(s)
                    recovered.append(s)
        wire_segs.extend(recovered)
        return recovered

"""
    lines.insert(insert_idx, code)
    with open("src/ml/clustering.py", "w") as f:
        f.writelines(lines)

def update_graph_builder():
    with open("src/core/graph_builder.py") as f:
        content = f.read()

    new_methods = """
    @staticmethod
    def _derive_wire_tol(wire_segs: list[PDFSegment]) -> float:
        if len(wire_segs) < 2:
            return 1.0
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        pts = []
        owner = []
        for i, s in enumerate(wire_segs):
            pts.extend([s.start, s.end])
            owner.extend([i, i])
        x = np.array(pts)
        k = min(6, len(pts))
        nbrs = NearestNeighbors(n_neighbors=k).fit(x)
        dists, idxs = nbrs.kneighbors(x)
        nn_other = []
        for r in range(len(pts)):
            for c in range(1, k):
                if owner[idxs[r, c]] != owner[r]:
                    nn_other.append(float(dists[r, c]))
                    break
        if not nn_other:
            return 1.0
        return max(float(np.percentile(nn_other, 25)), 1.0)

    @staticmethod
    def _derive_pin_border_tol(cluster: ComponentCluster) -> float:
        from src.ml.clustering import SpatialClusterer
        free = SpatialClusterer.free_endpoints(cluster.segments)
        if not free:
            return 1.0
        x0, y0, x1, y1 = cluster.bbox
        import numpy as np
        dists = []
        for px, py in free:
            dl = px - x0
            dr = x1 - px
            dt = py - y0
            db = y1 - py
            dists.append(min(dl, dr, dt, db))
        return float(np.median(dists))

    @staticmethod
    def select_pins(cluster: ComponentCluster) -> list[tuple[float, float]]:
        from src.ml.clustering import SpatialClusterer
        free = SpatialClusterer.free_endpoints(cluster.segments)
        if not free:
            return []
        tol = BipartiteGraphBuilder._derive_pin_border_tol(cluster)
        x0, y0, x1, y1 = cluster.bbox
        pins = []
        for px, py in free:
            dl = px - x0
            dr = x1 - px
            dt = py - y0
            db = y1 - py
            if min(dl, dr, dt, db) <= tol + 1e-3:
                pins.append((px, py))
        return pins

    def _all_wire_segs(self) -> list[PDFSegment]:
        segs = []
        for n in self.nets.values():
            segs.extend(n.segments)
        return segs

    @staticmethod
    def _point_to_seg_d2(px: float, py: float, seg: PDFSegment) -> float:
        x0, y0 = seg.start
        x1, y1 = seg.end
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            return (px - x0)**2 + (py - y0)**2
        t = ((px - x0) * dx + (py - y0) * dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        qx = x0 + t * dx
        qy = y0 + t * dy
        return (px - qx)**2 + (py - qy)**2

    def _find_nearest_net(self, px: float, py: float, pin_tol: float) -> str | None:
        best_net = None
        best_d2 = pin_tol * pin_tol
        for n in self.nets.values():
            for s in n.segments:
                d2 = self._point_to_seg_d2(px, py, s)
                if d2 <= best_d2:
                    best_d2 = d2
                    best_net = n.net_id
        return best_net
"""

    import re

    # Replace _build_nets
    content = re.sub(
        r"    def _build_nets\(self, wire_segs: list\[PDFSegment\], scale: float\) -> None:.*?    @staticmethod\n    def _estimate_scale",
        """    def _build_nets(self, wire_segs: list[PDFSegment], scale: float) -> None:
        if not wire_segs:
            return

        from src.ml.clustering import SpatialClusterer
        clusters = [comp.cluster for comp in self.components.values() if comp.cluster]
        SpatialClusterer.recover_stub_wires(clusters, wire_segs)

        wire_tol = self._derive_wire_tol(wire_segs)
        visited: set[int] = set()
        net_counter = 0

        for seg in wire_segs:
            if id(seg) in visited:
                continue
            net_counter += 1
            net = NetNode(net_id=f"N{net_counter}", name=f"Net-{net_counter}")
            stack = [seg]
            while stack:
                current = stack.pop()
                if id(current) in visited:
                    continue
                visited.add(id(current))
                net.segments.append(current)
                for other in wire_segs:
                    if id(other) not in visited and self._segments_touch(current, other, tol=wire_tol):
                        stack.append(other)

            self.nets[net.net_id] = net
            self.graph.add_node(net.net_id, bipartite=1, **net.__dict__)

    @staticmethod
    def _estimate_scale""",
        content,
        flags=re.DOTALL
    )

    # Replace _connect_pins_to_nets and the old _find_nearest_net
    # Find _connect_pins_to_nets
    idx_connect = content.find("    def _connect_pins_to_nets(")
    idx_segments_intersect = content.find("    @staticmethod\n    def _segments_intersect(")

    if idx_connect != -1 and idx_segments_intersect != -1:
        new_connect = """    def _connect_pins_to_nets(self, scale: float) -> None:
        all_wires = self._all_wire_segs()
        wire_tol = self._derive_wire_tol(all_wires)
        pin_tol = 3.0 * wire_tol

        for comp in self.components.values():
            if comp.cluster is None:
                continue

            pins = self.select_pins(comp.cluster)
            for i, (px, py) in enumerate(pins):
                pin_node = PinNode(pin_id=f"{comp.node_id}_{i+1}", position=(px, py))
                net_id = self._find_nearest_net(px, py, pin_tol)
                pin_node.connected_net = net_id
                comp.pins.append(pin_node)

                if net_id:
                    self.graph.add_edge(comp.node_id, net_id, pin_id=pin_node.pin_id)

"""
        content = content[:idx_connect] + new_connect + "\n" + content[idx_segments_intersect:]

    # Prepend new_methods before _build_nets
    idx_build = content.find("    def _build_nets(")
    content = content[:idx_build] + new_methods + "\n" + content[idx_build:]

    with open("src/core/graph_builder.py", "w") as f:
        f.write(content)

update_clustering()
update_graph_builder()
print("Done")
