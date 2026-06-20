from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import structlog

from src.core.pdf_parser import ExtractedPage, PDFSegment
from src.core.text_associator import SymbolAssociation, TextAssociator
from src.ml.classifier import ComponentClassifier, RuleBasedClassifier
from src.ml.clustering import ComponentCluster

logger = structlog.get_logger("graph_builder")


@dataclass
class ComponentNode:
    """Nodo componente nel grafo bipartito."""
    node_id: str
    ref: str
    class_name: str
    value: str | None = None
    cluster: ComponentCluster | None = None
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    pins: list[PinNode] = field(default_factory=list)


@dataclass
class PinNode:
    """Nodo pin (terminale) di un componente."""
    pin_id: str
    pin_name: str | None = None
    position: tuple[float, float] = (0.0, 0.0)
    is_nc: bool = False  # No Connect
    connected_net: str | None = None


@dataclass
class NetNode:
    """Nodo net (rete elettrica) nel grafo bipartito."""
    net_id: str
    name: str | None = None
    segments: list[PDFSegment] = field(default_factory=list)
    is_global: bool = False


class BipartiteGraphBuilder:
    """Costruisce grafo bipartito Componenti ↔ Nets tramite NetworkX.

    Pipeline:
    1. Clusterizza segmenti → ComponentCluster
    2. Classifica ogni cluster → ComponentNode
    3. Associa testo (Ref/Value) → ComponentNode
    4. Trova connessioni cluster-to-net via pin-point matching
    5. Esporta netlist SPICE (.cir) e KiCad (.net)
    """

    def __init__(
        self,
        classifier: ComponentClassifier | None = None,
        text_associator: TextAssociator | None = None,
        stub_length: float = 3.0,  # px
        cluster_eps: float | None = None,
        pin_tol_factor: float = 2.0,  # pin reach ≈ pin_tol_factor × characteristic stub length
        label_tol_factor: float = 6.0,  # label→net reach ≈ factor × wire_tol (labels float off the stub)
    ) -> None:
        self.classifier = classifier or ComponentClassifier()
        self.rule_classifier = RuleBasedClassifier()  # B1: attivo finché ML non è addestrato
        self.text_associator = text_associator or TextAssociator()
        self.stub_length = stub_length
        self.cluster_eps = cluster_eps  # override link_dist del clustering (None = adattivo)
        self.pin_tol_factor = pin_tol_factor
        self.label_tol_factor = label_tol_factor
        self.graph = nx.Graph()
        self.components: dict[str, ComponentNode] = {}
        self.nets: dict[str, NetNode] = {}

    def build_from_page(self, page: ExtractedPage) -> nx.Graph:
        """Costruisce il grafo bipartito da una pagina estratta."""
        from src.ml.clustering import SpatialClusterer

        # Wire/symbol separation: i fili (axis-aligned + lunghi) vanno in wire_segs;
        # i primitivi brevi/diagonali (corpi simbolo) vanno in symbol_segs.
        # DBSCAN vede solo symbol_segs → cluster piccoli e coerenti, nessun cluster gigante.
        symbol_segs, wire_segs = SpatialClusterer.separate_wires(page.segments)

        # 1. Clustering spaziale: SOLO symbol-primitive
        clusterer = SpatialClusterer(eps=self.cluster_eps)
        clusters = clusterer.cluster(symbol_segs, page.shapes, text_blocks=page.text_blocks)

        # Reclaim connecting stubs dropped as clustering noise: short axis-aligned
        # segments that did NOT form a symbol body are wires, not symbols. They were
        # invisible to recover_stub_wires (which only inspects clustered segments),
        # so short nets between close components never formed (measured pin-dangling:
        # ampli_ht 48%, ecc83 23%). Only noise orphans are reclaimed, so dense symbol
        # bodies (which DO cluster) are untouched -> no Bryston over-merge.
        orphan_wires = [
            s
            for s in clusterer.orphan_segments
            if SpatialClusterer._is_axis_aligned(s)
        ]
        wire_segs = wire_segs + orphan_wires

        # 2. Associazione testo
        refs, values, net_labels = self.text_associator.associate(page)
        # D4: dict-of-list per evitare collisioni quando due ref si mappano allo stesso cluster
        ref_map: dict[int, list[SymbolAssociation]] = {}
        val_map: dict[int, list[SymbolAssociation]] = {}
        for r in refs:
            ref_map.setdefault(self._nearest_cluster(r, clusters), []).append(r)
        for v in values:
            val_map.setdefault(self._nearest_cluster(v, clusters), []).append(v)

        # 3. Classifica e crea nodi componente
        for cluster in clusters:
            comp = self._create_component_node(cluster, ref_map, val_map)
            self.components[comp.node_id] = comp
            self.graph.add_node(comp.node_id, bipartite=0, **comp.__dict__)

        # D6: scala da wire_segs reali (o symbol_segs come fallback)
        scale = self._estimate_scale(wire_segs) if wire_segs else self._estimate_scale(symbol_segs)

        # 4. Trova nets: BFS sui wire-candidate pre-separati
        self._build_nets(wire_segs, scale, junctions=page.junction_candidates())

        # 4b. Net merging per label: net che condividono lo stesso nome (GND, +5V,
        # RESET…) sono un'unica net elettrica, come negli schematici si uniscono
        # per nome e non solo per filo. Eseguito PRIMA di connettere i pin.
        label_tol = self._derive_wire_tol(self._all_wire_segs())
        self._merge_nets_by_label(net_labels, tol=self.label_tol_factor * label_tol)

        # 5. Pin-point matching: connetti pin ai nets
        self._connect_pins_to_nets(scale)

        logger.info(
            "Graph built",
            components=len(self.components),
            nets=len(self.nets),
            edges=self.graph.number_of_edges(),
        )
        return self.graph

    def _create_component_node(
        self,
        cluster: ComponentCluster,
        ref_map: dict[int, list[SymbolAssociation]],
        val_map: dict[int, list[SymbolAssociation]],
    ) -> ComponentNode:
        """Crea un ComponentNode da un cluster con classificazione e testo."""
        # B1: ref calcolato prima della classificazione (è il segnale primario)
        ref_candidates = ref_map.get(cluster.cluster_id, [])
        val_candidates = val_map.get(cluster.cluster_id, [])
        ref = min(ref_candidates, key=lambda a: a.distance) if ref_candidates else None
        val = min(val_candidates, key=lambda a: a.distance) if val_candidates else None
        ref_text = ref.text if ref else f"U{cluster.cluster_id}"
        val_text = val.text if val else None

        # Classificazione: ML se addestrato, rule-based altrimenti
        class_name, confidence = "unknown", 0.0
        try:
            class_name, confidence = self.classifier.predict(cluster)
        except RuntimeError:
            class_name, confidence = self.rule_classifier.classify(
                ref.text if ref else None, cluster
            )

        return ComponentNode(
            node_id=ref_text,
            ref=ref_text,
            class_name=class_name,
            value=val_text,
            cluster=cluster,
            confidence=confidence,
            bbox=cluster.bbox,
        )

    @staticmethod
    def _nearest_cluster(
        assoc: SymbolAssociation, clusters: list[ComponentCluster]
    ) -> int:
        """Trova l'indice del cluster più vicino a un'associazione testo."""
        best = -1
        best_dist = float("inf")
        tx, ty = assoc.symbol_center  # D5: usa il centro del simbolo associato, non la pos. del testo
        for cluster in clusters:
            cx, cy = cluster.center
            dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = cluster.cluster_id
        return best


    @staticmethod
    def _derive_wire_tol(wire_segs: list[PDFSegment]) -> float:
        # wire_tol derivation: p25 of min endpoint→segment distance across all wire_segs.
        # Rationale: in real schematics, ~70% of junctions are T-junctions where one wire's endpoint
        # touches ANOTHER wire's SEGMENT (not its endpoint). Measuring endpoint→endpoint misses
        # these connections and inflates wire_tol, causing false net merges. Endpoint→segment
        # captures the true junction geometry.
        if len(wire_segs) < 2:
            return 1.0
        import numpy as np
        min_dists = []
        for i, s1 in enumerate(wire_segs):
            for pt in (s1.start, s1.end):
                best_d2 = float('inf')
                for j, s2 in enumerate(wire_segs):
                    if i == j:
                        continue
                    d2 = BipartiteGraphBuilder._point_to_seg_d2(pt, s2)
                    if d2 < best_d2:
                        best_d2 = d2
                if best_d2 != float('inf'):
                    min_dists.append(math.sqrt(best_d2))
        if not min_dists:
            return 1.0
        p25 = float(np.percentile(min_dists, 25))
        if p25 < 0.1:
            return 1.0
        return p25

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
        return max(float(np.median(dists)), 1.0)

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
    def _point_to_seg_d2(pt: tuple[float, float], seg: PDFSegment) -> float:
        px, py = pt
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
                d2 = self._point_to_seg_d2((px, py), s)
                if d2 <= best_d2:
                    best_d2 = d2
                    best_net = n.net_id
        return best_net

    def _build_nets(self, wire_segs: list[PDFSegment], scale: float, junctions: list[Any] | None = None) -> None:
        if not wire_segs:
            return

        junctions = junctions or []
        for j in junctions:
            bbox = j.bbox
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            j_seg = PDFSegment(start=(cx, cy), end=(cx, cy), item_type="junction")
            wire_segs.append(j_seg)

        from src.ml.clustering import SpatialClusterer
        clusters = [comp.cluster for comp in self.components.values() if comp.cluster]
        SpatialClusterer.recover_stub_wires(clusters, wire_segs)

        # Remove junctions from the wire_segs passed to derive_wire_tol, as zero-length segments skew the calculation
        real_wires = [s for s in wire_segs if s.item_type != "junction"]
        wire_tol = self._derive_wire_tol(real_wires)
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
    def _estimate_scale(segments: list[PDFSegment]) -> float:
        """Stima la scala caratteristica dalla distribuzione delle lunghezze dei segmenti.

        Usa il 10° percentile per catturare la lunghezza tipica dei segmenti corti
        (stub di connessione pin-filo) senza essere distorto dai segmenti lunghi.
        Si adatta automaticamente alle coordinate PDF reali senza costanti magiche.
        """
        lengths = sorted(s.length for s in segments if s.length > 0.1)
        if not lengths:
            return 1.0
        idx = max(0, int(len(lengths) * 0.10) - 1)
        return max(1.0, lengths[idx])

    def _segments_touch(self, a: PDFSegment, b: PDFSegment, tol: float = 1.0) -> bool:
        """Verifica se due segmenti si toccano (estremità vicine o T-junction).

        Two wires "touch" if any of these is within wire_tol:
        (a) endpoint-endpoint distance (L-junction, shared endpoint)
        (b) endpoint-segment distance A→B (T-junction, A terminates on B)
        (c) endpoint-segment distance B→A (T-junction, B terminates on A)
        Note: pure endpoint-endpoint (old behavior) misses ALL T-junctions, which are
        ~70% of real schematic junctions. This is why pre-V5 had grado 3+ = 0.
        """
        ends_a = [a.start, a.end]
        ends_b = [b.start, b.end]

        # (a) Endpoint-endpoint distance
        for ea in ends_a:
            for eb in ends_b:
                if ((ea[0] - eb[0]) ** 2 + (ea[1] - eb[1]) ** 2) ** 0.5 <= tol + 1e-3:
                    return True

        # (b) Endpoint-segment distance A -> B
        for ea in ends_a:
            if BipartiteGraphBuilder._point_to_seg_d2(ea, b) <= (tol + 1e-3)**2:
                return True

        # (c) Endpoint-segment distance B -> A
        for eb in ends_b:
            if BipartiteGraphBuilder._point_to_seg_d2(eb, a) <= (tol + 1e-3)**2:
                return True

        return False

    def _connect_pins_to_nets(self, scale: float) -> None:
        all_wires = self._all_wire_segs()
        wire_tol = self._derive_wire_tol(all_wires)
        # Pin reach is on the scale of a connection stub (one characteristic
        # segment length), NOT the wire-endpoint merge spacing: real pin→net gaps
        # are ~1-2 stub lengths, while wire_tol (p25 endpoint spacing) is far
        # smaller and connects almost nothing. Use the data-derived `scale`.
        pin_tol = max(3.0 * wire_tol, scale * self.pin_tol_factor)

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

    def _merge_nets_by_label(
        self, net_labels: list[SymbolAssociation], tol: float
    ) -> None:
        """Fonde le net che condividono lo stesso nome di label di rete.

        Negli schematici i punti etichettati con lo stesso nome (GND, +5V, VCC,
        RESET…) appartengono alla stessa net anche senza filo continuo — è così
        che la ground-truth KiCad unisce le net. Le label puramente numeriche
        sono numeri di pin, non nomi di rete, e vengono ignorate.
        """
        from collections import defaultdict

        name_to_nets: dict[str, set[str]] = defaultdict(set)
        for lbl in net_labels:
            name = lbl.text.strip()
            if not name or name.isdigit():
                continue
            nid = self._find_nearest_net(lbl.symbol_center[0], lbl.symbol_center[1], tol)
            if nid:
                name_to_nets[name].add(nid)

        for name, nids in name_to_nets.items():
            canonical = self._union_nets(nids)
            if canonical is None:
                continue
            self.nets[canonical].name = name
            if self.graph.has_node(canonical):
                self.graph.nodes[canonical]["name"] = name

    def _union_nets(self, net_ids: set[str]) -> str | None:
        """Unisce più net in una canonica: sposta i segmenti, elimina le altre
        dal grafo e da self.nets. Ritorna l'id canonico (None se vuoto)."""
        present = [nid for nid in net_ids if nid in self.nets]
        if not present:
            return None
        canonical = present[0]
        canon = self.nets[canonical]
        for nid in present[1:]:
            canon.segments.extend(self.nets[nid].segments)
            del self.nets[nid]
            if self.graph.has_node(nid):
                self.graph.remove_node(nid)
        if self.graph.has_node(canonical):
            self.graph.nodes[canonical]["segments"] = canon.segments
        return canonical

    @staticmethod
    def _segments_intersect(a: PDFSegment, b: PDFSegment) -> bool:
        """Verifica intersezione tra due segmenti (AABB + orientamento)."""
        # Bounding box overlap
        ax_min, ax_max = min(a.start[0], a.end[0]), max(a.start[0], a.end[0])
        ay_min, ay_max = min(a.start[1], a.end[1]), max(a.start[1], a.end[1])
        bx_min, bx_max = min(b.start[0], b.end[0]), max(b.start[0], b.end[0])
        by_min, by_max = min(b.start[1], b.end[1]), max(b.start[1], b.end[1])

        if ax_max < bx_min or bx_max < ax_min or ay_max < by_min or by_max < ay_min:
            return False

        # Orientamento cross-product
        def cross(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        c1 = float(cross(a.start[0], a.start[1], a.end[0], a.end[1], b.start[0], b.start[1]))
        c2 = float(cross(a.start[0], a.start[1], a.end[0], a.end[1], b.end[0], b.end[1]))
        c3 = float(cross(b.start[0], b.start[1], b.end[0], b.end[1], a.start[0], a.start[1]))
        c4 = float(cross(b.start[0], b.start[1], b.end[0], b.end[1], a.end[0], a.end[1]))

        # D6: <= 0 invece di < 0 per rilevare T-junction (un cross-product = 0 = endpoint su segmento).
        return bool((c1 * c2 <= 0) and (c3 * c4 <= 0))

    # ===================== EXPORT =====================

    def export_spice(self, output_path: Path | str) -> None:
        """Esporta netlist SPICE strutturale (.cir) senza modelli."""
        path = Path(output_path)
        lines = ["* Schematic AI Reasoner - SPICE Structural Netlist", ""]

        for comp in self.components.values():
            pins_str = " ".join(p.connected_net or p.pin_id for p in comp.pins)
            lines.append(f"{comp.ref} {pins_str} {comp.class_name}")

        lines.append("")
        lines.append(".end")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("SPICE exported", path=str(path))

    def export_kicad_netlist(self, output_path: Path | str) -> None:
        """Esporta netlist in formato KiCad (.net)."""
        path = Path(output_path)
        lines = [
            "(export (version D)",
            "  (design",
            "    (source \"extracted.pdf\")",
            "  )",
            "  (components",
        ]

        for comp in self.components.values():
            lines.append(f'    (comp (ref "{comp.ref}")')
            lines.append(f'      (value "{comp.value or comp.class_name}")')
            lines.append("    )")

        lines.append("  )")
        lines.append("  (nets")

        for i, net in enumerate(self.nets.values(), 1):
            lines.append(f'    (net (code "{i}") (name "{net.name or net.net_id}")')
            for comp in self.components.values():
                for pin in comp.pins:
                    if pin.connected_net == net.net_id:
                        lines.append(f'      (node (ref "{comp.ref}") (pin "{pin.pin_name}"))')
            lines.append("    )")

        lines.append("  )")
        lines.append(")")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("KiCad netlist exported", path=str(path))

    def export_json(self, output_path: Path | str) -> dict[str, Any]:
        """Esporta grafo in formato JSON strutturato."""
        path = Path(output_path)
        data = {
            "components": [
                {
                    "ref": c.ref,
                    "class": c.class_name,
                    "value": c.value,
                    "confidence": c.confidence,
                    "bbox": c.bbox,
                    "pins": [
                        {
                            "pin_id": p.pin_id,
                            "pin_name": p.pin_name,
                            "position": p.position,
                            "net": p.connected_net,
                            "nc": p.is_nc,
                        }
                        for p in c.pins
                    ],
                }
                for c in self.components.values()
            ],
            "nets": [
                {
                    "net_id": n.net_id,
                    "name": n.name,
                    "is_global": n.is_global,
                    "segments": len(n.segments),
                }
                for n in self.nets.values()
            ],
            "edges": [
                {"comp": u, "net": v}
                for u, v in self.graph.edges()
                if u in self.components and v in self.nets
            ],
        }
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("JSON exported", path=str(path))
        return data
