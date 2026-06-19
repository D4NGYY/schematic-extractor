from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.core.coordinate_system import CoordinateSystem, Vec2

logger = structlog.get_logger("kicad_net_reconstructor")


@dataclass
class WireSegment:
    """Singolo segmento di filo (wire) in uno schema KiCad."""

    start: Vec2
    end: Vec2
    net_id: int | None = None

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    def is_horizontal(self, tol: float = 1e-6) -> bool:
        return abs(self.start.y - self.end.y) < tol

    def is_vertical(self, tol: float = 1e-6) -> bool:
        return abs(self.start.x - self.end.x) < tol

    def merge_with(self, other: WireSegment, tolerance: float = 0.01) -> WireSegment | None:
        """Unisce due segmenti collineari e sovrapposti in un unico segmento.

        Ritorna il segmento unito se merge possibile, altrimenti None.
        Tolerance in mm (default 0.01 = 10 micron).
        """
        if not self.is_collinear_with(other, tolerance):
            return None

        points = [self.start, self.end, other.start, other.end]

        # Se il segmento è orizzontale, ordina per x; se verticale, per y
        if self.is_horizontal(tolerance):
            points.sort(key=lambda p: p.x)
        else:
            points.sort(key=lambda p: p.y)

        # Criterio di sovrapposizione: la distanza tra gli estremi più lontani
        # deve essere <= alla somma delle lunghezze dei due segmenti + tolerance.
        # Se non si sovrappongono, il punto interno del primo è lontano dal
        # punto interno del secondo.
        span = points[0].distance_to(points[-1])
        len_self = self.length
        len_other = other.length
        if span > len_self + len_other + tolerance:
            return None

        return WireSegment(start=points[0], end=points[-1], net_id=self.net_id or other.net_id)

    def is_collinear_with(self, other: WireSegment, tolerance: float = 1e-6) -> bool:
        """Verifica collinearità (stessa linea retta, anche orientamento opposto)."""

        # Due segmenti sono collineari se i 4 punti giacciono sulla stessa retta
        def cross_2d(a: Vec2, b: Vec2, c: Vec2) -> float:
            """Prodotto vettoriale (AB x AC)."""
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

        return (
            abs(cross_2d(self.start, self.end, other.start)) < tolerance
            and abs(cross_2d(self.start, self.end, other.end)) < tolerance
        )

    def point_at(self, t: float) -> Vec2:
        """Punto parametrico t in [0,1] sul segmento."""
        return Vec2(
            self.start.x + t * (self.end.x - self.start.x),
            self.start.y + t * (self.end.y - self.start.y),
        )

    def contains_point(self, p: Vec2, tolerance: float = 0.01) -> bool:
        """Verifica se p giace sul segmento (entro tolerance)."""
        if not self.is_collinear_with(WireSegment(p, p), tolerance):
            return False
        # Verifica che p sia dentro il bounding box del segmento (con margini)
        min_x = min(self.start.x, self.end.x) - tolerance
        max_x = max(self.start.x, self.end.x) + tolerance
        min_y = min(self.start.y, self.end.y) - tolerance
        max_y = max(self.start.y, self.end.y) + tolerance
        return min_x <= p.x <= max_x and min_y <= p.y <= max_y


@dataclass
class Junction:
    """Punto di giunzione (nodo) nel circuito."""

    position: Vec2
    is_explicit: bool = True  # True = dot disegnato, False = T-junction implicita
    connected_wires: list[int] = field(default_factory=list)  # indici di WireSegment


@dataclass
class NetLabel:
    """Etichetta di net (locale o globale)."""

    text: str
    position: Vec2
    is_global: bool = False
    orientation: float = 0.0  # gradi


@dataclass
class Net:
    """Rete elettrica (insieme di segmenti connessi)."""

    net_id: int
    name: str | None = None
    segments: list[WireSegment] = field(default_factory=list)
    labels: list[NetLabel] = field(default_factory=list)

    def add_segment(self, seg: WireSegment) -> None:
        self.segments.append(seg)
        seg.net_id = self.net_id


class KiCadNetReconstructor:
    """Ricostruisce i net da un file .kicad_sch (formato S-expression KiCad 6+)."""

    def __init__(self, coord_sys: CoordinateSystem | None = None) -> None:
        self.coord_sys = coord_sys or CoordinateSystem()
        self.wires: list[WireSegment] = []
        self.junctions: list[Junction] = []
        self.labels: list[NetLabel] = []
        self.nets: list[Net] = []
        self._net_counter = 0

    def parse_file(self, path: Path | str) -> None:
        """Parse di un file .kicad_sch (S-expression)."""
        path = Path(path)
        logger.info("Parsing KiCad schematic", file=str(path))

        with open(path, encoding="utf-8") as f:
            content = f.read()

        self._parse_sexpr(content)

    def _parse_sexpr(self, content: str) -> None:
        """Parser S-expression semplificato per .kicad_sch."""
        # TODO: implementare parser robusto. Per ora stub.
        # KiCad 6+ usa formato testo con sezioni (wire, junction, label, symbol...)
        # Esempio:
        #   (wire (pts (xy 1.0 2.0) (xy 3.0 2.0)))
        #   (junction (at 2.0 2.0))
        #   (label "GND" (at 4.0 5.0) (effects ...))

        # Extract wire segments
        import re

        # Pattern per wire: (wire (pts (xy x1 y1) (xy x2 y2)))
        wire_pattern = re.compile(
            r"\(wire\s+\(pts\s+\(xy\s+([-\d.]+)\s+([-\d.]+)\)\s+\(xy\s+([-\d.]+)\s+([-\d.]+)\)\)"
        )
        for m in wire_pattern.finditer(content):
            x1, y1, x2, y2 = map(float, m.groups())
            self.wires.append(
                WireSegment(
                    start=Vec2(x1, y1),
                    end=Vec2(x2, y2),
                )
            )

        # Pattern per junction: (junction (at x y) ...)
        junction_pattern = re.compile(r"\(junction\s+\(at\s+([-\d.]+)\s+([-\d.]+)")
        for m in junction_pattern.finditer(content):
            x, y = map(float, m.groups())
            self.junctions.append(
                Junction(
                    position=Vec2(x, y),
                    is_explicit=True,
                )
            )

        # Pattern per label: (global_label "text" ... (at x y ...) )
        label_pattern = re.compile(
            r'\((?:global_)?label\s+"([^"]+)".*?\(at\s+([-\d.]+)\s+([-\d.]+)'
        )
        for m in label_pattern.finditer(content):
            text, x, y = m.group(1), float(m.group(2)), float(m.group(3))
            is_global = m.group(0).startswith("(global_label")
            self.labels.append(
                NetLabel(
                    text=text,
                    position=Vec2(x, y),
                    is_global=is_global,
                )
            )

        logger.info(
            "Parsed elements",
            wires=len(self.wires),
            junctions=len(self.junctions),
            labels=len(self.labels),
        )

    def wire_merge(self, tolerance: float = 0.01) -> None:
        """Unisce segmenti di wire collineari e sovrapposti.

        Algoritmo: iterativo fino a convergenza.
        """
        logger.info("Starting wire merge", tolerance=tolerance)

        if not self.wires:
            return

        changed = True
        iterations = 0
        max_iterations = len(self.wires) * 2  # safeguard

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            new_wires: list[WireSegment] = []

            for wire in self.wires:
                merged = False
                for i, existing in enumerate(new_wires):
                    merged_wire = wire.merge_with(existing, tolerance)
                    if merged_wire is not None:
                        new_wires[i] = merged_wire
                        merged = True
                        changed = True
                        break

                if not merged:
                    new_wires.append(wire)

            self.wires = new_wires

        logger.info("Wire merge complete", iterations=iterations, wires=len(self.wires))

    def junction_detect(self, implicit_tolerance: float = 0.01) -> None:
        """Rileva junction esplicite (dot) e implicite (T-junction).

        Una T-junction implicita si verifica quando un segmento termina
        esattamente su un altro segmento (ma non ad un'estremità).
        """
        logger.info("Starting junction detection")

        # 1. Collega junction esplicite ai wire
        explicit_positions = {j.position for j in self.junctions if j.is_explicit}

        # 2. Trova T-junctions implicite: un'estremità di un wire giace su un altro wire
        implicit_junctions: list[Junction] = []

        for i, wire_a in enumerate(self.wires):
            for j, wire_b in enumerate(self.wires):
                if i == j:
                    continue

                # Controlla se un'estremità di wire_a è su wire_b (ma non ad un'estremità di wire_b)
                for end_pt in [wire_a.start, wire_a.end]:
                    if (
                        wire_b.contains_point(end_pt, implicit_tolerance)
                        and end_pt.distance_to(wire_b.start) > implicit_tolerance
                        and end_pt.distance_to(wire_b.end) > implicit_tolerance
                        and end_pt not in explicit_positions
                    ):
                        implicit_junctions.append(
                            Junction(
                                position=end_pt,
                                is_explicit=False,
                            )
                        )
                        explicit_positions.add(end_pt)

        self.junctions.extend(implicit_junctions)

        # 3. Associa ogni junction ai wire connessi
        for junc in self.junctions:
            junc.connected_wires = []
            for wi, wire in enumerate(self.wires):
                if wire.contains_point(junc.position, implicit_tolerance):
                    junc.connected_wires.append(wi)

        logger.info(
            "Junction detection complete",
            explicit=sum(1 for j in self.junctions if j.is_explicit),
            implicit=sum(1 for j in self.junctions if not j.is_explicit),
        )

    def label_global_scope(self) -> None:
        """Unisce nets tramite label globali identiche.

        Due nets con lo stesso nome globale sono la stessa net,
        indipendentemente dalla loro connessione topologica.
        """
        logger.info("Starting global label scope merge")

        global_labels: dict[str, list[NetLabel]] = {}
        for label in self.labels:
            if label.is_global:
                global_labels.setdefault(label.text, []).append(label)

        # Per ogni gruppo di label globali con stesso nome,
        # assicurati che i wire più vicini siano nella stessa net
        for name, labels in global_labels.items():
            if len(labels) < 2:
                continue

            # Trova i wire più vicini a ciascuna label
            closest_wires: list[int] = []
            for label in labels:
                closest_wire = self._find_closest_wire(label.position)
                if closest_wire is not None:
                    closest_wires.append(closest_wire)

            # Se abbiamo wire vicini, forza la stessa net_id
            if len(closest_wires) >= 2:
                # Trova la net del primo wire
                first_net = self._find_net_for_wire(closest_wires[0])
                if first_net is None:
                    first_net = self._create_net()
                    first_net.add_segment(self.wires[closest_wires[0]])

                # Unisci tutti gli altri wire nella stessa net
                for wi in closest_wires[1:]:
                    other_net = self._find_net_for_wire(wi)
                    if other_net is None:
                        first_net.add_segment(self.wires[wi])
                    elif other_net.net_id != first_net.net_id:
                        self._merge_nets(first_net, other_net)

                first_net.name = name
                first_net.labels.extend(labels)

        logger.info("Global label scope merge complete")

    def build_nets(self) -> list[Net]:
        """Costruisce i nets connettendo i wire ai junction points.

        Algoritmo: BFS/DFS sui wire. Due wire sono nella stessa net se
        condividono un junction point (esplicito o implicito) o un'estremità.
        """
        logger.info("Building nets from wire graph")

        # Mappa: punto -> indici di wire connessi
        point_to_wires: dict[tuple[float, float], list[int]] = {}

        for wi, wire in enumerate(self.wires):
            for pt in [wire.start, wire.end]:
                key = (
                    round(pt.x, 6),
                    round(pt.y, 6),
                )  # arrotonda per evitare floating point issues
                point_to_wires.setdefault(key, []).append(wi)

        # Aggiungi anche i junction points
        for junc in self.junctions:
            key = (round(junc.position.x, 6), round(junc.position.y, 6))
            for wi in junc.connected_wires:
                if wi not in point_to_wires.get(key, []):
                    point_to_wires.setdefault(key, []).append(wi)

        # DFS per trovare componenti connesse
        visited = set()
        nets: list[Net] = []

        for wi in range(len(self.wires)):
            if wi in visited:
                continue

            net = self._create_net()
            stack = [wi]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                net.add_segment(self.wires[current])

                # Trova wire connessi attraverso point_to_wires
                wire = self.wires[current]
                for pt in [wire.start, wire.end]:
                    key = (round(pt.x, 6), round(pt.y, 6))
                    for connected_wi in point_to_wires.get(key, []):
                        if connected_wi not in visited:
                            stack.append(connected_wi)

                # Aggiungi anche i wire connessi via junction
                for junc in self.junctions:
                    if current in junc.connected_wires:
                        for connected_wi in junc.connected_wires:
                            if connected_wi not in visited:
                                stack.append(connected_wi)

            # Assegna nome dalla label se presente
            for label in self.labels:
                for seg in net.segments:
                    if seg.contains_point(label.position, tolerance=0.5):
                        if label.is_global or net.name is None:
                            net.name = label.text
                        net.labels.append(label)
                        break

            nets.append(net)

        self.nets = nets
        logger.info("Net building complete", nets=len(nets))
        return nets

    def _create_net(self) -> Net:
        self._net_counter += 1
        return Net(net_id=self._net_counter)

    def _find_net_for_wire(self, wire_index: int) -> Net | None:
        """Trova la net a cui appartiene un wire dato il suo indice."""
        for net in self.nets:
            for seg in net.segments:
                if seg is self.wires[wire_index]:
                    return net
        return None

    def _merge_nets(self, net_a: Net, net_b: Net) -> Net:
        """Unisce net_b in net_a."""
        for seg in net_b.segments:
            net_a.add_segment(seg)
        net_a.labels.extend(net_b.labels)
        if net_a.name is None and net_b.name is not None:
            net_a.name = net_b.name
        # Rimuovi net_b dalla lista
        self.nets = [n for n in self.nets if n.net_id != net_b.net_id]
        return net_a

    def _find_closest_wire(self, position: Vec2) -> int | None:
        """Trova l'indice del wire più vicino a una posizione."""
        min_dist = float("inf")
        closest = None

        for wi, wire in enumerate(self.wires):
            # Distanza punto-segmento
            dist = self._point_to_segment_distance(position, wire)
            if dist < min_dist:
                min_dist = dist
                closest = wi

        return closest

    @staticmethod
    def _point_to_segment_distance(p: Vec2, seg: WireSegment) -> float:
        """Distanza minima da un punto a un segmento."""
        # Proiezione del punto sulla retta del segmento
        line_vec = Vec2(seg.end.x - seg.start.x, seg.end.y - seg.start.y)
        point_vec = Vec2(p.x - seg.start.x, p.y - seg.start.y)

        line_len_sq = line_vec.x**2 + line_vec.y**2
        if line_len_sq < 1e-12:
            return p.distance_to(seg.start)

        t = max(0.0, min(1.0, (point_vec.x * line_vec.x + point_vec.y * line_vec.y) / line_len_sq))
        projection = Vec2(
            seg.start.x + t * line_vec.x,
            seg.start.y + t * line_vec.y,
        )
        return p.distance_to(projection)

    def export_netlist(self, format: str = "json") -> dict[str, Any]:
        """Esporta la netlist in formato dict (pronto per JSON)."""
        return {
            "nets": [
                {
                    "id": net.net_id,
                    "name": net.name or f"Net-{net.net_id}",
                    "segments": [
                        {
                            "start": {"x": seg.start.x, "y": seg.start.y},
                            "end": {"x": seg.end.x, "y": seg.end.y},
                        }
                        for seg in net.segments
                    ],
                    "labels": [{"text": lbl.text, "global": lbl.is_global} for lbl in net.labels],
                }
                for net in self.nets
            ],
            "junctions": [
                {
                    "x": j.position.x,
                    "y": j.position.y,
                    "explicit": j.is_explicit,
                }
                for j in self.junctions
            ],
        }


def main() -> None:
    """CLI entry point per test rapidi."""
    import typer

    app = typer.Typer()

    @app.command()
    def reconstruct(input: Path, output: Path | None = None, tolerance: float = 0.01) -> None:
        recon = KiCadNetReconstructor()
        recon.parse_file(input)
        recon.wire_merge(tolerance)
        recon.junction_detect(tolerance)
        recon.label_global_scope()
        recon.build_nets()

        netlist = recon.export_netlist()

        if output:
            import json

            with open(output, "w", encoding="utf-8") as f:
                json.dump(netlist, f, indent=2)
            typer.echo(f"Netlist scritta su {output}")
        else:
            import json

            typer.echo(json.dumps(netlist, indent=2))

    app()


if __name__ == "__main__":
    main()
