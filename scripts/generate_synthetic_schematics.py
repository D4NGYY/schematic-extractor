from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from scripts.kicad_net_reconstructor import Junction, KiCadNetReconstructor, NetLabel, WireSegment
from src.core.coordinate_system import Vec2

logger = structlog.get_logger("synthetic_schematics")


@dataclass
class SyntheticSchematic:
    """Schema sintetico generato per stress test topologico."""

    name: str
    wires: list[WireSegment] = field(default_factory=list)
    junctions: list[Junction] = field(default_factory=list)
    labels: list[NetLabel] = field(default_factory=list)

    def to_kicad_sch(self) -> str:
        """Genera un file .kicad_sch minimale."""
        lines = [
            "(kicad_sch (version 20211123) (generator eeschema)",
            '  (uuid "00000000-0000-0000-0000-000000000000")',
            '  (paper "A4")',
            "  (lib_symbols)",
        ]

        for j in self.junctions:
            lines.append(f"  (junction (at {j.position.x} {j.position.y}))")

        for i, w in enumerate(self.wires):
            lines.append(
                f"  (wire (pts (xy {w.start.x} {w.start.y}) (xy {w.end.x} {w.end.y}))"
                f' (uuid "00000000-0000-0000-0000-{i:012d}"))'
            )

        for lbl in self.labels:
            kind = "global_label" if lbl.is_global else "label"
            lines.append(f'  ({kind} "{lbl.text}" (at {lbl.position.x} {lbl.position.y}))')

        lines.append("  (sheet_instances")
        lines.append('    (path "/" (page "1"))')
        lines.append("  )")
        lines.append(")")

        return "\n".join(lines)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(self.to_kicad_sch(), encoding="utf-8")
        logger.info("Saved synthetic schematic", path=str(path), wires=len(self.wires))


def generate_resistor_ladder(
    num_stages: int = 5,
    spacing: float = 10.0,
    origin: Vec2 | None = None,
) -> SyntheticSchematic:
    """Genera un partitore resistivo a scala (seriale-parallelo).

    Topology: VCC -- R1 -- R2 -- R3 -- ... -- GND
              |     |     |     |
              |     |     |     |
             C1    C2    C3    C4
    """
    if origin is None:
        origin = Vec2(50.0, 50.0)
    sch = SyntheticSchematic(name=f"resistor_ladder_{num_stages}")

    x = origin.x
    y = origin.y

    # Traccia principale orizzontale
    for i in range(num_stages):
        # Segmento orizzontale tra resistori
        next_x = x + spacing
        sch.wires.append(WireSegment(Vec2(x, y), Vec2(next_x, y)))

        # Capacitore a massa (se non ultimo)
        if i < num_stages - 1:
            sch.wires.append(
                WireSegment(Vec2(x + spacing / 2, y), Vec2(x + spacing / 2, y + spacing / 2))
            )
            sch.labels.append(
                NetLabel("GND", Vec2(x + spacing / 2, y + spacing / 2), is_global=True)
            )

        x = next_x

    # Label VCC all'inizio e GND alla fine
    sch.labels.append(NetLabel("VCC", Vec2(origin.x, y), is_global=True))
    sch.labels.append(NetLabel("GND", Vec2(x, y), is_global=True))

    return sch


def generate_bridge_rectifier(
    size: float = 20.0,
    origin: Vec2 | None = None,
) -> SyntheticSchematic:
    """Genera un ponte di Graetz (4 diodi + trasformatore + carico).

    Topology:
           AC_in+ ----|>|----+---- VCC
                      |     |
                     ---    R_load
                      |     |
           AC_in- ----|<|----+---- GND
    """
    if origin is None:
        origin = Vec2(50.0, 50.0)
    sch = SyntheticSchematic(name="bridge_rectifier")

    x, y = origin.x, origin.y

    # Input AC
    sch.wires.append(WireSegment(Vec2(x, y), Vec2(x + size / 2, y)))
    sch.wires.append(WireSegment(Vec2(x, y + size), Vec2(x + size / 2, y + size)))

    # Ponte
    sch.wires.append(WireSegment(Vec2(x + size / 2, y), Vec2(x + size, y + size / 2)))
    sch.wires.append(WireSegment(Vec2(x + size / 2, y + size), Vec2(x + size, y + size / 2)))
    sch.wires.append(WireSegment(Vec2(x + size, y + size / 2), Vec2(x + size * 1.5, y + size / 2)))

    # Carico
    sch.wires.append(WireSegment(Vec2(x + size, y + size / 2), Vec2(x + size, y + size * 1.5)))

    # Labels
    sch.labels.append(NetLabel("AC_IN", Vec2(x, y + size / 2), is_global=True))
    sch.labels.append(NetLabel("VCC", Vec2(x + size * 1.5, y + size / 2), is_global=True))
    sch.labels.append(NetLabel("GND", Vec2(x + size, y + size * 1.5), is_global=True))

    # Junction al nodo di uscita
    sch.junctions.append(Junction(Vec2(x + size, y + size / 2), is_explicit=True))

    return sch


def generate_random_mesh(
    nodes: int = 10,
    edges: int = 15,
    area: float = 100.0,
    origin: Vec2 | None = None,
    seed: int | None = None,
) -> SyntheticSchematic:
    """Genera una mesh casuale per stress test del net reconstructor.

    Crea `nodes` punti casuali in un'area e li collega con `edges` wire.
    Utile per verificare che il BFS di build_nets() gestisca grafi complessi.
    """
    rng = random.Random(seed)
    if origin is None:
        origin = Vec2(50.0, 50.0)
    sch = SyntheticSchematic(name=f"random_mesh_{nodes}_{edges}")

    points = [
        Vec2(
            origin.x + rng.uniform(0, area),
            origin.y + rng.uniform(0, area),
        )
        for _ in range(nodes)
    ]

    # Crea una Minimum Spanning Tree per garantire connettività
    for i in range(nodes - 1):
        sch.wires.append(WireSegment(points[i], points[i + 1]))

    # Aggiungi edge casuali extra
    for _ in range(edges - (nodes - 1)):
        a, b = rng.sample(range(nodes), 2)
        sch.wires.append(WireSegment(points[a], points[b]))

    return sch


def generate_stress_test_suite(output_dir: Path | str = "data/kicad/synthetic") -> None:
    """Genera la suite completa di schemi sintetici per stress test."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schematics = [
        generate_resistor_ladder(num_stages=3, spacing=10.0),
        generate_resistor_ladder(num_stages=10, spacing=5.0),
        generate_bridge_rectifier(size=20.0),
        generate_bridge_rectifier(size=50.0),
        generate_random_mesh(nodes=5, edges=7, seed=42),
        generate_random_mesh(nodes=20, edges=30, seed=123),
        generate_random_mesh(nodes=50, edges=80, seed=999),
    ]

    for sch in schematics:
        sch.save(output_dir / f"{sch.name}.kicad_sch")

    logger.info("Stress test suite generated", count=len(schematics), dir=str(output_dir))


def run_stress_tests(output_dir: Path | str = "data/kicad/synthetic") -> dict[str, Any]:
    """Esegue la suite di stress test e valida ogni schema."""
    output_dir = Path(output_dir)
    results = {}

    for path in output_dir.glob("*.kicad_sch"):
        logger.info("Running stress test", file=path.name)

        recon = KiCadNetReconstructor()
        recon.parse_file(path)
        recon.wire_merge(tolerance=0.01)
        recon.junction_detect(implicit_tolerance=0.01)
        recon.label_global_scope()
        nets = recon.build_nets()

        # Metriche di stress
        total_wire_length = sum(w.length for w in recon.wires)
        avg_segments_per_net = sum(len(n.segments) for n in nets) / len(nets) if nets else 0

        results[path.name] = {
            "wires": len(recon.wires),
            "junctions": len(recon.junctions),
            "labels": len(recon.labels),
            "nets": len(nets),
            "total_wire_length_mm": round(total_wire_length, 2),
            "avg_segments_per_net": round(avg_segments_per_net, 2),
            "pass": len(nets) > 0 and all(len(n.segments) > 0 for n in nets),
        }

    return results


def main() -> None:
    import json

    generate_stress_test_suite()
    results = run_stress_tests()

    print(json.dumps(results, indent=2))

    all_pass = all(r["pass"] for r in results.values())
    if all_pass:
        print("PASS: Tutti gli stress test passati")
    else:
        print("FAIL: Alcuni stress test falliti")
        for name, r in results.items():
            if not r["pass"]:
                print(f"  - {name}: {r}")


if __name__ == "__main__":
    main()
