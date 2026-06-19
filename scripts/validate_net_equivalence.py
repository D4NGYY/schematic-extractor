from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_net_equivalence(
    netlist_a: dict[str, Any] | Path | str,
    netlist_b: dict[str, Any] | Path | str,
) -> dict[str, Any]:
    """Valida che due netlist siano equivalenti.

    Due netlist sono equivalenti se l'insieme dei pin collegati per ogni
    componente è identico, a meno del renaming interno dei net IDs.

    Per ora, implementazione semplificata: confronta la topologia dei segmenti.
    """
    if isinstance(netlist_a, (Path, str)):
        with open(netlist_a, encoding="utf-8") as f:
            netlist_a = json.load(f)
    if isinstance(netlist_b, (Path, str)):
        with open(netlist_b, encoding="utf-8") as f:
            netlist_b = json.load(f)

    nets_a = netlist_a.get("nets", [])
    nets_b = netlist_b.get("nets", [])

    # Estrai set di segmenti per ogni net (ignorando net_id e name)
    def net_to_segments(
        net: dict[str, Any],
    ) -> set[tuple[tuple[float, float], tuple[float, float]]]:
        segs = set()
        for seg in net.get("segments", []):
            start = (round(seg["start"]["x"], 6), round(seg["start"]["y"], 6))
            end = (round(seg["end"]["x"], 6), round(seg["end"]["y"], 6))
            # Normalizza: punto più piccolo (lex) prima
            if start > end:
                start, end = end, start
            segs.add((start, end))
        return segs

    nets_a_sets = [net_to_segments(n) for n in nets_a]
    nets_b_sets = [net_to_segments(n) for n in nets_b]

    # Verifica che ogni net di A abbia una corrispondente in B
    matched = []
    unmatched_a = []
    used_b = set()

    for i, segs_a in enumerate(nets_a_sets):
        found = False
        for j, segs_b in enumerate(nets_b_sets):
            if j in used_b:
                continue
            if segs_a == segs_b:
                matched.append((i, j))
                used_b.add(j)
                found = True
                break
        if not found:
            unmatched_a.append(i)

    unmatched_b = [j for j in range(len(nets_b_sets)) if j not in used_b]

    return {
        "equivalent": len(unmatched_a) == 0 and len(unmatched_b) == 0,
        "matched_nets": len(matched),
        "nets_a": len(nets_a),
        "nets_b": len(nets_b),
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
    }


def main() -> None:
    import typer

    app = typer.Typer()

    @app.command()
    def validate(a: Path, b: Path) -> None:
        result = validate_net_equivalence(a, b)
        import json

        typer.echo(json.dumps(result, indent=2))
        if result["equivalent"]:
            typer.echo("✅ Netlist equivalenti")
        else:
            typer.echo("❌ Netlist NON equivalenti")

    app()


if __name__ == "__main__":
    main()
