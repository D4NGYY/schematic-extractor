"""Expand the (PDF, KiCad ground-truth) dataset beyond arduino_micro / nano.

WHY THIS EXISTS
---------------
The headline net-connectivity F1 (~0.36) is measured on only TWO boards, and the
greedy net-mapping tie-break is interpreter-sensitive (it shifted 0.356 -> 0.344
just moving Python 3.12 -> 3.10). To know whether 0.36 is representative we need
many more (PDF, .kicad_sch) pairs. The sandbox can't render them (Ubuntu 22.04
only ships KiCad 6, which has no `kicad-cli`; processes don't persist). So this
script runs on YOUR machine, where KiCad 7+ (with `kicad-cli`) is installed.

WHAT IT DOES
------------
For every `.kicad_sch` you point it at (a custom folder, or KiCad's own bundled
demo projects), it renders that sheet to a PDF with `kicad-cli` and drops the
(PDF, .kicad_sch) pair into `test_input/multi_schematic/<name>/`. Each sheet
file becomes one board, so the rendered PDF and the GT parsed from the SAME file
always correspond. Then run the generalized scorer:

    PYTHONHASHSEED=0 PYTHONPATH=. python diagnosi_d3/f1_all_boards.py

USAGE
-----
    # 1) KiCad's bundled demos (real, license-clean, multi-component):
    python scripts/expand_dataset.py --demos

    # 2) A folder of your own KiCad projects:
    python scripts/expand_dataset.py --source "C:/path/to/kicad/projects"

    # 3) One specific schematic:
    python scripts/expand_dataset.py --project "C:/.../board.kicad_sch"

Options: --max N (cap boards), --dpi (unused by kicad-cli, kept for parity),
--dry-run (list what would happen).

NOTE ON HIERARCHY: rendering a *root* sheet of a hierarchical design produces a
multi-page PDF but the root .kicad_sch may hold few components (the rest live in
sub-sheets). For a clean 1:1 PDF<->GT board, prefer flat single-sheet projects,
or let this script render each sub-sheet file independently (it does, since it
iterates over every *.kicad_sch it finds).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEST = Path("test_input/multi_schematic")


def find_kicad_cli() -> str | None:
    """Locate kicad-cli on PATH or in common Windows/macOS/Linux install dirs."""
    exe = shutil.which("kicad-cli")
    if exe:
        return exe
    candidates: list[Path] = []
    # Windows: C:\Program Files\KiCad\<ver>\bin\kicad-cli.exe
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    for ver in ("9.0", "8.0", "7.0"):
        candidates.append(Path(pf) / "KiCad" / ver / "bin" / "kicad-cli.exe")
    # macOS
    candidates.append(Path("/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"))
    # Linux
    candidates.append(Path("/usr/bin/kicad-cli"))
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def find_demos_dir(cli: str) -> Path | None:
    """KiCad ships demo projects under share/kicad/demos relative to the install."""
    bin_dir = Path(cli).resolve().parent
    for rel in ("../share/kicad/demos", "../../share/kicad/demos",
                "../Resources/share/kicad/demos"):
        d = (bin_dir / rel).resolve()
        if d.is_dir():
            return d
    # common Windows location
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    for ver in ("9.0", "8.0", "7.0"):
        d = Path(pf) / "KiCad" / ver / "share" / "kicad" / "demos"
        if d.is_dir():
            return d
    return None


def render(cli: str, sch: Path, out_pdf: Path, dry: bool) -> bool:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cli, "sch", "export", "pdf", "--output", str(out_pdf), str(sch)]
    if dry:
        print("  DRY:", " ".join(cmd))
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT rendering {sch.name}")
        return False
    if r.returncode != 0:
        print(f"  FAIL {sch.name}: {r.stderr.strip()[:200]}")
        return False
    return out_pdf.exists()


def collect_schematics(args: argparse.Namespace, cli: str) -> list[Path]:
    if args.project:
        return [Path(args.project)]
    if args.source:
        return sorted(Path(args.source).rglob("*.kicad_sch"))
    if args.demos:
        demos = find_demos_dir(cli)
        if not demos:
            print("Could not locate KiCad bundled demos dir.", file=sys.stderr)
            return []
        print(f"Demos dir: {demos}")
        return sorted(demos.rglob("*.kicad_sch"))
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demos", action="store_true", help="use KiCad's bundled demos")
    g.add_argument("--source", help="folder of KiCad projects to render")
    g.add_argument("--project", help="a single .kicad_sch to render")
    ap.add_argument("--max", type=int, default=0, help="cap number of boards (0=all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cli = find_kicad_cli()
    if not cli:
        print("kicad-cli not found. Install KiCad 7+ and/or add its bin/ to PATH.",
              file=sys.stderr)
        print("Windows default: C:\\Program Files\\KiCad\\<ver>\\bin\\kicad-cli.exe",
              file=sys.stderr)
        sys.exit(2)
    print(f"kicad-cli: {cli}")

    schs = collect_schematics(args, cli)
    if not schs:
        print("No .kicad_sch files found.", file=sys.stderr)
        sys.exit(1)
    if args.max:
        schs = schs[: args.max]

    print(f"Rendering {len(schs)} schematic sheet(s) -> {DEST}\n")
    ok = 0
    for sch in schs:
        name = sch.stem
        board_dir = DEST / name
        out_pdf = board_dir / f"{name}.pdf"
        gt_copy = board_dir / f"{name}.kicad_sch"
        print(f"- {name}")
        if render(cli, sch, out_pdf, args.dry_run):
            if not args.dry_run:
                shutil.copy2(sch, gt_copy)
            ok += 1
            print(f"  OK -> {out_pdf}")

    print(f"\nDone: {ok}/{len(schs)} boards ready.")
    if ok and not args.dry_run:
        print("Now score them all:")
        print("  PYTHONHASHSEED=0 PYTHONPATH=. python diagnosi_d3/f1_all_boards.py")


if __name__ == "__main__":
    main()
