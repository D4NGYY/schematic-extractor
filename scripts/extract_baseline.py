"""Baseline extraction report: conta ref designator e valori rilevati per pagina."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pdf_parser import VectorExtractor


def report(pdf_path: str) -> None:
    pages = VectorExtractor().extract(pdf_path)
    for page in pages:
        refs = page.ref_blocks()
        values = page.value_blocks()
        print(f"\n=== Pagina {page.page_num} ===")
        print(f"  Testo totale : {len(page.text_blocks)} blocchi")
        print(f"  Ref designator: {len(refs)} -> {[r.text for r in refs]}")
        print(f"  Valori        : {len(values)} -> {[v.text for v in values]}")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "test_input/bryston_schematic.pdf")
