from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from src.core.coordinate_system import CoordinateSystem, Vec2

logger = structlog.get_logger("align_kicad_to_pdf")


@dataclass
class SymbolMapping:
    """Mappatura di un simbolo KiCad al bounding box del cluster PDF corrispondente."""

    ref: str
    kicad_pos: Vec2
    pdf_bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points
    confidence: float


class KiCadToPDFAligner:
    """Allinea simboli KiCad ai cluster PDF per generare training set automatico.

    Assunzioni:
    - Il PDF è esportato da KiCad con stessa scala e orientamento
    - Coordinate KiCad (mm) → PDF (points) via CoordinateSystem.mm_to_points()
    - Margine di tolleranza per traslazione e scala

    Output:
    - JSON con mapping simboli → cluster per ogni pagina
    """

    def __init__(self, tolerance_pts: float = 5.0) -> None:
        """
        Args:
            tolerance_pts: tolleranza in punti PDF per matching (default 5pt ≈ 1.76mm)
        """
        self.tolerance = tolerance_pts
        self.coord = CoordinateSystem()

    def extract_kicad_symbols(self, kicad_path: Path) -> list[dict[str, Any]]:
        """Estrae simboli e coordinate da un file .kicad_sch."""
        # Per ora, stub che usa regex sul file .kicad_sch
        # In produzione, usare il parser S-expression completo

        symbols = []
        content = kicad_path.read_text(encoding="utf-8")

        import re

        # Estrazione più robusta: cerca blocchi (symbol ...)
        for match in re.finditer(r"\(symbol\s+", content):
            start = match.start()
            # Trova la fine del blocco (parentesi bilanciate)
            depth = 0
            end = start
            for _i, ch in enumerate(content[start:], start):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = _i + 1
                        break

            block = content[start:end]

            # Estrai posizione
            pos_match = re.search(r"\(at\s+([-\d.]+)\s+([-\d.]+)", block)
            if not pos_match:
                continue
            x, y = float(pos_match.group(1)), float(pos_match.group(2))

            # Estrai Reference e Value
            ref_match = re.search(r'"Reference"\s+"([^"]+)"', block)
            val_match = re.search(r'"Value"\s+"([^"]+)"', block)

            symbols.append(
                {
                    "lib_id": re.search(r'\(lib_id\s+"([^"]+)"\)', block),
                    "pos_mm": (x, y),
                    "ref": ref_match.group(1) if ref_match else None,
                    "value": val_match.group(1) if val_match else None,
                }
            )

        logger.info("Extracted KiCad symbols", count=len(symbols), file=str(kicad_path))
        return symbols

    def extract_pdf_clusters(self, pdf_path: Path) -> list[dict[str, Any]]:
        """Estrae cluster di segmenti dal PDF tramite PyMuPDF."""
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        clusters = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Estrai disegni vettoriali (linee, archi, rettangoli)
            drawings = page.get_drawings()

            # Raggruppa per prossimità spaziale (semplice: bounding box dell'intero gruppo)
            for _i, drawing in enumerate(drawings):
                bbox = drawing.get("rect", page.rect)
                clusters.append(
                    {
                        "page": page_num,
                        "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                        "items": len(drawing.get("items", [])),
                    }
                )

        doc.close()

        logger.info("Extracted PDF clusters", count=len(clusters), file=str(pdf_path))
        return clusters

    def align(
        self,
        kicad_path: Path | str,
        pdf_path: Path | str,
    ) -> list[SymbolMapping]:
        """Mappa simboli KiCad ai cluster PDF per generare training set.

        Algoritmo:
        1. Converte coordinate KiCad (mm) → PDF (points)
        2. Per ogni simbolo KiCad, trova il cluster PDF più vicino
        3. Verifica che la distanza sia < tolerance
        4. Assegna confidence inversamente proporzionale alla distanza
        """
        kicad_path = Path(kicad_path)
        pdf_path = Path(pdf_path)

        symbols = self.extract_kicad_symbols(kicad_path)
        clusters = self.extract_pdf_clusters(pdf_path)

        mappings: list[SymbolMapping] = []

        for sym in symbols:
            kx_mm, ky_mm = sym["pos_mm"]
            # KiCad y è verso l'alto, PDF y è verso il basso → inverti y
            kx_pts = self.coord.mm_to_points(kx_mm)
            ky_pts = self.coord.mm_to_points(-ky_mm)  # flip Y

            # Trova cluster più vicino
            best_cluster = None
            best_dist = float("inf")

            for cluster in clusters:
                cx = (cluster["bbox"][0] + cluster["bbox"][2]) / 2
                cy = (cluster["bbox"][1] + cluster["bbox"][3]) / 2

                dist = ((kx_pts - cx) ** 2 + (ky_pts - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cluster

            if best_cluster and best_dist < self.tolerance:
                confidence = max(0.0, 1.0 - best_dist / self.tolerance)
                mappings.append(
                    SymbolMapping(
                        ref=sym["ref"] or "UNKNOWN",
                        kicad_pos=Vec2(kx_mm, ky_mm),
                        pdf_bbox=best_cluster["bbox"],
                        confidence=confidence,
                    )
                )
            else:
                logger.warning(
                    "No matching cluster for symbol",
                    ref=sym["ref"],
                    dist=best_dist,
                )

        logger.info(
            "Alignment complete",
            mappings=len(mappings),
            symbols=len(symbols),
        )
        return mappings

    def export_training_set(
        self,
        mappings: list[SymbolMapping],
        output_path: Path | str,
    ) -> None:
        """Esporta il training set in formato JSON."""
        output_path = Path(output_path)

        data = [
            {
                "ref": m.ref,
                "kicad_x": m.kicad_pos.x,
                "kicad_y": m.kicad_pos.y,
                "pdf_bbox": m.pdf_bbox,
                "confidence": m.confidence,
            }
            for m in mappings
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Training set exported", path=str(output_path), samples=len(data))


def main() -> None:
    import typer

    app = typer.Typer()
    aligner = KiCadToPDFAligner()

    @app.command()
    def align(
        kicad: Path,
        pdf: Path,
        output: Path = Path("data/ground_truth/mapping.json"),
    ) -> None:
        """Allinea simboli KiCad ai cluster PDF e genera training set."""
        mappings = aligner.align(kicad, pdf)
        aligner.export_training_set(mappings, output)
        typer.echo(f"✅ Mappatura generata: {output} ({len(mappings)} simboli)")

    app()


if __name__ == "__main__":
    main()
