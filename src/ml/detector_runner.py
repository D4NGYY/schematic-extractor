"""Production glue between the trained YOLO detector and the graph builder.

Encapsulates the detector path that used to live only in the diagnose scripts
(`diagnosi_d3/compare_detector.py`): load weights, render page 0 at the trained
dpi, run YOLO, convert boxes to `Detection`, and let `DetectorComponentSource`
build graph components (with hybrid geometric fallback when the detector is
sparse on a page — the shipped production behavior, HANDOFF §24/§26).

Both the CLI (`src/cli/query.py`) and the Streamlit UI (`src/ui/app.py`) go
through `run_detector_or_none` so the integration logic is defined once.

Design:
- `ultralytics` + `torch` are an OPTIONAL extra (`[detector]`). They are imported
  lazily inside `DetectorRunner` so the core pipeline stays lightweight and the
  module imports cleanly even without them.
- If the extra is not installed, weights are missing, or inference raises, the
  runner returns `None` and the caller falls back silently to the geometric
  pipeline — the system NEVER crashes because of the detector.
- The detector is page-0 only (the dataset HANDOFF §23 labels page 0); multi-page
  PDFs are handled by the caller scoring pages 1+ geometrically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from src.core.pdf_parser import ExtractedPage
from src.ml.detector_source import Detection, DetectorComponentSource

logger = structlog.get_logger("detector_runner")

# Production defaults (HANDOFF §26): 150-dpi detector weights, hybrid fallback
# when the detector covers <50% of the page's visible ref-designators, container
# exclusion handled inside DetectorComponentSource via ref assignment.
DEFAULT_WEIGHTS = "runs/detect/schematic_detector-9/weights/best.pt"
DEFAULT_DPI = 150.0
DEFAULT_MIN_FRAC = 0.5


class DetectorRunner:
    """Loads YOLO weights once and runs inference on rendered page-0 images.

    Construction imports `ultralytics` (heavy); guard it behind an availability
    check so callers that never enable the detector pay nothing.
    """

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        dpi: float = DEFAULT_DPI,
        min_frac: float = DEFAULT_MIN_FRAC,
        imgsz: int | None = None,
    ) -> None:
        self.weights = str(weights)
        self.dpi = float(dpi)
        self.min_frac = float(min_frac)
        self.imgsz = imgsz
        self._model: Any = None  # ultralytics.YOLO, lazy

    def _ensure_model(self) -> bool:
        """Lazy-load the YOLO model. Returns False if unavailable (-> fallback)."""
        if self._model is not None:
            return True
        if not Path(self.weights).exists():
            logger.warning("detector_weights_missing", weights=self.weights)
            return False
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError:
            logger.warning("detector_extra_missing", hint="uv sync --extra detector")
            return False
        try:
            self._model = YOLO(self.weights)
            logger.info("detector_loaded", weights=self.weights)
            return True
        except Exception as exc:  # noqa: BLE001 — never crash the pipeline
            logger.error("detector_load_failed", error=str(exc)[:200])
            return False

    def _render_page0(self, pdf_path: str | Path) -> Path:
        """Render page 0 at self.dpi to a temp PNG; return its path.

        Uses a stable per-(pdf,dpi) cache path so repeated queries on the same
        file don't re-render. fitz is already a core dep (pymupdf).
        """
        import fitz  # noqa: PLC0415 — core dep, not optional

        cache_dir = Path(".cache/detector_render")
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = Path(pdf_path).stem
        out = cache_dir / f"{key}_{int(self.dpi)}.png"
        if not out.exists():
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            page.get_pixmap(matrix=fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)).save(str(out))
            doc.close()
        return out

    def _predict_detections(self, image_path: Path) -> list[Detection]:
        """Run YOLO inference and map raw boxes to Detection records."""
        kwargs: dict[str, Any] = {"verbose": False}
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        result = self._model(str(image_path), **kwargs)[0]
        names = result.names
        dets: list[Detection] = []
        for xyxy, cls, conf in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            strict=False,
        ):
            dets.append(
                Detection(
                    class_name=names[int(cls)],
                    bbox_px=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=float(conf),
                )
            )
        logger.info("detector_predicted", n=len(dets))
        return dets

    def run_detector_or_none(
        self,
        pdf_path: str | Path,
        page: ExtractedPage,
    ) -> tuple[list[Any] | None, str]:
        """Run the detector on page 0 and return (components, source_label).

        Returns (None, "geometric_fallback") when the detector is unavailable
        or sparse on this page (hybrid gate). Otherwise returns the detector-
        built ComponentNodes plus a human-readable label of the active path:
        "detector" or "hybrid_detector".
        """
        if not self._ensure_model():
            return None, "geometric_fallback"
        try:
            img = self._render_page0(pdf_path)
            dets = self._predict_detections(img)
            src = DetectorComponentSource(dpi=self.dpi)
            comps = src.components_or_fallback(dets, page, min_frac=self.min_frac)
        except Exception as exc:  # noqa: BLE001 — never crash the pipeline
            logger.error("detector_inference_failed", error=str(exc)[:200])
            return None, "geometric_fallback"
        if comps is None:
            # Hybrid gate tripped: detector was sparse -> caller uses geometric.
            logger.info("detector_hybrid_fallback", reason="sparse_coverage")
            return None, "geometric_fallback"
        return comps, "detector"


def is_available(weights: str | Path = DEFAULT_WEIGHTS) -> bool:
    """Cheap check: are detector weights present on disk?

    Does NOT import ultralytics. Use this to decide whether to even attempt the
    detector path (e.g. UI badge / CLI default for --detector).
    """
    return Path(weights).exists()
