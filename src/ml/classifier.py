from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier

from src.ml.clustering import ComponentCluster
from src.ml.feature_extractor import FeatureExtractor

logger = structlog.get_logger("classifier")

# Classi componenti (superset usato anche dal rule-based; il RF ne usa un sottoinsieme
# nelle versioni pre-training, ma il set va mantenuto stabile per compatibilità pkl)
COMPONENT_CLASSES = [
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "transistor",
    "opamp",
    "connector",
    "ic",
    "power_symbol",
    "testpoint",
    "regulator",
    "crystal",
    "fuse",
    "switch",
    "transformer",
    "relay",
    "unknown",
]

# ── Rule-based classifier ──────────────────────────────────────────────────────

_PREFIX_RE = re.compile(r"^([A-Z]+)\d")

# Prefissi a 2 lettere con significato proprio (priorità su 1 lettera)
_TWO_PREFIX_MAP: dict[str, tuple[str, float]] = {
    "TP": ("testpoint", 0.95),
    "SW": ("switch", 0.95),
    "VR": ("regulator", 0.90),
    "IC": ("ic", 0.95),
    "RN": ("resistor", 0.90),   # resistor network
    "FB": ("inductor", 0.85),   # ferrite bead
    "TR": ("transformer", 0.85),
}

# Prefissi a 1 lettera (convenzione EDA standard IEC 60617 / IEEE 315)
_ONE_PREFIX_MAP: dict[str, tuple[str, float]] = {
    "R": ("resistor", 0.95),
    "C": ("capacitor", 0.95),
    "L": ("inductor", 0.95),
    "D": ("diode", 0.95),
    "Q": ("transistor", 0.95),
    "U": ("ic", 0.90),
    "J": ("connector", 0.95),
    "P": ("connector", 0.90),
    "Y": ("crystal", 0.95),
    "X": ("crystal", 0.90),
    "F": ("fuse", 0.95),
    "T": ("transformer", 0.90),
    "K": ("relay", 0.95),
    "S": ("switch", 0.85),
    "M": ("ic", 0.70),          # motor driver / IC — incerto
}


class RuleBasedClassifier:
    """Classificatore provvisorio rule-based (attivo finché ML non è addestrato).

    Segnale primario: prefisso del reference designator (R→resistor, QB→transistor…).
    Fallback geometrico: caratteristiche del cluster quando manca il reference.
    """

    def classify(
        self, ref: str | None, cluster: ComponentCluster
    ) -> tuple[str, float]:
        """Ritorna (class_name, confidence). Prova prima il ref, poi la geometria."""
        if ref is not None:
            result = self._classify_by_ref(ref)
            if result is not None:
                return result
        return self._classify_geometric(cluster)

    @staticmethod
    def _classify_by_ref(ref: str) -> tuple[str, float] | None:
        """Estrae il prefisso alpha e lo mappa alla classe."""
        m = _PREFIX_RE.match(ref.upper())
        if not m:
            return None
        prefix = m.group(1)
        # Priorità: 2 lettere specifiche → 1a lettera (QB→Q→transistor)
        if len(prefix) >= 2 and prefix[:2] in _TWO_PREFIX_MAP:
            return _TWO_PREFIX_MAP[prefix[:2]]
        first = prefix[0]
        if first in _ONE_PREFIX_MAP:
            return _ONE_PREFIX_MAP[first]
        return None

    @staticmethod
    def _classify_geometric(cluster: ComponentCluster) -> tuple[str, float]:
        """Fallback con segnali geometrici deboli; ritorna unknown se incerto."""
        n_segs = len(cluster.segments)
        n_shapes = len(cluster.shapes)
        x0, y0, x1, y1 = cluster.bbox
        w = x1 - x0
        h = y1 - y0
        area = w * h
        ar = w / h if h > 0 else 1.0

        # Power symbols: cluster piccolo, pochi segmenti, nessuna shape
        if n_segs <= 4 and n_shapes == 0 and area < 400:
            return "power_symbol", 0.45
        # Corpo IC largo con molti segmenti
        if ar > 3.0 and n_segs > 4:
            return "ic", 0.35
        return "unknown", 0.25


class ComponentClassifier:
    """Classificatore Random Forest per componenti schematici.

    Target: >90% accuracy sulle 10 classi base.
    Fallback: se confidence < 0.7, classifica come 'unknown'.
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model: RandomForestClassifier | None = None
        self.fe = FeatureExtractor()
        self.classes = COMPONENT_CLASSES
        if model_path is not None:
            self.load(model_path)

    def fit(self, features: np.ndarray, y: np.ndarray) -> None:
        """Addestra il Random Forest."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(features, y)
        logger.info("Model trained", samples=len(features), features=features.shape[1])

    def predict(self, cluster: ComponentCluster) -> tuple[str, float]:
        """Predice la classe di un componente e ritorna (class_name, confidence)."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        features = self.fe.extract(cluster).to_array().reshape(1, -1)
        proba = self.model.predict_proba(features)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        class_name = self.model.classes_[pred_idx]

        if confidence < 0.7:
            class_name = "unknown"

        logger.debug(
            "Prediction",
            class_name=class_name,
            confidence=confidence,
            cluster_id=cluster.cluster_id,
        )
        return str(class_name), confidence

    def save(self, path: Path | str) -> None:
        """Salva il modello addestrato."""
        path = Path(path)
        if self.model is None:
            raise RuntimeError("No model to save")
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "classes": self.classes}, f)
        logger.info("Model saved", path=str(path))

    def load(self, path: Path | str) -> None:
        """Carica un modello salvato."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.classes = data.get("classes", COMPONENT_CLASSES)
        logger.info("Model loaded", path=str(path))

    def extract_features_batch(
        self, clusters: list[ComponentCluster]
    ) -> np.ndarray:
        """Estrae feature vector per una lista di cluster."""
        return np.array([self.fe.extract(c).to_array() for c in clusters])
