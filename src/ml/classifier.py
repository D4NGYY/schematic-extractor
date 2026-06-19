from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier

from src.ml.clustering import ComponentCluster
from src.ml.feature_extractor import FeatureExtractor

logger = structlog.get_logger("classifier")

# Classi base componenti elettrici (10 classi per l'MVP)
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
    "unknown",
]


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
