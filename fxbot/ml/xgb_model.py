# ml/xgb_model.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import joblib

from .model_base import MLModel
from core.logging import get_logger


class XGBModel(MLModel):
    """
    Wrapper para o XGB calibrado salvo pelo tools/train_xgb.py
    Espera um pickle com as chaves:
      - "model": CalibratedClassifierCV (isotônico ou sigmoid, prefit)
      - "feature_names": List[str]
      - "threshold": float (opcional; default 0.5)
      - "meta": dict (opcional; pode conter "features_version", etc.)
    """

    def __init__(self, path: str = "models/xgb_vkbp.pkl", min_prob: float | None = None):
        self.log = get_logger(__name__)
        base = Path(__file__).resolve().parents[1]  # .../fxbot
        p = Path(path) if Path(path).is_absolute() else (base / path)
        if not p.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {p}")

        blob = joblib.load(p)
        self.model = blob.get("model", None)
        if self.model is None:
            raise ValueError(f"Payload inválido: chave 'model' ausente em {p}")

        # compat: aceita "feature_names" ou "features"
        feats = blob.get("feature_names", blob.get("features"))
        if feats is None:
            raise ValueError("Payload sem lista de features ('feature_names').")
        self.features: List[str] = list(feats)

        # threshold salvo (por EV). Pode ser sobrescrito por min_prob
        self.threshold: float = float(blob.get("threshold", 0.5))
        if min_prob is not None:
            self.threshold = float(min_prob)

        self.meta: Dict[str, Any] = blob.get("meta", {}) or {}
        self.features_version: Optional[str] = self.meta.get("features_version")

        self._warned_schema_mismatch = False
        self.log.info(
            f"[ML] XGBModel carregado | feats={len(self.features)} thr={self.threshold:.3f} "
            f"ver={self.features_version}"
        )

    # -------- utils internos --------
    @staticmethod
    def _safe_float(x: Any) -> float:
        try:
            v = float(x)
            if not np.isfinite(v):
                return 0.0
            return v
        except Exception:
            return 0.0

    def _vectorize(self, features: Dict[str, Any]) -> np.ndarray:
        missing = [f for f in self.features if f not in features]
        extra = [k for k in features.keys() if k not in self.features]

        if (missing or extra) and (not self._warned_schema_mismatch):
            self.log.warning(
                "[ML] Mismatch de schema de features: "
                f"faltando={missing[:8]}{'...' if len(missing) > 8 else ''} "
                f"extras={extra[:8]}{'...' if len(extra) > 8 else ''}"
            )
            self._warned_schema_mismatch = True

        vec = [self._safe_float(features.get(name, 0.0)) for name in self.features]
        x = np.asarray([vec], dtype=float)
        return x

    # -------- API pública usada nas estratégias/executor --------
    def predict_proba_dict(self, features: Dict[str, Any]) -> float:
        """
        Retorna P(y=1) calibrada a partir de um dict {feature: valor}.
        Clampa para [0, 1].
        """
        x = self._vectorize(features)
        try:
            p = float(self.model.predict_proba(x)[0, 1])
        except Exception as e:
            # fallback defensivo
            self.log.exception(f"[ML] predict_proba falhou: {e}")
            p = 0.5
        return max(0.0, min(1.0, p))

    def predict(self, symbol: str, features: Dict[str, Any]) -> float:
        """
        Compat: retorna probabilidade calibrada (mesmo que predict_proba_dict).
        O Executor usa esse valor como 'confidence'.
        """
        return self.predict_proba_dict(features)
