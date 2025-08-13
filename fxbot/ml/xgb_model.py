from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib
from .model_base import MLModel

class XGBModel(MLModel):
    def __init__(self, path="models/xgb.pkl", min_prob: float | None = None):
        base = Path(__file__).resolve().parents[1]  # .../fxbot
        p = Path(path) if Path(path).is_absolute() else (base / path)
        if not p.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {p}")
        blob = joblib.load(p)
        self.model = blob["model"]
        self.features = blob.get("feature_names", None)
        self.threshold = blob.get("threshold", 0.5)
        # permite forçar um limiar diferente
        if min_prob is not None:
            self.threshold = float(min_prob)

    def predict(self, symbol: str, features: dict) -> float:
        """
        Retorna probabilidade calibrada de sucesso (hit TP antes de SL).
        O executor usa essa probabilidade como 'confidence'.
        """
        if self.features is None:
            # fallback para 2 features legadas
            x = np.array([[features.get("ema_slope",0.0), features.get("atr",0.0)]], dtype=float)
        else:
            x = np.array([[features.get(k, 0.0) for k in self.features]], dtype=float)
        p = float(self.model.predict_proba(x)[0,1])
        return max(0.0, min(1.0, p))
