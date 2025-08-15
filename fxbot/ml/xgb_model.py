# fxbot/ml/xgb_model.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import joblib

from .model_base import MLModel
from core.logging import get_logger


class XGBModel(MLModel):
    """
    Wrapper para o modelo calibrado salvo pelo train_xgb.
    Espera um pickle (joblib) com as chaves:
      - "model": estimador com .predict_proba (ex.: CalibratedClassifierCV)
      - "feature_names": List[str] (ou "features" em payloads antigos)
      - "threshold": float (limiar sugerido por EV; pode ser sobrescrito por min_prob)
      - "meta": dict opcional (ex.: {"features_version": "v5", ...})

    O método principal para uso nas estratégias é `predict_proba_dict(features: dict)`,
    que aceita um dicionário parcial e preenche faltantes com 0.0 na ordem correta.
    """

    def __init__(self, path: str = "models/xgb_vkbp.pkl", min_prob: float | None = None):
        self.log = get_logger(__name__)

        # Resolve caminho relativo ao pacote fxbot/
        base = Path(__file__).resolve().parents[1]  # .../fxbot
        p = Path(path) if Path(path).is_absolute() else (base / path)
        if not p.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {p}")

        # Carrega payload
        blob: Dict[str, Any] = joblib.load(p)

        self.model = blob.get("model")
        if self.model is None:
            raise ValueError(f"Payload inválido (sem chave 'model'): {p}")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Modelo carregado não possui método predict_proba().")

        # Lista de features (ordem é crítica!)
        feats = blob.get("feature_names", blob.get("features"))
        if feats is None:
            raise ValueError("Payload sem lista de features: 'feature_names' (ou 'features').")
        self.features: List[str] = list(map(str, feats))

        # Limiar padrão salvo no treino (pode ser sobrescrito via min_prob)
        self.threshold: float = float(blob.get("threshold", 0.5))
        if min_prob is not None:
            self.threshold = float(min_prob)

        # Metadados (ex.: versão do schema de features)
        self.meta: Dict[str, Any] = blob.get("meta", {}) or {}
        self.features_version: str = str(self.meta.get("features_version", "unknown"))

        # Flag interna para não spammar mismatch de schema
        self._warned_schema_mismatch = False

        self.log.info(
            f"[ML] XGBModel carregado | feats={len(self.features)} "
            f"thr={self.threshold:.3f} ver={self.features_version}"
        )

    # ---------------- utils internos ----------------

    @staticmethod
    def _safe_float(x: Any) -> float:
        """Converte para float, clampando NaN/inf e erros para 0.0."""
        try:
            v = float(x)
            if not np.isfinite(v):
                return 0.0
            return v
        except Exception:
            return 0.0

    def _vectorize(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Constrói o vetor X[1, n_features] na ordem do treino.
        - Campos ausentes são preenchidos com 0.0 (e loga um warning apenas 1x).
        - Campos extras são ignorados (e aparecem no warning apenas 1x).
        """
        missing = [name for name in self.features if name not in features]
        extra = [k for k in features.keys() if k not in self.features]

        if (missing or extra) and (not self._warned_schema_mismatch):
            # Loga só uma vez por instância para não poluir o console
            miss_str = f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
            extra_str = f"{extra[:8]}{'...' if len(extra) > 8 else ''}"
            self.log.warning(f"[ML] Mismatch de schema de features: faltando={miss_str} extras={extra_str}")
            self._warned_schema_mismatch = True

        vec = [self._safe_float(features.get(name, 0.0)) for name in self.features]
        x = np.asarray([vec], dtype=float)
        return x

    # ---------------- API pública ----------------

    def predict_proba_dict(self, features: Dict[str, Any]) -> float:
        """
        Retorna P(y=1) calibrada a partir de um dict {feature: valor}.
        Aceita dicionários parciais e preenche faltantes com 0.0.
        Retorno clampado em [0, 1].
        """
        x = self._vectorize(features)
        try:
            with np.errstate(all="ignore"):
                p = float(self.model.predict_proba(x)[0, 1])
        except Exception as e:
            # Fallback defensivo para não quebrar execução ao vivo
            self.log.exception(f"[ML] predict_proba falhou: {e}")
            p = 0.5
        # clamp
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        return p

    # Compatibilidade com caminhos antigos onde chamamos .predict(...)
    def predict(self, symbol: str, features: Dict[str, Any]) -> float:
        return self.predict_proba_dict(features)

    # Auxiliares (opcionais)

    @property
    def feature_names(self) -> List[str]:
        """Retorna a lista de nomes de features na ordem de treino."""
        return list(self.features)

    @property
    def version(self) -> str:
        """Versão declarada do schema de features (ex.: 'v5')."""
        return self.features_version
