# strategies/ensemble_router.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from core.utils import import_from_path
from core.types import Side

class EnsembleRouter:
    """
    Orquestra as estratégias-filhas e retorna o melhor sinal (best_conf).
    Espera no config:
      strategy:
        class_path: strategies.ensemble_router.EnsembleRouter
        params:
          mode: "best_conf"
          children:
            - { class_path: "strategies.vwap_keltner.VWAPKeltner", params: {...} }
            - ...
    """

    def __init__(self, mode: str = "best_conf", children: Optional[List[Dict[str, Any]]] = None, ml_model=None, **_):
        self.mode = mode
        self.ml_model = ml_model
        self.children = []
        for ch in (children or []):
            Cls = import_from_path(ch["class_path"])
            params = ch.get("params", {})
            self.children.append(Cls(ml_model=ml_model, **params))

    def _pick_best(self, signals: List[SimpleNamespace]) -> Optional[SimpleNamespace]:
        if not signals:
            return None
        # maior confiança leva; empate: fica com o primeiro
        signals = sorted(signals, key=lambda s: float(getattr(s, "confidence", 0.0)), reverse=True)
        return signals[0]

    def generate_signal(self, symbol, df_exec, df_regime):
        sigs = []
        for strat in self.children:
            try:
                s = strat.generate_signal(symbol, df_exec, df_regime)
                if s is not None:
                    sigs.append(s)
            except Exception:
                # evitar que uma falha numa filha derrube o conjunto
                continue

        if self.mode == "best_conf":
            return self._pick_best(sigs)
        # fallback simples: primeira que aparecer
        return sigs[0] if sigs else None
