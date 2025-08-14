from __future__ import annotations
from typing import Optional, List, Dict, Any
import importlib
import traceback

from core.types import Signal


def _load_class(path: str):
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(f"{mod}"), cls)


class EnsembleRouter:
    """
    Agrega várias estratégias.
    mode:
      - "best_conf": escolhe o Signal de maior confidence
      - "first": primeira que gerar sinal, na ordem de 'children'
    children: lista de { class_path: "...", params: {...} }
    """

    def __init__(self, **params):
        self.mode: str = params.get("mode", "best_conf")
        self.children_cfg: List[Dict[str, Any]] = params.get("children", [])
        self.strategies = []
        for ch in self.children_cfg:
            cls = _load_class(ch["class_path"])
            inst = cls(**(ch.get("params", {})))
            self.strategies.append(inst)

    def generate_signal(self, symbol, df_e, df_r) -> Optional[Signal]:
        best: Optional[Signal] = None
        for s in self.strategies:
            try:
                sig = s.generate_signal(symbol, df_e, df_r)
            except Exception:
                traceback.print_exc()
                continue
            if sig is None:
                continue
            sig.meta = sig.meta or {}
            sig.meta.setdefault("strategy", s.__class__.__name__)
            if self.mode == "first":
                return sig
            if (best is None) or (sig.confidence > best.confidence):
                best = sig
        return best
