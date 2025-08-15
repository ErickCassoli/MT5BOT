# strategies/ensemble_router.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from core.utils import import_from_path
from core.types import Side
from core.logging import get_logger


class EnsembleRouter:
    """
    Orquestra as estratégias-filhas e retorna um único sinal.

    Config esperado no YAML:
      strategy:
        class_path: "strategies.ensemble_router.EnsembleRouter"
        params:
          mode: "best_conf"        # "best_conf" | "vote" | "first"
          min_conf: 0.0            # filtra sinais abaixo deste valor
          child_weights:           # (opcional) pesos por classe
            VWAPKeltner: 1.0
            DonchianADX: 1.0
            BreakoutVolume: 1.0
            ScalperRSIBB: 1.0
          prefer: ["BreakoutVolume","DonchianADX","VWAPKeltner","ScalperRSIBB"]  # desempate

          children:
            - class_path: "strategies.vwap_keltner.VWAPKeltner"
              params: {...}
            - class_path: "strategies.donchian_adx.DonchianADX"
              params: {...}
            - class_path: "strategies.breakout_volume.BreakoutVolume"
              params: {...}
            # - class_path: "strategies.scalper_rsi_bb.ScalperRSIBB"
            #   params: {...}
    """

    def __init__(
        self,
        mode: str = "best_conf",
        children: Optional[List[Dict[str, Any]]] = None,
        ml_model=None,
        min_conf: float = 0.0,
        child_weights: Optional[Dict[str, float]] = None,
        prefer: Optional[List[str]] = None,
        **_,
    ):
        self.log = get_logger(__name__)
        self.mode = (mode or "best_conf").lower()
        self.ml_model = ml_model
        self.min_conf = float(min_conf)
        self.child_weights = {**(child_weights or {})}
        self.prefer = list(prefer or [])
        self.children = []

        for ch in (children or []):
            Cls = import_from_path(ch["class_path"])
            params = ch.get("params", {}) or {}
            inst = Cls(ml_model=ml_model, **params)
            self.children.append(inst)

        if not self.children:
            self.log.warning("[Ensemble] sem estratégias-filhas configuradas.")

    # ------------- helpers -------------
    @staticmethod
    def _child_name(obj: Any) -> str:
        return getattr(obj, "__name__", None) or obj.__class__.__name__

    def _weight(self, child_obj: Any) -> float:
        return float(self.child_weights.get(self._child_name(child_obj), 1.0))

    def _annotate(self, sig: SimpleNamespace, child_obj: Any, n_candidates: int) -> SimpleNamespace:
        # garante dict meta
        if not hasattr(sig, "meta") or sig.meta is None:
            sig.meta = {}
        ens = {
            "mode": self.mode,
            "child": self._child_name(child_obj),
            "candidates": n_candidates,
            "min_conf": self.min_conf,
        }
        # anexa (sem sobrescrever se já existir)
        if "ensemble" not in sig.meta:
            sig.meta["ensemble"] = ens
        else:
            sig.meta["ensemble"].update(ens)
        # mantém a confiança original do filho
        return sig

    # ------------- pickers -------------
    def _pick_best_conf(self, cands: List[Tuple[SimpleNamespace, Any]]) -> Optional[Tuple[SimpleNamespace, Any]]:
        if not cands:
            return None
        # ordena por (conf * weight), depois por ordem de preferência (se fornecida)
        def _key(item):
            sig, child = item
            w = self._weight(child)
            conf = float(getattr(sig, "confidence", 0.0))
            pref_idx = self.prefer.index(self._child_name(child)) if self._child_name(child) in self.prefer else 1_000
            return (conf * w, -1.0 / (1 + pref_idx))  # maior conf*w primeiro; prefer mais cedo

        cands_sorted = sorted(cands, key=_key, reverse=True)
        return cands_sorted[0]

    def _pick_vote(self, cands: List[Tuple[SimpleNamespace, Any]]) -> Optional[Tuple[SimpleNamespace, Any]]:
        if not cands:
            return None
        # soma pesos por lado (BUY/SELL) e pega o lado vencedor; dentro do lado, usa best_conf
        side_buckets: Dict[Side, List[Tuple[SimpleNamespace, Any]]] = defaultdict(list)
        side_score: Dict[Side, float] = defaultdict(float)

        for sig, child in cands:
            side = getattr(sig, "side", None)
            if side not in (Side.BUY, Side.SELL):
                continue
            w = self._weight(child)
            conf = float(getattr(sig, "confidence", 0.0))
            side_buckets[side].append((sig, child))
            side_score[side] += w * conf

        if not side_buckets:
            return None

        # escolhe o lado com maior soma de w*conf (empate -> BUY por convenção)
        winner = max(side_score.items(), key=lambda kv: (kv[1], 1 if kv[0] == Side.BUY else 0))[0]
        # dentro do lado, reaplica best_conf
        return self._pick_best_conf(side_buckets[winner])

    # ------------- core -------------
    def generate_signal(self, symbol, df_exec, df_regime):
        # coleta sinais das filhas
        raw_cands: List[Tuple[SimpleNamespace, Any]] = []
        for strat in self.children:
            try:
                s = strat.generate_signal(symbol, df_exec, df_regime)
                if s is None:
                    continue
                # filtra por min_conf local (executor ainda aplica thr global do ML)
                if float(getattr(s, "confidence", 0.0)) < self.min_conf:
                    continue
                raw_cands.append((s, strat))
            except Exception:
                # não derruba o ensemble se uma filha falhar
                continue

        if not raw_cands:
            return None

        # escolhe conforme o modo
        picked: Optional[Tuple[SimpleNamespace, Any]]
        if self.mode == "first":
            picked = raw_cands[0]
        elif self.mode == "vote":
            picked = self._pick_vote(raw_cands)
        else:  # default: best_conf
            picked = self._pick_best_conf(raw_cands)

        if picked is None:
            return None

        sig, child = picked
        return self._annotate(sig, child, n_candidates=len(raw_cands))
