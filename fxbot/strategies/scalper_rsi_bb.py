# strategies/scalper_rsi_bb.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from core.types import Side
from core.utils import atr as _atr, adx as _adx
from core.indicators import rsi as _rsi, bollinger as _bb

class ScalperRSIBB:
    """
    Reversão curta para mercado lateral:
      - Regime: ADX(H1) <= max_adx_h1 (default 18)
      - BUY: RSI < 30 e toque/fechamento fora da banda inferior (BB 20/2)
      - SELL: RSI > 70 e fora da banda superior
      - SL ≈ 1×ATR, TP ≈ 2×ATR (Executor já cuida de SL/TP padrão; aqui só emitimos o sinal)
    """

    def __init__(
        self,
        rsi_len: int = 14,
        rsi_buy: int = 30,
        rsi_sell: int = 70,
        bb_len: int = 20,
        bb_k: float = 2.0,
        max_adx_h1: int = 18,
        min_bars: int = 120,
        ml_model=None,
        **__,
    ):
        self.rsi_len = rsi_len
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.bb_len = bb_len
        self.bb_k = float(bb_k)
        self.max_adx_h1 = int(max_adx_h1)
        self.min_bars = int(min_bars)
        self.ml = ml_model

    def _ml_conf(self, feats: dict) -> Optional[float]:
        if self.ml is None:
            return None
        try:
            import pandas as pd
            if hasattr(self.ml, "predict_proba_dict"):
                return float(self.ml.predict_proba_dict(feats))
            if hasattr(self.ml, "predict_proba"):
                X = pd.DataFrame([feats])
                proba = self.ml.predict_proba(X)
                if hasattr(proba, "shape") and proba.shape[1] >= 2:
                    return float(proba[0, 1])
                return float(proba[0])
            if hasattr(self.ml, "predict"):
                return float(self.ml.predict([feats])[0])
        except Exception:
            return None
        return None

    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame):
        if len(df_e) < max(self.min_bars, self.bb_len + 5) or len(df_r) < 50:
            return None

        # Regime lateral
        adx_h1 = float(_adx(df_r["h"], df_r["l"], df_r["c"], 14).iloc[-1])
        if adx_h1 > self.max_adx_h1:
            return None

        c = df_e["c"].astype(float)
        a = _atr(df_e["h"], df_e["l"], c, 14)
        atr_now = float(a.iloc[-1])

        bb_up, bb_mid, bb_lo = _bb(c, self.bb_len, self.bb_k)
        r = _rsi(c, self.rsi_len)
        r0 = float(r.iloc[-1])
        c0 = float(c.iloc[-1])
        up0, mid0, lo0 = float(bb_up.iloc[-1]), float(bb_mid.iloc[-1]), float(bb_lo.iloc[-1])

        meta = {
            "strategy": "ScalperRSIBB",
            "adx_h1": adx_h1,
            "rsi": r0,
            "bb_up": up0, "bb_mid": mid0, "bb_lo": lo0,
        }

        # BUY: RSI baixo + fora/na borda inferior
        if (r0 <= self.rsi_buy) and (c0 <= lo0):
            feats = {"adx_h1": adx_h1, "rsi": r0, "dist_lo": (lo0 - c0) / max(atr_now, 1e-9), "atr_now": atr_now}
            conf = self._ml_conf(feats) or 0.60
            return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

        # SELL: RSI alto + fora/na borda superior
        if (r0 >= self.rsi_sell) and (c0 >= up0):
            feats = {"adx_h1": adx_h1, "rsi": r0, "dist_up": (c0 - up0) / max(atr_now, 1e-9), "atr_now": atr_now}
            conf = self._ml_conf(feats) or 0.60
            return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        return None
