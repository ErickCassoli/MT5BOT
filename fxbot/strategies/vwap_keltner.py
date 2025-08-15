# strategies/vwap_keltner.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from core.types import Side
from core.utils import atr as _atr, adx as _adx
from core.indicators import rsi as _rsi, vwap_session, keltner_channels

class VWAPKeltner:
    """
    Tendência/pullback com VWAP + Keltner no M5, regime por ADX(H1).
    Regras:
      - BUY: preço acima do VWAP e rompendo/segurando entre banda média e upper;
              ADX(H1) >= adx_thr; RSI(M5) > rsi_trig
      - SELL: simétrico abaixo do VWAP, RSI < (100 - rsi_trig) se quiser espelhar; aqui usamos < 50
      - Entrada: breakout da banda OU pullback à média/VWAP com rejeição (close volta p/ dentro)
    """

    def __init__(
        self,
        k_ema_len: int = 20,
        k_atr_len: int = 10,
        k_mult: float = 1.8,
        adx_thr: int = 14,
        rsi_len: int = 14,
        rsi_trig: int = 50,
        near_vwap_by_atr: float = 0.35,
        confirm_ema20: bool = True,
        allow_break_close: bool = True,
        min_bars: int = 150,
        donchian: int = 16,  # mantido para logs/compat
        ml_model=None,
        **__,
    ):
        self.k_ema_len = k_ema_len
        self.k_atr_len = k_atr_len
        self.k_mult = float(k_mult)
        self.adx_thr = int(adx_thr)
        self.rsi_len = rsi_len
        self.rsi_trig = rsi_trig
        self.near_vwap_by_atr = float(near_vwap_by_atr)
        self.confirm_ema20 = bool(confirm_ema20)
        self.allow_break_close = bool(allow_break_close)
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
                # scikit-like: coluna 1 é prob de classe positiva
                if hasattr(proba, "shape") and proba.shape[1] >= 2:
                    return float(proba[0, 1])
                return float(proba[0])
            if hasattr(self.ml, "predict"):
                return float(self.ml.predict([feats])[0])
        except Exception:
            return None
        return None

    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame):
        if len(df_e) < max(self.min_bars, 50) or len(df_r) < 50:
            return None

        # Regime H1
        adx_h1 = float(_adx(df_r["h"], df_r["l"], df_r["c"], 14).iloc[-1])
        if adx_h1 < self.adx_thr:
            return None

        # Indicadores M5
        a = _atr(df_e["h"], df_e["l"], df_e["c"], self.k_atr_len)
        atr_now = float(a.iloc[-1])

        vwap = vwap_session(df_e)
        ku, km, kl = keltner_channels(df_e["c"], a, ema_len=self.k_ema_len, mult=self.k_mult)
        r = _rsi(df_e["c"], self.rsi_len)
        rsi_now = float(r.iloc[-1])

        c0 = float(df_e["c"].iloc[-1])
        vwap0 = float(vwap.iloc[-1])
        ku0, km0, kl0 = float(ku.iloc[-1]), float(km.iloc[-1]), float(kl.iloc[-1])

        # Condições de tendência
        above_vwap = c0 > vwap0
        below_vwap = c0 < vwap0
        in_upper_band = (c0 >= km0) and (c0 <= ku0)
        in_lower_band = (c0 <= km0) and (c0 >= kl0)
        near_vwap = abs(c0 - vwap0) <= self.near_vwap_by_atr * atr_now

        # Regras de entrada
        buy_break = above_vwap and (c0 > ku0)  # breakout Keltner superior
        buy_pull = above_vwap and in_upper_band and near_vwap  # pullback com rejeição perto da média/VWAP
        sell_break = below_vwap and (c0 < kl0)
        sell_pull = below_vwap and in_lower_band and near_vwap

        meta = {
            "strategy": "VWAPKeltner",
            "adx_h1": adx_h1,
            "near_thr": self.near_vwap_by_atr * atr_now,
            "dist_up": max(0.0, ku0 - c0),
            "dist_low": max(0.0, c0 - kl0),
            "vwap": vwap0,
            "k_upper": ku0,
            "k_mid": km0,
            "k_lower": kl0,
            "rsi": rsi_now,
        }

        # Filtros de RSI
        if (buy_break or buy_pull) and rsi_now > max(50, self.rsi_trig):
            feats = {
                "adx_h1": adx_h1,
                "rsi_m5": rsi_now,
                "c_kdist_up": (c0 - km0) / max(atr_now, 1e-9),
                "near_vwap": float(near_vwap),
                "atr_now": atr_now,
            }
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.62  # confiança técnica padrão
            return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

        if (sell_break or sell_pull) and rsi_now < 50:
            feats = {
                "adx_h1": adx_h1,
                "rsi_m5": rsi_now,
                "c_kdist_low": (km0 - c0) / max(atr_now, 1e-9),
                "near_vwap": float(near_vwap),
                "atr_now": atr_now,
            }
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.60
            return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        return None
