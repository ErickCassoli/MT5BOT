# strategies/breakout_volume.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from core.types import Side
from core.utils import atr as _atr, adx as _adx

class BreakoutVolume:
    """
    Expansão com volume:
      - Rompe SR dos últimos N (sr_win), com volume > vol_mult * média(vol_lookback)
      - Regime: ADX(H1) >= adx_thr
      - Permite retest (close volta pra faixa e sai novamente)
    """

    def __init__(
        self,
        sr_win: int = 24,
        vol_lookback: int = 20,
        vol_mult: float = 1.6,
        adx_thr: int = 12,
        allow_retest: bool = True,
        min_bars: int = 200,
        ml_model=None,
        **__,
    ):
        self.sr_win = int(sr_win)
        self.vol_lookback = int(vol_lookback)
        self.vol_mult = float(vol_mult)
        self.adx_thr = int(adx_thr)
        self.allow_retest = bool(allow_retest)
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
        if len(df_e) < max(self.min_bars, self.sr_win + 5) or len(df_r) < 50:
            return None

        # Regime
        adx_h1 = float(_adx(df_r["h"], df_r["l"], df_r["c"], 14).iloc[-1])
        if adx_h1 < self.adx_thr:
            return None

        # SR + volume
        c = df_e["c"].astype(float)
        h = df_e["h"].astype(float)
        l = df_e["l"].astype(float)
        v = df_e["v"].astype(float)

        atr_now = float(_atr(h, l, c, 14).iloc[-1])
        sr_high = h.rolling(self.sr_win, min_periods=self.sr_win).max().iloc[-2]  # ultimo SR completo
        sr_low = l.rolling(self.sr_win, min_periods=self.sr_win).min().iloc[-2]
        vol_avg = v.rolling(self.vol_lookback, min_periods=self.vol_lookback).mean().iloc[-1]
        vol_now = float(v.iloc[-1])

        c0 = float(c.iloc[-1])
        broke_up = c0 > sr_high and (vol_now >= self.vol_mult * vol_avg if vol_avg > 0 else False)
        broke_down = c0 < sr_low and (vol_now >= self.vol_mult * vol_avg if vol_avg > 0 else False)

        meta = {
            "strategy": "BreakoutVolume",
            "adx_h1": adx_h1,
            "sr_high": float(sr_high),
            "sr_low": float(sr_low),
            "vol_now": float(vol_now),
            "vol_avg": float(vol_avg),
        }

        if broke_up:
            feats = {
                "adx_h1": adx_h1,
                "vol_ratio": (vol_now / max(vol_avg, 1e-9)),
                "break_dist": (c0 - sr_high) / max(atr_now, 1e-9),
                "atr_now": atr_now,
            }
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.61
            return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

        if broke_down:
            feats = {
                "adx_h1": adx_h1,
                "vol_ratio": (vol_now / max(vol_avg, 1e-9)),
                "break_dist": (sr_low - c0) / max(atr_now, 1e-9),
                "atr_now": atr_now,
            }
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.60
            return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        # Retest (opcional): último candle rompeu e o atual confirma
        if self.allow_retest and len(df_e) >= self.sr_win + 3:
            c1 = float(c.iloc[-2])
            if c1 > sr_high and c0 > c1:
                conf = self._ml_conf({"adx_h1": adx_h1, "retest": 1.0, "atr_now": atr_now}) or 0.58
                return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)
            if c1 < sr_low and c0 < c1:
                conf = self._ml_conf({"adx_h1": adx_h1, "retest": 1.0, "atr_now": atr_now}) or 0.58
                return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        return None
