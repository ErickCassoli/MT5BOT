# strategies/donchian_adx.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from core.types import Side
from core.utils import atr as _atr, adx as _adx, donchian as _donchian, ema as _ema


class DonchianADX:
    """
    Trend breakout:
      - Rompimento dos últimos N (default 16) com ADX(H1) >= adx_thr
      - Filtro de médias para BUY: EMA20 > EMA50 > EMA200; SELL: inverso
      - Exige margem mínima de rompimento: min_break_atr * ATR (fechamento ou pavio)
    Mesmas saídas (SL/TP/partial/trailing) geridas fora.
    """

    def __init__(
        self,
        donchian: int = 16,
        adx_thr: int = 18,
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        allow_close_break: bool = True,
        min_bars: int = 150,
        min_break_atr: float = 0.10,  # margem mínima além da borda (em ATR)
        ml_model=None,
        **__,
    ):
        self.n = int(donchian)
        self.adx_thr = int(adx_thr)
        self.ema_f = int(ema_fast)
        self.ema_m = int(ema_mid)
        self.ema_s = int(ema_slow)
        self.allow_close_break = bool(allow_close_break)
        self.min_bars = int(min_bars)
        self.min_break_atr = float(min_break_atr)
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
        if len(df_e) < max(self.min_bars, self.ema_s + 5) or len(df_r) < 50:
            return None

        # Regime H1
        adx_h1 = float(_adx(df_r["h"], df_r["l"], df_r["c"], 14).iloc[-1])
        if adx_h1 < self.adx_thr:
            return None

        # M5 indicadores
        a = _atr(df_e["h"], df_e["l"], df_e["c"], 14)
        atr_now = float(a.iloc[-1])

        up, lo = _donchian(df_e["h"], df_e["l"], self.n)
        up0, lo0 = float(up.iloc[-1]), float(lo.iloc[-1])

        c0 = float(df_e["c"].iloc[-1])
        h0 = float(df_e["h"].iloc[-1])
        l0 = float(df_e["l"].iloc[-1])

        ema20 = float(_ema(df_e["c"], self.ema_f).iloc[-1])
        ema50 = float(_ema(df_e["c"], self.ema_m).iloc[-1])
        ema200 = float(_ema(df_e["c"], self.ema_s).iloc[-1])

        buy_trend = (ema20 > ema50) and (ema50 > ema200)
        sell_trend = (ema20 < ema50) and (ema50 < ema200)

        # Margem de rompimento requerida (em preço)
        margin = self.min_break_atr * max(atr_now, 1e-9)

        # Meta para logs/diagnóstico
        meta = {
            "strategy": "DonchianADX",
            "adx_h1": adx_h1,
            "dist_up": max(0.0, up0 - c0),
            "dist_low": max(0.0, c0 - lo0),
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "break_margin_atr": self.min_break_atr,
        }

        # --- BUY breakout ---
        if buy_trend:
            if self.allow_close_break:
                # precisa FECHAR acima da borda com margem
                break_ok = (c0 > up0) and ((c0 - up0) >= margin)
                break_dist_atr = (c0 - up0) / max(atr_now, 1e-9)
            else:
                # aceita PAVIO acima com a mesma margem
                break_ok = (h0 > up0) and ((h0 - up0) >= margin)
                break_dist_atr = (h0 - up0) / max(atr_now, 1e-9)

            if break_ok:
                feats = {
                    "adx_h1": adx_h1,
                    "ema20_50": ema20 - ema50,
                    "ema50_200": ema50 - ema200,
                    "break_dist": break_dist_atr,
                    "atr_now": atr_now,
                }
                conf = self._ml_conf(feats)
                if conf is None:
                    conf = 0.65
                meta["break_dist_atr"] = break_dist_atr
                return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

        # --- SELL breakout ---
        if sell_trend:
            if self.allow_close_break:
                break_ok = (c0 < lo0) and ((lo0 - c0) >= margin)
                break_dist_atr = (lo0 - c0) / max(atr_now, 1e-9)
            else:
                break_ok = (l0 < lo0) and ((lo0 - l0) >= margin)
                break_dist_atr = (lo0 - l0) / max(atr_now, 1e-9)

            if break_ok:
                feats = {
                    "adx_h1": adx_h1,
                    "rsi_m5": 0.0,
                    "atr_now": atr_now,

                    "c_kdist_up": 0.0,
                    "c_kdist_low": 0.0,
                    "near_vwap": 0.0,
                    "confirm_ema20": 0.0,

                    "room_atr": 0.0,
                    "break_dist": (c0 - up0) / max(atr_now, 1e-9) if (buy_trend and c0 >= up0)
                                else (lo0 - c0) / max(atr_now, 1e-9),

                    "ema20_50": (ema20 - ema50),
                    "ema50_200": (ema50 - ema200),

                    "vol_ratio": 0.0,

                    "src_vk": 0.0, "src_don": 1.0, "src_bv": 0.0,
                }


                conf = self._ml_conf(feats)
                if conf is None:
                    conf = 0.63
                meta["break_dist_atr"] = break_dist_atr
                return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        return None
