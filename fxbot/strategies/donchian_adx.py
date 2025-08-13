from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from core.types import Signal, Side
from core.utils import ema, atr, adx, donchian


class DonchianADX:
    """
    Breakout de Donchian no M1/M5 filtrado por regime H1 (EMA50>EMA200 para buy, EMA50<EMA200 para sell) + ADX.
    Retorna Signal(symbol, side, atr, confidence, meta).
    """

    def __init__(self, **params):
        self.win: int = int(params.get("donchian", 16))
        self.adx_thr: float = float(params.get("adx_thr", 18))
        self.min_bars: int = int(params.get("min_bars", 150))
        self.ema_fast: int = int(params.get("ema_fast", 50))
        self.ema_slow: int = int(params.get("ema_slow", 200))
        self.boost_strong_adx: float = float(
            params.get("boost_strong_adx", 5.0))  # +5 sobre o thr
        self.allow_close_break: bool = bool(
            params.get("allow_close_break", True))

    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame) -> Optional[Signal]:
        if len(df_e) < max(self.min_bars, self.win + 5) or len(df_r) < max(self.ema_slow + 5, 210):
            return None

        # Regime H1
        ema_f = ema(df_r["c"], self.ema_fast)
        ema_s = ema(df_r["c"], self.ema_slow)
        adx_h1 = adx(df_r["h"], df_r["l"], df_r["c"], 14)
        uptrend = (df_r["c"].iloc[-1] > ema_s.iloc[-1]) and (ema_f.iloc[-1]
                                                             > ema_s.iloc[-1]) and (adx_h1.iloc[-1] >= self.adx_thr)
        downtrend = (df_r["c"].iloc[-1] < ema_s.iloc[-1]) and (ema_f.iloc[-1]
                                                               < ema_s.iloc[-1]) and (adx_h1.iloc[-1] >= self.adx_thr)
        if not (uptrend or downtrend):
            return None

        up, lo = donchian(df_e["h"], df_e["l"], self.win)
        a = atr(df_e["h"], df_e["l"], df_e["c"], 14)
        a0 = float(a.iloc[-1])
        c0 = float(df_e["c"].iloc[-1])
        if a0 <= 0 or not np.isfinite(a0):
            return None

        long_break = uptrend and (c0 > float(
            up.iloc[-1])) and self.allow_close_break
        short_break = downtrend and (c0 < float(
            lo.iloc[-1])) and self.allow_close_break
        if not (long_break or short_break):
            return None

        side = Side.BUY if long_break else Side.SELL
        conf = 0.58
        if float(adx_h1.iloc[-1]) >= self.adx_thr + self.boost_strong_adx:
            conf += 0.08
        # distância além do canal em ATR (quanto mais, mais convicção)
        edge = abs((c0 - float(up.iloc[-1])) if side ==
                   Side.BUY else (float(lo.iloc[-1]) - c0)) / max(a0, 1e-9)
        conf += float(min(0.10, 0.05 * edge))
        conf = float(min(0.95, max(0.50, conf)))

        meta: Dict[str, Any] = {
            "adx_h1": float(adx_h1.iloc[-1]),
            "dc_win": self.win,
        }
        return Signal(symbol=symbol, side=side, atr=a0, confidence=conf, meta=meta)
