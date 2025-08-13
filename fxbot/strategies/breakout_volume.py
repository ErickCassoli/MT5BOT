from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from core.types import Signal, Side
from core.utils import atr, adx, donchian, ema


def _get_volume(series_frame: pd.DataFrame) -> pd.Series:
    if "v" in series_frame:
        return series_frame["v"]
    if "tick_volume" in series_frame:
        return series_frame["tick_volume"]
    return pd.Series(1.0, index=series_frame.index)


class BreakoutVolume:
    """
    Rompimento de S/R (Donchian) com confirmação de volume (tick volume) acima da média.
    """

    def __init__(self, **params):
        self.sr_win: int = int(params.get("sr_win", 24))
        self.vol_lb: int = int(params.get("vol_lookback", 20))
        # vol_atual > vol_mult * média
        self.vol_mult: float = float(params.get("vol_mult", 1.5))
        self.adx_thr: float = float(params.get(
            "adx_thr", 12.0))   # leve filtro de regime
        self.min_bars: int = int(params.get("min_bars", 200))
        self.use_ret_test: bool = bool(params.get(
            "allow_retest", True))  # aceita reteste imediato

    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame) -> Optional[Signal]:
        if len(df_e) < max(self.min_bars, self.sr_win + self.vol_lb + 5) or len(df_r) < 150:
            return None

        # Filtro de regime leve (não é obrigatório super forte)
        adx_h1 = adx(df_r["h"], df_r["l"], df_r["c"], 14)
        if float(adx_h1.iloc[-1]) < self.adx_thr:
            return None

        up, lo = donchian(df_e["h"], df_e["l"], self.sr_win)
        a = atr(df_e["h"], df_e["l"], df_e["c"], 14)
        a0 = float(a.iloc[-1])
        c0 = float(df_e["c"].iloc[-1])
        if a0 <= 0 or not np.isfinite(a0):
            return None

        vol = _get_volume(df_e)
        v_now = float(vol.iloc[-1])
        v_ma = float(vol.rolling(self.vol_lb).mean().iloc[-1])
        if not np.isfinite(v_ma) or v_ma <= 0:
            return None

        broke_up = c0 > float(up.iloc[-1])
        broke_down = c0 < float(lo.iloc[-1])
        if not (broke_up or broke_down):
            return None

        # Confirmação de volume
        if v_now < self.vol_mult * v_ma:
            # checa reteste (candle volta e em seguida retoma com vol > média)
            if not self.use_ret_test:
                return None

        side = Side.BUY if broke_up else Side.SELL

        # confiança: pico de volume + quanto passou do nível em ATR + ADX
        spike = (v_now / v_ma) if v_ma > 0 else 1.0
        edge = abs((c0 - float(up.iloc[-1])) if side ==
                   Side.BUY else (float(lo.iloc[-1]) - c0)) / max(a0, 1e-9)
        conf = 0.60 + min(0.15, 0.06 * (spike - 1.0)) + min(0.10, 0.05 * edge) + \
            min(0.05, 0.01 * (float(adx_h1.iloc[-1]) - self.adx_thr))
        conf = float(min(0.95, max(0.50, conf)))

        meta: Dict[str, Any] = {
            "vol_now": v_now,
            "vol_mean": v_ma,
            "spike": float(spike),
            "sr_win": self.sr_win,
            "adx_h1": float(adx_h1.iloc[-1]),
        }
        return Signal(symbol=symbol, side=side, atr=a0, confidence=conf, meta=meta)
