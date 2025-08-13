from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from core.types import Signal, Side
from core.utils import atr, adx, ema


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1.0/n, adjust=False).mean()
    loss = dn.ewm(alpha=1.0/n, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    up = mid + k * std
    lo = mid - k * std
    return mid, up, lo


class ScalperRSIBB:
    """
    Mean-reversion curto: RSI + Bollinger. Evita tendência forte com ADX(H1) baixo.
    """

    def __init__(self, **params):
        self.rsi_len: int = int(params.get("rsi_len", 14))
        self.rsi_buy: float = float(params.get("rsi_buy", 30))
        self.rsi_sell: float = float(params.get("rsi_sell", 70))
        self.bb_len: int = int(params.get("bb_len", 20))
        self.bb_k: float = float(params.get("bb_k", 2.0))
        self.max_adx_h1: float = float(params.get(
            "max_adx_h1", 18.0))  # só operar em regime fraco
        self.min_bars: int = int(params.get("min_bars", 120))

    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame) -> Optional[Signal]:
        if len(df_e) < max(self.min_bars, self.bb_len + self.rsi_len + 5) or len(df_r) < 100:
            return None

        # Evita operar contra tendências fortes
        adx_h1 = adx(df_r["h"], df_r["l"], df_r["c"], 14)
        if float(adx_h1.iloc[-1]) > self.max_adx_h1:
            return None

        a = atr(df_e["h"], df_e["l"], df_e["c"], 14)
        a0 = float(a.iloc[-1])
        if a0 <= 0 or not np.isfinite(a0):
            return None

        mid, up, lo = bollinger(df_e["c"], self.bb_len, self.bb_k)
        r = rsi(df_e["c"], self.rsi_len)

        c0 = float(df_e["c"].iloc[-1])
        up0 = float(up.iloc[-1])
        lo0 = float(lo.iloc[-1])
        r0 = float(r.iloc[-1])

        long_setup = (c0 < lo0) and (r0 <= self.rsi_buy)
        short_setup = (c0 > up0) and (r0 >= self.rsi_sell)
        if not (long_setup or short_setup):
            return None

        side = Side.BUY if long_setup else Side.SELL

        # confiança: quão extremo está o RSI + quão fora da banda em ATR
        dist_band = (lo0 - c0) if side == Side.BUY else (c0 - up0)
        edge = dist_band / max(a0, 1e-9)
        r_ext = (self.rsi_buy - r0) if side == Side.BUY else (r0 - self.rsi_sell)
        conf = 0.56 + min(0.12, 0.06 * max(0.0, r_ext) /
                          10.0) + min(0.12, 0.06 * edge)
        conf = float(min(0.92, max(0.50, conf)))

        meta: Dict[str, Any] = {
            "rsi": r0,
            "bb_len": self.bb_len,
            "bb_k": self.bb_k,
            "adx_h1": float(adx_h1.iloc[-1]),
        }
        return Signal(symbol=symbol, side=side, atr=a0, confidence=conf, meta=meta)
