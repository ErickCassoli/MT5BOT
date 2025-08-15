# core/indicators.py
from __future__ import annotations
import numpy as np
import pandas as pd

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    close = pd.Series(close).astype(float)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def bollinger(close: pd.Series, length: int = 20, k: float = 2.0):
    close = pd.Series(close).astype(float)
    ma = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return upper, ma, lower

def vwap_session(df: pd.DataFrame) -> pd.Series:
    """
    VWAP reinicia a cada dia (baseado em df['time']). Usa preço típico
    e volume cumulativo do dia. Retorna série alinhada ao df, com ffill.
    """
    if "time" not in df.columns:
        raise ValueError("df precisa ter coluna 'time'")
    if "v" not in df.columns:
        raise ValueError("df precisa ter coluna 'v' (tick_volume)")

    # price típico
    tp = (df["h"] + df["l"] + df["c"]) / 3.0

    # agrupa por dia mantendo dtype datetime64 (melhor que .dt.date)
    day = df["time"].dt.normalize()

    # cumulativos por dia
    pv = (tp * df["v"]).groupby(day).cumsum()
    vv = df["v"].groupby(day).cumsum()

    # VWAP com proteção p/ vv=0 e forward-fill dos primeiros NaNs do dia
    vwap = pv.div(vv.replace(0, np.nan))
    return vwap.ffill()


def keltner_channels(close: pd.Series, atr: pd.Series, ema_len: int = 20, mult: float = 1.8):
    # EMA do close como linha central
    ema = close.ewm(span=ema_len, adjust=False).mean()
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, ema, lower
