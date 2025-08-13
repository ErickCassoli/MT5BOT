from __future__ import annotations
import pandas as pd
import numpy as np
from fxbot.core.utils import ema, atr, adx, donchian


def compute_features(df_exec: pd.DataFrame,
                     df_regime: pd.DataFrame,
                     donch_n: int = 16,
                     near_ratio: float = 0.10) -> dict:
    """
    Extrai features no último candle dos dataframes:
    df_exec: M5 com colunas ['o','h','l','c','v']
    df_regime: H1 com colunas ['o','h','l','c','v']
    """
    # Exec (M5)
    ema20 = ema(df_exec['c'], 20)
    atr_m5 = atr(df_exec['h'], df_exec['l'], df_exec['c'], 14)
    up, lo = donchian(df_exec['h'], df_exec['l'], donch_n)

    c0, h0, l0 = df_exec['c'].iloc[-1], df_exec['h'].iloc[-1], df_exec['l'].iloc[-1]
    ema20_0 = ema20.iloc[-1]
    atr0 = float(atr_m5.iloc[-1])
    up0, lo0 = float(up.iloc[-1]), float(lo.iloc[-1])

    dist_up = max(0.0, up0 - c0)
    dist_low = max(0.0, c0 - lo0)
    near_thr = near_ratio * max(atr0, 1e-12)
    near_up = dist_up <= near_thr
    near_low = dist_low <= near_thr

    buy_break_hl = h0 >= up0
    sell_break_hl = l0 <= lo0
    buy_close = c0 >= up0
    sell_close = c0 <= lo0

    # Regime (H1)
    ema50_h1 = ema(df_regime['c'], 50)
    ema200_h1 = ema(df_regime['c'], 200)
    adx_h1 = adx(df_regime['h'], df_regime['l'], df_regime['c'], 14)

    ema50_over_200 = float(ema50_h1.iloc[-1] - ema200_h1.iloc[-1])
    trend_up_h1 = (df_regime['c'].iloc[-1] > ema200_h1.iloc[-1]
                   ) and (ema50_h1.iloc[-1] > ema200_h1.iloc[-1])
    trend_dn_h1 = (df_regime['c'].iloc[-1] < ema200_h1.iloc[-1]
                   ) and (ema50_h1.iloc[-1] < ema200_h1.iloc[-1])
    adx_h1_val = float(adx_h1.iloc[-1])

    feats = {
        # Exec
        "c0": float(c0),
        "ema20_slope": float(ema20.diff().iloc[-1]),
        "atr_m5": float(atr0),
        "dist_up": float(dist_up),
        "dist_low": float(dist_low),
        "near_thr": float(near_thr),
        "frac_to_up": float(dist_up / (near_thr + 1e-12)),
        "frac_to_low": float(dist_low / (near_thr + 1e-12)),
        "above_ema20": float(c0 > ema20_0),
        "buy_break_hl": float(buy_break_hl),
        "sell_break_hl": float(sell_break_hl),
        "buy_close": float(buy_close),
        "sell_close": float(sell_close),
        # Regime
        "adx_h1": float(adx_h1_val),
        "ema50_over_200_h1": float(ema50_over_200),
        "trend_up_h1": float(trend_up_h1),
        "trend_dn_h1": float(trend_dn_h1),
        # Aux
        "donch_n": float(donch_n),
        "near_ratio": float(near_ratio),
    }
    return feats
