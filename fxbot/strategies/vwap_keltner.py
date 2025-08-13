from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from core.types import Signal, Side
from core.utils import ema, atr, adx

# VWAP ancorado no dia (usa tick volume como proxy)


def vwap_session(df: pd.DataFrame) -> pd.Series:
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    vol = df["v"] if "v" in df.columns else pd.Series(1.0, index=df.index)
    day = pd.to_datetime(
        df["time"]).dt.date if "time" in df.columns else pd.Series(0, index=df.index)
    pv = (tp * vol).groupby(day).cumsum()
    vv = vol.groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def keltner(df: pd.DataFrame, ema_len: int, atr_len: int, mult: float):
    mid = ema(df["c"], ema_len)
    a = atr(df["h"], df["l"], df["c"], atr_len)
    up = mid + mult * a
    lo = mid - mult * a
    return mid, up, lo, a


class VWAPKeltner:
    """
    Estratégia VWAP + Keltner (breakout e pullback com filtro de regime em H1).
    Espera-se que o executor chame:
        generate_signal(symbol, df_exec(M5), df_regime(H1))
    Retorna:
        Signal(symbol=..., side=Side.BUY/SELL, atr=..., confidence=..., meta=dict)
    """

    def __init__(self, **params):
        # parâmetros com defaults seguros
        self.k_ema_len: int = int(params.get("k_ema_len", 20))
        self.k_atr_len: int = int(params.get("k_atr_len", 10))
        self.k_mult: float = float(params.get("k_mult", 1.8))

        self.adx_thr: float = float(params.get("adx_thr", 14.0))

        self.rsi_len: int = int(params.get("rsi_len", 14))
        self.rsi_trig: float = float(params.get("rsi_trig", 50))

        # quão “perto” da VWAP (em ATR M5) p/ validar pullback
        self.near_vwap_by_atr: float = float(
            params.get("near_vwap_by_atr", 0.35))

        # confirmações auxiliares
        self.confirm_ema20: bool = bool(params.get("confirm_ema20", True))
        self.allow_break_close: bool = bool(
            params.get("allow_break_close", True))
        self.min_bars: int = int(params.get("min_bars", 150))

        # opcional: janela donchian usada em logs/diagnóstico (não é o motor da VKBP)
        self.donchian_win: int = int(params.get("donchian", 14))

    @staticmethod
    def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0.0)
        dn = -delta.clip(upper=0.0)
        gain = up.ewm(alpha=1.0/n, adjust=False).mean()
        loss = dn.ewm(alpha=1.0/n, adjust=False).mean()
        rs = gain / (loss + 1e-12)
        return 100.0 - (100.0 / (1.0 + rs))

    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame) -> Optional[Signal]:
        # sanity
        if len(df_e) < max(self.min_bars, self.k_ema_len + self.k_atr_len + 10) or len(df_r) < 210:
            return None

        # ===== Regime (H1) =====
        ema50_h1 = ema(df_r["c"], 50)
        ema200_h1 = ema(df_r["c"], 200)
        adx_h1 = adx(df_r["h"], df_r["l"], df_r["c"], 14)
        uptrend = (df_r["c"].iloc[-1] > ema200_h1.iloc[-1]) and (ema50_h1.iloc[-1]
                                                                 > ema200_h1.iloc[-1]) and (adx_h1.iloc[-1] >= self.adx_thr)
        downtrend = (df_r["c"].iloc[-1] < ema200_h1.iloc[-1]) and (ema50_h1.iloc[-1]
                                                                   < ema200_h1.iloc[-1]) and (adx_h1.iloc[-1] >= self.adx_thr)
        if not (uptrend or downtrend):
            return None

        # ===== Exec (M5) =====
        mid, kup, klo, a = keltner(
            df_e, self.k_ema_len, self.k_atr_len, self.k_mult)
        a0 = float(a.iloc[-1])
        if not np.isfinite(a0) or a0 <= 0:
            return None

        vwap = vwap_session(df_e)
        rsi_ser = self._rsi(df_e["c"], self.rsi_len)
        ema20_m5 = ema(df_e["c"], 20)

        c0 = float(df_e["c"].iloc[-1])
        up0 = float(kup.iloc[-1])
        lo0 = float(klo.iloc[-1])
        vwap0 = float(vwap.iloc[-1])
        rsi0 = float(rsi_ser.iloc[-1])
        ema20_ok = (c0 > float(
            ema20_m5.iloc[-1])) if uptrend else (c0 < float(ema20_m5.iloc[-1]))

        near_thr = self.near_vwap_by_atr * a0
        near_vwap = abs(c0 - vwap0) <= near_thr

        # Gatilhos VKBP
        long_break = uptrend and (c0 > up0) and (
            self.allow_break_close) and ema20_ok and (c0 > vwap0)
        short_break = downtrend and (c0 < lo0) and (
            self.allow_break_close) and ema20_ok and (c0 < vwap0)

        long_pull = uptrend and near_vwap and (
            rsi0 >= self.rsi_trig) and (c0 > vwap0) and ema20_ok
        short_pull = downtrend and near_vwap and (
            rsi0 <= (100 - self.rsi_trig)) and (c0 < vwap0) and ema20_ok

        side: Optional[Side] = None
        trigger = None
        if long_break or short_break:
            side = Side.BUY if long_break else Side.SELL
            trigger = "break"
        elif long_pull or short_pull:
            side = Side.BUY if long_pull else Side.SELL
            trigger = "pullback"
        else:
            return None

        # Confiança heurística (o ML filtra depois via Executor; isso aqui ajuda sizing/ordem)
        conf = 0.55
        if trigger == "break":
            conf += 0.10
        if near_vwap:
            conf += 0.05
        if float(adx_h1.iloc[-1]) >= self.adx_thr + 5:
            conf += 0.05
        conf = float(min(0.95, max(0.50, conf)))

        # Meta p/ logs e risk manager
        meta: Dict[str, Any] = {
            "adx_h1": float(adx_h1.iloc[-1]),
            "near_thr": float(near_thr),
            "trigger": trigger,
        }

        # Distâncias tipo Donchian (só p/ debug; não é requisito da VKBP)
        try:
            from core.utils import donchian
            up_d, lo_d = donchian(df_e["h"], df_e["l"], self.donchian_win)
            meta["dist_up"] = float(max(0.0, up_d.iloc[-1] - c0))
            meta["dist_low"] = float(max(0.0, c0 - lo_d.iloc[-1]))
        except Exception:
            pass

        # >>>>>>> AQUI ESTÁ O FIX IMPORTANTE: incluir symbol= <<<<<<<
        return Signal(
            symbol=symbol,
            side=side,
            atr=a0,
            confidence=conf,
            meta=meta
        )
