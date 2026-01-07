# strategies/price_action.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Optional, Dict, Any

import pandas as pd

from core.types import Side
from core.utils import atr as _atr, adx as _adx, ema as _ema, donchian as _donch
from core.indicators import vwap_session


class PriceActionConfluence:
    """
    Price Action com confluência “de verdade”:
      - Contexto: ADX(H1) >= adx_thr_trend e tendência local via EMA20 + slope>0 (BUY) / <0 (SELL)
      - Nível: proximidade de Donchian (sr_win) e/ou VWAP
      - Padrão: Engulfing OU Pin Bar (versão mais estrita: corpo forte / pavio dominante + posição do fechamento)
      - BOS (Break of Structure): close além da borda com margem ATR + (preferência) reteste
      - Room to run: precisa haver espaço até o alvo (borda oposta Donchian) em ATR

    Retorna SimpleNamespace(side, confidence, atr, meta)
    Confiança heurística 0.55–0.78. Use o gate do Executor para filtrar.
    """

    def __init__(
        self,
        adx_thr_trend: int = 20,       # ↑ mais seletivo
        ema_len: int = 20,
        sr_win: int = 24,
        min_break_atr: float = 0.14,   # ↑ margem BOS
        near_sr_by_atr: float = 0.18,  # ↑ exige mais “grude” no nível
        near_vwap_by_atr: float = 0.30,
        min_room_atr: float = 1.20,    # NOVO: espaço mínimo até alvo/oposto (em ATR)
        min_bars: int = 200,
        allow_retest: bool = True,
        ml_model=None,                 # compat
        **__,
    ):
        self.adx_thr_trend = int(adx_thr_trend)
        self.ema_len = int(ema_len)
        self.sr_win = int(sr_win)
        self.min_break_atr = float(min_break_atr)
        self.near_sr_by_atr = float(near_sr_by_atr)
        self.near_vwap_by_atr = float(near_vwap_by_atr)
        self.min_room_atr = float(min_room_atr)
        self.min_bars = int(min_bars)
        self.allow_retest = bool(allow_retest)
        self.ml = ml_model

    # ---------- helpers de candle/padrões (versão mais estrita) ----------
    @staticmethod
    def _engulfing_strict(o_prev: float, c_prev: float, o: float, c: float, h: float, l: float) -> tuple[bool, bool, bool]:
        rng = max(h - l, 1e-9)
        body = abs(c - o)
        body_ratio = body / rng
        # exige corpo “forte”
        strong_body = body_ratio >= 0.45

        bull = (c > o) and (c_prev < o_prev) and (c >= max(o_prev, c_prev)) and (o <= min(o_prev, c_prev))
        bear = (c < o) and (c_prev > o_prev) and (o <= min(o_prev, c_prev)) and (c >= max(o_prev, c_prev))
        return bull and strong_body, bear and strong_body, strong_body

    @staticmethod
    def _pinbar_strict(o: float, h: float, l: float, c: float, tail_mult: float = 2.2) -> tuple[bool, bool]:
        rng = max(h - l, 1e-9)
        body = abs(c - o)
        up_wick = h - max(c, o)
        lo_wick = min(c, o) - l

        # encerramento “posicionado”
        close_pos = (c - l) / rng if c >= o else (o - l) / rng  # 0=na mínima, 1=na máxima
        bull_pin = (lo_wick >= max(tail_mult * body, 0.35 * rng)) and (close_pos >= 0.65)
        bear_pin = (up_wick >= max(tail_mult * body, 0.35 * rng)) and (close_pos <= 0.35)
        return bull_pin, bear_pin

    @staticmethod
    def _is_near(x: float, y: float, thr: float) -> bool:
        return abs(x - y) <= thr

    def _confidence(self, base: float, bumps: Dict[str, float]) -> float:
        conf = base + sum(bumps.values())
        return max(0.55, min(0.78, conf))

    def _bos_signal(self, c0: float, h0: float, l0: float, up0: float, lo0: float, atr_now: float, allow_close_break: bool):
        margin = self.min_break_atr * max(atr_now, 1e-9)
        if allow_close_break:
            buy_ok = (c0 > up0 + margin)
            sell_ok = (c0 < lo0 - margin)
            buy_dist = (c0 - up0) / max(atr_now, 1e-9)
            sell_dist = (lo0 - c0) / max(atr_now, 1e-9)
        else:
            buy_ok = (h0 > up0 + margin)
            sell_ok = (l0 < lo0 - margin)
            buy_dist = (h0 - up0) / max(atr_now, 1e-9)
            sell_dist = (lo0 - l0) / max(atr_now, 1e-9)
        return buy_ok, sell_ok, buy_dist, sell_dist

    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame):
        if len(df_e) < max(self.min_bars, self.sr_win + 5) or len(df_r) < 50:
            return None

        # Regime H1
        adx_h1 = float(_adx(df_r["h"], df_r["l"], df_r["c"], 14).iloc[-1])
        if adx_h1 < self.adx_thr_trend:
            return None

        # M5
        o = df_e["o"].astype(float)
        h = df_e["h"].astype(float)
        l = df_e["l"].astype(float)
        c = df_e["c"].astype(float)

        a = _atr(h, l, c, 14)
        atr_now = float(a.iloc[-1])

        ema20_series = _ema(c, self.ema_len)
        ema20 = float(ema20_series.iloc[-1])
        ema20_prev = float(ema20_series.iloc[-4]) if len(ema20_series) >= 4 else ema20
        ema_slope = ema20 - ema20_prev

        up, lo = _donch(h, l, self.sr_win)
        up0, lo0 = float(up.iloc[-1]), float(lo.iloc[-1])

        vwap = vwap_session(df_e)
        vwap0 = float(vwap.iloc[-1])

        if len(df_e) < 3:
            return None
        o0, c0, h0, l0 = float(o.iloc[-1]), float(c.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1])
        o1, c1 = float(o.iloc[-2]), float(c.iloc[-2])

        # Tendência + slope
        up_trend = (c0 > ema20) and (ema_slope > 0)
        dn_trend = (c0 < ema20) and (ema_slope < 0)

        # Proximidades
        sr_thr = self.near_sr_by_atr * max(atr_now, 1e-9)
        vwap_thr = self.near_vwap_by_atr * max(atr_now, 1e-9)
        near_sup = self._is_near(c0, lo0, sr_thr)
        near_res = self._is_near(c0, up0, sr_thr)
        near_vwp = self._is_near(c0, vwap0, vwap_thr)

        # Padrões (estritos)
        bull_eng, bear_eng, body_strong = self._engulfing_strict(o1, c1, o0, c0, h0, l0)
        bull_pin, bear_pin = self._pinbar_strict(o0, h0, l0, c0, tail_mult=2.2)
        bull_rev = bull_eng or bull_pin
        bear_rev = bear_eng or bear_pin

        # BOS
        bos_buy, bos_sell, bos_buy_dist, bos_sell_dist = self._bos_signal(c0, h0, l0, up0, lo0, atr_now, allow_close_break=True)

        # Room to run (para BUY, espaço até up0; para SELL, até lo0)
        room_buy_atr = (up0 - c0) / max(atr_now, 1e-9)
        room_sell_atr = (c0 - lo0) / max(atr_now, 1e-9)
        room_buy_ok = room_buy_atr >= self.min_room_atr
        room_sell_ok = room_sell_atr >= self.min_room_atr

        # ---- BUY ----
        buy_reasons: Dict[str, float] = {}
        if up_trend and room_buy_ok:
            # PRECISA de confluência: nível (SR ou VWAP) + padrão
            if (near_sup or near_vwp) and bull_rev:
                buy_reasons["pa_confluence"] = 0.04
            # BOS preferencialmente com reteste
            if bos_buy:
                buy_reasons["bos"] = 0.03
                if self.allow_retest and c0 > c1:
                    buy_reasons["retest"] = 0.02

            if buy_reasons:
                base = 0.58
                if adx_h1 >= 30:
                    buy_reasons["adx30"] = 0.02
                elif adx_h1 >= 25:
                    buy_reasons["adx25"] = 0.01
                if near_vwp and near_sup:
                    buy_reasons["double_near"] = 0.02
                if body_strong:
                    buy_reasons["body"] = 0.01
                if ema_slope > 0:
                    buy_reasons["ema_slope"] = 0.01
                buy_reasons["room"] = 0.02  # já validado

                conf = self._confidence(base, buy_reasons)
                meta = {
                    "strategy": "PriceActionConfluence",
                    "side": "BUY",
                    "adx_h1": adx_h1,
                    "ema20": ema20,
                    "ema_slope": ema_slope,
                    "don_up": up0,
                    "don_lo": lo0,
                    "vwap": vwap0,
                    "near_sup": bool(near_sup),
                    "near_vwap": bool(near_vwp),
                    "bull_eng": bool(bull_eng),
                    "bull_pin": bool(bull_pin),
                    "bos_buy": bool(bos_buy),
                    "bos_buy_dist_atr": float(bos_buy_dist),
                    "atr_now": atr_now,
                    "room_buy_atr": room_buy_atr,
                    "reasons": list(buy_reasons.keys()),
                }
                return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

        # ---- SELL ----
        sell_reasons: Dict[str, float] = {}
        if dn_trend and room_sell_ok:
            if (near_res or near_vwp) and bear_rev:
                sell_reasons["pa_confluence"] = 0.04
            if bos_sell:
                sell_reasons["bos"] = 0.03
                if self.allow_retest and c0 < c1:
                    sell_reasons["retest"] = 0.02

            if sell_reasons:
                base = 0.58
                if adx_h1 >= 30:
                    sell_reasons["adx30"] = 0.02
                elif adx_h1 >= 25:
                    sell_reasons["adx25"] = 0.01
                if near_vwp and near_res:
                    sell_reasons["double_near"] = 0.02
                if body_strong:
                    sell_reasons["body"] = 0.01
                if ema_slope < 0:
                    sell_reasons["ema_slope"] = 0.01
                sell_reasons["room"] = 0.02

                conf = self._confidence(base, sell_reasons)
                meta = {
                    "strategy": "PriceActionConfluence",
                    "side": "SELL",
                    "adx_h1": adx_h1,
                    "ema20": ema20,
                    "ema_slope": ema_slope,
                    "don_up": up0,
                    "don_lo": lo0,
                    "vwap": vwap0,
                    "near_res": bool(near_res),
                    "near_vwap": bool(near_vwp),
                    "bear_eng": bool(bear_eng),
                    "bear_pin": bool(bear_pin),
                    "bos_sell": bool(bos_sell),
                    "bos_sell_dist_atr": float(bos_sell_dist),
                    "atr_now": atr_now,
                    "room_sell_atr": room_sell_atr,
                    "reasons": list(sell_reasons.keys()),
                }
                return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        return None
