# strategies/vwap_keltner.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from core.types import Side
from core.utils import atr as _atr, adx as _adx, ema as _ema
from core.indicators import rsi as _rsi, vwap_session, keltner_channels


class VWAPKeltner:
    """
    Tendência/pullback com VWAP + Keltner no M5, regime por ADX(H1).
    Regras:
      - BUY: preço acima do VWAP e rompendo/segurando entre banda média e upper;
              ADX(H1) >= adx_thr; RSI(M5) > rsi_trig
      - SELL: simétrico abaixo do VWAP; RSI(M5) < (100 - rsi_trig) ou < 50 conforme config
      - Entrada: breakout da banda OU pullback à média/VWAP com rejeição (close volta p/ dentro)
      - Filtros adicionais:
          * Breakout precisa de margem mínima: min_break_atr × ATR
          * Pullback precisa de “room to run”: min_room_atr × ATR até a banda alvo
          * (opcional) confirm_ema20: close precisa respeitar o lado da EMA20
    """

    def __init__(
        self,
        k_ema_len: int = 20,
        k_atr_len: int = 10,
        k_mult: float = 1.8,
        adx_thr: int = 18,
        rsi_len: int = 14,
        rsi_trig: int = 50,
        near_vwap_by_atr: float = 0.35,
        confirm_ema20: bool = True,
        allow_break_close: bool = True,
        min_bars: int = 150,
        donchian: int = 16,          # compat/logs
        min_break_atr: float = 0.12, # margem mínima no breakout (em ATR)
        min_room_atr: float = 1.30,  # espaço mínimo até a banda alvo (em ATR) p/ pullback
        ml_model=None,
        **__,
    ):
        self.k_ema_len = int(k_ema_len)
        self.k_atr_len = int(k_atr_len)
        self.k_mult = float(k_mult)
        self.adx_thr = int(adx_thr)
        self.rsi_len = int(rsi_len)
        self.rsi_trig = int(rsi_trig)
        self.near_vwap_by_atr = float(near_vwap_by_atr)
        self.confirm_ema20 = bool(confirm_ema20)
        self.allow_break_close = bool(allow_break_close)
        self.min_bars = int(min_bars)
        self.min_break_atr = float(min_break_atr)
        self.min_room_atr = float(min_room_atr)
        self.ml = ml_model

    # ---------- ML helper ----------
    def _ml_conf(self, feats: dict) -> Optional[float]:
        if self.ml is None:
            return None
        try:
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

    # ---------- Core ----------
    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame):
        if len(df_e) < max(self.min_bars, 50) or len(df_r) < 50:
            return None

        # Regime H1
        adx_h1 = float(_adx(df_r["h"], df_r["l"], df_r["c"], 14).iloc[-1])
        if adx_h1 < self.adx_thr:
            return None

        # M5 indicadores
        a = _atr(df_e["h"], df_e["l"], df_e["c"], self.k_atr_len)
        atr_now = float(a.iloc[-1])

        vwap = vwap_session(df_e)
        ku, km, kl = keltner_channels(df_e["c"], a, ema_len=self.k_ema_len, mult=self.k_mult)
        r = _rsi(df_e["c"], self.rsi_len)
        rsi_now = float(r.iloc[-1])

        c0 = float(df_e["c"].iloc[-1])
        c1 = float(df_e["c"].iloc[-2]) if len(df_e) >= 2 else c0
        vwap0 = float(vwap.iloc[-1])
        ku0, km0, kl0 = float(ku.iloc[-1]), float(km.iloc[-1]), float(kl.iloc[-1])
        h0 = float(df_e["h"].iloc[-1])
        l0 = float(df_e["l"].iloc[-1])

        ema20 = float(_ema(df_e["c"], 20).iloc[-1])

        # Condições
        above_vwap = c0 > vwap0
        below_vwap = c0 < vwap0
        in_upper_band = (c0 >= km0) and (c0 <= ku0)
        in_lower_band = (c0 <= km0) and (c0 >= kl0)
        near_vwap = abs(c0 - vwap0) <= self.near_vwap_by_atr * atr_now

        # Regras de breakout (close ou wick, conforme flag)
        buy_break_close = (c0 > ku0 + self.min_break_atr * atr_now) if self.allow_break_close else False
        buy_break_wick  = (h0 > ku0 + self.min_break_atr * atr_now) if not self.allow_break_close else False
        buy_break = above_vwap and (buy_break_close or buy_break_wick)

        sell_break_close = (c0 < kl0 - self.min_break_atr * atr_now) if self.allow_break_close else False
        sell_break_wick  = (l0 < kl0 - self.min_break_atr * atr_now) if not self.allow_break_close else False
        sell_break = below_vwap and (sell_break_close or sell_break_wick)

        # Pullback com rejeição
        buy_pull = above_vwap and in_upper_band and near_vwap and (c1 <= km0) and (c0 > km0)
        sell_pull = below_vwap and in_lower_band and near_vwap and (c1 >= km0) and (c0 < km0)

        # “Room to run”
        room_buy_atr = (ku0 - c0) / max(atr_now, 1e-9)
        room_sell_atr = (c0 - kl0) / max(atr_now, 1e-9)
        buy_pull_ok = buy_pull and (room_buy_atr >= self.min_room_atr)
        sell_pull_ok = sell_pull and (room_sell_atr >= self.min_room_atr)

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
            "break_margin_atr": self.min_break_atr,
            "room_atr": room_buy_atr if above_vwap else room_sell_atr,
            "ema20": ema20,
        }

        # Filtros de RSI + confirmação EMA20 (se habilitada)
        def _ok_buy_filters() -> bool:
            if rsi_now <= max(50, self.rsi_trig):
                return False
            if self.confirm_ema20 and not (c0 > ema20):
                return False
            return True

        def _ok_sell_filters() -> bool:
            # mais rígido por padrão; (100 - rsi_trig) pode ser >50
            if rsi_now >= min(50, 100 - self.rsi_trig):
                return False
            if self.confirm_ema20 and not (c0 < ema20):
                return False
            return True

        # Helper para montar o vetor v5 completo
        def _feats(side: Side, is_break: bool) -> dict:
            # básicos
            feats = {
                "adx_h1": adx_h1,
                "rsi_m5": rsi_now,
                "atr_now": atr_now,
                # VWAP/Keltner
                "c_kdist_up": (c0 - km0) / max(atr_now, 1e-9),
                "c_kdist_low": (km0 - c0) / max(atr_now, 1e-9),
                "near_vwap": float(near_vwap),
                "confirm_ema20": float(c0 > ema20) if side == Side.BUY else float(c0 < ema20),
                # espaço e distância de rompimento
                "room_atr": (ku0 - c0) / max(atr_now, 1e-9) if side == Side.BUY else (c0 - kl0) / max(atr_now, 1e-9),
                "break_dist": (
                    (max(c0, h0) - ku0) / max(atr_now, 1e-9) if (side == Side.BUY and is_break)
                    else (kl0 - min(c0, l0)) / max(atr_now, 1e-9) if (side == Side.SELL and is_break)
                    else 0.0
                ),
                # não usados por VK (zerados)
                "ema20_50": 0.0,
                "ema50_200": 0.0,
                "vol_ratio": 0.0,
                "bb_z": 0.0,
                "bb_width": 0.0,
                # origem
                "src_vk": 1.0,
                "src_don": 0.0,
                "src_bv": 0.0,
                "src_scalper": 0.0,
            }
            return feats

        # --- BUYs ---
        if buy_break and _ok_buy_filters():
            feats = _feats(Side.BUY, is_break=True)
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.64
            return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

        if buy_pull_ok and _ok_buy_filters():
            feats = _feats(Side.BUY, is_break=False)
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.62
            return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

        # --- SELLs ---
        if sell_break and _ok_sell_filters():
            feats = _feats(Side.SELL, is_break=True)
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.62
            return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        if sell_pull_ok and _ok_sell_filters():
            feats = _feats(Side.SELL, is_break=False)
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.60
            return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        return None
