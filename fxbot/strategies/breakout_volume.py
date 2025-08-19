# strategies/breakout_volume.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from core.types import Side
from core.utils import atr as _atr, adx as _adx


class BreakoutVolume:
    """
    Expansão com volume (M5), regime por ADX(H1).

    Regras:
      - Rompe SR calculado nos últimos N candles (sr_win), COM MARGEM de min_break_atr * ATR.
      - Volume atual deve ser >= vol_mult * média(vol_lookback).
      - (Opcional) Reteste: candle anterior já fechou além do SR+margem e o atual confirma;
        no reteste o volume pode ser levemente relaxado (retest_vol_relax).

    Parâmetros típicos:
      sr_win=24~32, vol_lookback=20, vol_mult=1.8, adx_thr=16,
      min_break_atr=0.12, retest_vol_relax=0.90, min_bars>=200

    Saídas (SL/TP/partial/trailing) são geridas no Executor/RiskManager.
    """

    def __init__(
        self,
        sr_win: int = 24,
        vol_lookback: int = 20,
        vol_mult: float = 1.8,
        adx_thr: int = 16,
        allow_retest: bool = True,
        min_bars: int = 200,
        min_break_atr: float = 0.12,
        retest_vol_relax: float = 0.90,
        ml_model=None,
        **__,
    ):
        self.sr_win = int(sr_win)
        self.vol_lookback = int(vol_lookback)
        self.vol_mult = float(vol_mult)
        self.adx_thr = int(adx_thr)
        self.allow_retest = bool(allow_retest)
        self.min_bars = int(min_bars)
        self.min_break_atr = float(min_break_atr)
        self.retest_vol_relax = float(retest_vol_relax)
        self.ml = ml_model

    # -------- ML helper --------
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

    # -------- Feature builder (v5 schema) --------
    @staticmethod
    def _feats_v5(side: Side, atr_now: float, break_dist_atr: float, vol_ratio: float) -> dict:
        """Monta vetor de features v5 padronizado para Breakout+Volume."""
        return {
            # básicos
            "adx_h1": 0.0,          # preenchido no call site (placeholder)
            "rsi_m5": 0.0,
            "atr_now": atr_now,

            # VWAP/Keltner (não usados aqui)
            "c_kdist_up": 0.0,
            "c_kdist_low": 0.0,
            "near_vwap": 0.0,
            "confirm_ema20": 0.0,

            # espaço e distância de rompimento
            "room_atr": 0.0,
            "break_dist": break_dist_atr,  # (c - sr)/ATR com sinal correto

            # EMAs (não usados nesta estratégia)
            "ema20_50": 0.0,
            "ema50_200": 0.0,

            # volume/BB
            "vol_ratio": vol_ratio,
            "bb_z": 0.0,
            "bb_width": 0.0,

            # origem (one-hot)
            "src_vk": 0.0,
            "src_don": 0.0,
            "src_bv": 1.0,
            "src_scalper": 0.0,
        }

    # -------- Core --------
    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame):
        if len(df_e) < max(self.min_bars, self.sr_win + 5) or len(df_r) < 50:
            return None

        # Regime H1
        adx_h1 = float(_adx(df_r["h"], df_r["l"], df_r["c"], 14).iloc[-1])
        if adx_h1 < self.adx_thr:
            return None

        # Séries M5
        c = df_e["c"].astype(float)
        h = df_e["h"].astype(float)
        l = df_e["l"].astype(float)
        v = df_e["v"].astype(float)

        atr_now = float(_atr(h, l, c, 14).iloc[-1])

        # SR sobre janela COMPLETA anterior ao candle atual
        sr_high_series = h.rolling(self.sr_win, min_periods=self.sr_win).max()
        sr_low_series  = l.rolling(self.sr_win, min_periods=self.sr_win).min()
        sr_high = float(sr_high_series.iloc[-2])
        sr_low  = float(sr_low_series.iloc[-2])

        # Volume
        vol_avg = float(v.rolling(self.vol_lookback, min_periods=self.vol_lookback).mean().iloc[-1])
        vol_now = float(v.iloc[-1])
        vol_ratio = vol_now / max(vol_avg, 1e-9)

        c0 = float(c.iloc[-1])
        c1 = float(c.iloc[-2]) if len(c) >= 2 else c0

        # Margem mínima em preço
        margin_px = self.min_break_atr * max(atr_now, 1e-9)

        # Rompimento por FECHAMENTO com margem + volume forte
        broke_up   = (c0 > sr_high + margin_px) and (vol_ratio >= self.vol_mult)
        broke_down = (c0 < sr_low  - margin_px) and (vol_ratio >= self.vol_mult)

        meta = {
            "strategy": "BreakoutVolume",
            "adx_h1": adx_h1,
            "sr_high": float(sr_high),
            "sr_low": float(sr_low),
            "vol_now": float(vol_now),
            "vol_avg": float(vol_avg),
            "vol_ratio": float(vol_ratio),
            "break_margin_atr": self.min_break_atr,
        }

        # --- BUY breakout ---
        if broke_up:
            break_dist_atr = (c0 - sr_high) / max(atr_now, 1e-9)
            feats = self._feats_v5(Side.BUY, atr_now, break_dist_atr, vol_ratio)
            feats["adx_h1"] = adx_h1
            meta["break_dist_atr"] = break_dist_atr

            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.64
            return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

        # --- SELL breakout ---
        if broke_down:
            break_dist_atr = (sr_low - c0) / max(atr_now, 1e-9)
            feats = self._feats_v5(Side.SELL, atr_now, break_dist_atr, vol_ratio)
            feats["adx_h1"] = adx_h1
            meta["break_dist_atr"] = break_dist_atr

            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.62
            return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        # --- Reteste (opcional) ---
        if self.allow_retest and len(df_e) >= self.sr_win + 3:
            # BUY retest: candle anterior já fechou além de sr_high+margin e o atual confirma (close > close[-1])
            if (c1 > sr_high + margin_px) and (c0 > c1) and (vol_ratio >= self.vol_mult * self.retest_vol_relax):
                feats = self._feats_v5(Side.BUY, atr_now, break_dist_atr=0.0, vol_ratio=vol_ratio)
                feats["adx_h1"] = adx_h1
                meta["retest"] = True

                conf = self._ml_conf(feats)
                if conf is None:
                    conf = 0.60
                return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

            # SELL retest: candle anterior já fechou além de sr_low-margin e o atual confirma (close < close[-1])
            if (c1 < sr_low - margin_px) and (c0 < c1) and (vol_ratio >= self.vol_mult * self.retest_vol_relax):
                feats = self._feats_v5(Side.SELL, atr_now, break_dist_atr=0.0, vol_ratio=vol_ratio)
                feats["adx_h1"] = adx_h1
                meta["retest"] = True

                conf = self._ml_conf(feats)
                if conf is None:
                    conf = 0.60
                return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        return None
