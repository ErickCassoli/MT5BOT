# strategies/scalper_rsi_bb.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Optional

import pandas as pd

from core.types import Side
from core.utils import atr as _atr, adx as _adx
from core.indicators import rsi as _rsi, bollinger as _bb


class ScalperRSIBB:
    """
    Reversão curta para mercado lateral (M5):
      - Regime: ADX(H1) <= max_adx_h1 (default 14)
      - BUY:  RSI <= rsi_buy  e close <= banda inferior (BB len/k)
      - SELL: RSI >= rsi_sell e close >= banda superior
    Saídas (SL/TP/parcial/trailing) são geridas pelo Executor/RiskManager.
    """

    def __init__(
        self,
        rsi_len: int = 14,
        rsi_buy: int = 22,
        rsi_sell: int = 78,
        bb_len: int = 20,
        bb_k: float = 2.0,
        max_adx_h1: int = 14,
        min_bars: int = 120,
        ml_model=None,
        **__,
    ):
        self.rsi_len = int(rsi_len)
        self.rsi_buy = int(rsi_buy)
        self.rsi_sell = int(rsi_sell)
        self.bb_len = int(bb_len)
        self.bb_k = float(bb_k)
        self.max_adx_h1 = int(max_adx_h1)
        self.min_bars = int(min_bars)
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

    # -------- feature builder (v5 schema) --------
    @staticmethod
    def _feats_v5(
        adx_h1: float,
        rsi_m5: float,
        atr_now: float,
        bb_z: float,
        bb_width: float,
    ) -> dict:
        """
        Monta vetor padronizado v5 para o modo scalper:
        - usa rsi_m5, atr_now, bb_z, bb_width;
        - zera o que não se aplica (VWAP/Keltner, EMAs, volume, etc.);
        - marca origem src_scalper=1.0.
        """
        return {
            # básicos
            "adx_h1": adx_h1,
            "rsi_m5": rsi_m5,
            "atr_now": atr_now,

            # VWAP/Keltner (não usados aqui)
            "c_kdist_up": 0.0,
            "c_kdist_low": 0.0,
            "near_vwap": 0.0,
            "confirm_ema20": 0.0,

            # espaço/distância (não aplicável)
            "room_atr": 0.0,
            "break_dist": 0.0,

            # EMAs (não usados)
            "ema20_50": 0.0,
            "ema50_200": 0.0,

            # volume/BB (BB usados)
            "vol_ratio": 0.0,
            "bb_z": bb_z,
            "bb_width": bb_width,

            # origem
            "src_vk": 0.0,
            "src_don": 0.0,
            "src_bv": 0.0,
            "src_scalper": 1.0,
        }

    # -------- core --------
    def generate_signal(self, symbol: str, df_e: pd.DataFrame, df_r: pd.DataFrame):
        if len(df_e) < max(self.min_bars, self.bb_len + 5) or len(df_r) < 50:
            return None

        # Regime lateral (H1)
        adx_h1 = float(_adx(df_r["h"], df_r["l"], df_r["c"], 14).iloc[-1])
        if adx_h1 > self.max_adx_h1:
            return None

        c = df_e["c"].astype(float)
        a = _atr(df_e["h"], df_e["l"], c, 14)
        atr_now = float(a.iloc[-1])

        bb_up, bb_mid, bb_lo = _bb(c, self.bb_len, self.bb_k)
        r = _rsi(c, self.rsi_len)

        c0 = float(c.iloc[-1])
        r0 = float(r.iloc[-1])
        up0 = float(bb_up.iloc[-1])
        mid0 = float(bb_mid.iloc[-1])
        lo0 = float(bb_lo.iloc[-1])

        # BB features:
        # - bb_z: distância do close à média em "meias larguras" da banda ( (up - lo)/2 )
        half_width = max((up0 - lo0) / 2.0, 1e-12)
        bb_z = (c0 - mid0) / half_width
        # - bb_width: largura normalizada por ATR (torna dimensionless)
        bb_width = (up0 - lo0) / max(atr_now, 1e-9)

        meta = {
            "strategy": "ScalperRSIBB",
            "adx_h1": adx_h1,
            "rsi": r0,
            "bb_up": up0,
            "bb_mid": mid0,
            "bb_lo": lo0,
            "bb_z": bb_z,
            "bb_width": bb_width,
        }

        # BUY: RSI baixo + fora/na borda inferior
        if (r0 <= self.rsi_buy) and (c0 <= lo0):
            feats = self._feats_v5(adx_h1=adx_h1, rsi_m5=r0, atr_now=atr_now, bb_z=bb_z, bb_width=bb_width)
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.60
            return SimpleNamespace(side=Side.BUY, confidence=float(conf), atr=atr_now, meta=meta)

        # SELL: RSI alto + fora/na borda superior
        if (r0 >= self.rsi_sell) and (c0 >= up0):
            feats = self._feats_v5(adx_h1=adx_h1, rsi_m5=r0, atr_now=atr_now, bb_z=bb_z, bb_width=bb_width)
            conf = self._ml_conf(feats)
            if conf is None:
                conf = 0.60
            return SimpleNamespace(side=Side.SELL, confidence=float(conf), atr=atr_now, meta=meta)

        return None
