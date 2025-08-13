from typing import Optional
import pandas as pd
from core.types import Signal, Side
from core.utils import donchian, ema, atr, adx


class DonchianBreakout(
    __import__("strategies.base", fromlist=["Strategy"]).Strategy
):
    def _regime(self, df_reg: pd.DataFrame, ema_fast: int, ema_slow: int, adx_thr: int):
        ema_f = ema(df_reg['c'], ema_fast)
        ema_s = ema(df_reg['c'], ema_slow)
        adx_h = adx(df_reg['h'], df_reg['l'], df_reg['c'], 14)
        up = (df_reg['c'].iloc[-1] > ema_s.iloc[-1]) and (ema_f.iloc[-1]
                                                          > ema_s.iloc[-1]) and (adx_h.iloc[-1] >= adx_thr)
        down = (df_reg['c'].iloc[-1] < ema_s.iloc[-1]) and (ema_f.iloc[-1]
                                                            < ema_s.iloc[-1]) and (adx_h.iloc[-1] >= adx_thr)
        return up, down, float(ema_f.diff().iloc[-1]), float(adx_h.iloc[-1])

    def generate_signal(self, symbol: str, df_exec: pd.DataFrame, df_regime: pd.DataFrame) -> Optional[Signal]:
        p = {
            "donchian":          self.params.get("donchian", 16),
            "ema_fast":          self.params.get("ema_fast", 50),
            "ema_slow":          self.params.get("ema_slow", 200),
            "adx_thr":           self.params.get("adx_thr", 18),
            "confirm_ema20":     self.params.get("confirm_ema20", True),
            "use_near_break":    self.params.get("use_near_break", True),
            "near_by_atr_ratio": self.params.get("near_by_atr_ratio", 0.10),
            "break_source":      self.params.get("break_source", "HL").upper(),
        }

        uptrend, downtrend, ema_slope, adx_val = self._regime(
            df_regime, p["ema_fast"], p["ema_slow"], p["adx_thr"])

        upc, loc = donchian(df_exec['h'], df_exec['l'], p["donchian"])
        ema20 = ema(df_exec['c'], 20)
        atr_e = atr(df_exec['h'], df_exec['l'], df_exec['c'], 14)

        c0, h0, l0 = df_exec['c'].iloc[-1], df_exec['h'].iloc[-1], df_exec['l'].iloc[-1]
        up0, lo0 = upc.iloc[-1], loc.iloc[-1]
        a = float(atr_e.iloc[-1])

        dist_up = max(0.0, up0 - c0)
        dist_low = max(0.0, c0 - lo0)
        near_thr = p["near_by_atr_ratio"] * a
        near_up = dist_up <= near_thr
        near_low = dist_low <= near_thr

        if p["break_source"] == "CLOSE":
            buy_break = c0 >= up0
            sell_break = c0 <= lo0
        else:
            buy_break = h0 >= up0
            sell_break = l0 <= lo0

        if p["confirm_ema20"]:
            buy_break = buy_break and (c0 > ema20.iloc[-1])
            sell_break = sell_break and (c0 < ema20.iloc[-1])

        if p["use_near_break"]:
            buy_break = buy_break or near_up
            sell_break = sell_break or near_low

        meta = {
            "dist_up": dist_up, "dist_low": dist_low,
            "ema20_slope": float(ema20.diff().iloc[-1]),
            "ema_slope_h1": ema_slope,
            "adx_h1": adx_val,
            "near_thr": near_thr
        }

        conf = 0.55
        if self.ml is not None:
            features = {
                "ema20_slope": meta["ema20_slope"],
                "atr_m5": a,
                "dist_up": meta["dist_up"],
                "dist_low": meta["dist_low"],
                "near_thr": meta["near_thr"],
                "frac_to_up": meta["dist_up"]/(meta["near_thr"]+1e-12) if meta["near_thr"] > 0 else 999.0,
                "frac_to_low": meta["dist_low"]/(meta["near_thr"]+1e-12) if meta["near_thr"] > 0 else 999.0,
                "above_ema20": 1.0 if (df_exec['c'].iloc[-1] > ema(df_exec['c'], 20).iloc[-1]) else 0.0,
                "buy_break_hl": 1.0 if (df_exec['h'].iloc[-1] >= up0) else 0.0,
                "sell_break_hl": 1.0 if (df_exec['l'].iloc[-1] <= lo0) else 0.0,
                "buy_close": 1.0 if (df_exec['c'].iloc[-1] >= up0) else 0.0,
                "sell_close": 1.0 if (df_exec['c'].iloc[-1] <= lo0) else 0.0,
                "adx_h1": meta["adx_h1"],
                "ema50_over_200_h1": meta.get("ema_slope_h1", 0.0),
                "trend_up_h1": 1.0 if (df_regime['c'].iloc[-1] > ema(df_regime['c'], 200).iloc[-1]) else 0.0,
                "trend_dn_h1": 1.0 if (df_regime['c'].iloc[-1] < ema(df_regime['c'], 200).iloc[-1]) else 0.0
            }
            conf = float(self.ml.predict(symbol, features)
                         ) if self.ml else 0.55

        if uptrend and buy_break:
            return Signal(symbol=symbol, side=Side.BUY, confidence=conf, atr=a, meta=meta)
        if downtrend and sell_break:
            return Signal(symbol=symbol, side=Side.SELL, confidence=conf, atr=a, meta=meta)
        return None
