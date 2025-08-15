# fxbot/tools/build_dataset_super.py — v5-super
# Core compatível com o live (18 feats) + extras de pesquisa (não usados no live).
#
# Core (v5 live-safe):
#   adx_h1, rsi_m5, atr_now,
#   c_kdist_up, c_kdist_low, near_vwap, confirm_ema20,
#   room_atr, break_dist,
#   ema20_50, ema50_200,
#   vol_ratio,
#   bb_z, bb_width,
#   src_vk, src_don, src_bv, src_scalper
#
# Extras (pesquisa, ficam no .parquet):
#   hour, dow, hour_sin, hour_cos, dow_sin, dow_cos,
#   atr_h1, atr_h1_ratio, ret1, ret5, rv12,
#   don10_pos, don16_pos, don24_pos,
#   v_ratio20, v_surge, range_atr, r_room,
#   y_short(=y_a24), y_long(=y_a48), r_best, t_hit, hit_type
#
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from core.logging import get_logger
from core.utils import atr, ema, adx, donchian as _donchian
from core.indicators import rsi as _rsi  # usa mesmo RSI do live

log = get_logger(__name__)

_TF = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
}

# --------------------- helpers ---------------------

def _fetch(sym: str, tf: str, n: int) -> pd.DataFrame:
    mt5.symbol_select(sym, True)
    rates = mt5.copy_rates_from_pos(sym, _TF[tf], 0, n)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos falhou p/ {sym}/{tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"open":"o","high":"h","low":"l","close":"c","tick_volume":"v"}, inplace=True)
    return df[["time","o","h","l","c","v"]]

def _bollinger(series: pd.Series, length: int, k: float) -> Tuple[pd.Series,pd.Series,pd.Series]:
    m = series.rolling(length, min_periods=length).mean()
    s = series.rolling(length, min_periods=length).std(ddof=0)
    up = m + k*s
    lo = m - k*s
    return up, m, lo

def _label_tb(exec_df: pd.DataFrame, i: int, side: str, m_sl: float, m_tp: float, ahead: int) -> Tuple[int,int,float,str]:
    """
    Triple-barrier:
      retorna (y, t_hit, r_best, hit_type)
      y: 1 se TP antes de SL; 0 se SL antes de TP; -1 se nenhum até ahead
      t_hit: barras até o evento (ou ahead se nenhum)
      r_best: melhor R atingido dentro do horizonte
      hit_type: "TP"/"SL"/"NONE"
    Entrada na ABERTURA do (i+1); o (i+1) CONTA na trajetória.
    """
    a = float(atr(exec_df["h"], exec_df["l"], exec_df["c"], 14).iloc[i])
    if not np.isfinite(a) or a <= 0:
        return -1, 0, 0.0, "NONE"
    j = i + 1
    if j >= len(exec_df):
        return -1, 0, 0.0, "NONE"

    entry = float(exec_df["o"].iloc[j])
    lows  = exec_df["l"].iloc[j:j+ahead]
    highs = exec_df["h"].iloc[j:j+ahead]

    if side == "BUY":
        sl, tp = entry - m_sl*a, entry + m_tp*a
        r_best = 0.0
        for k, (lo, hi) in enumerate(zip(lows, highs), start=1):
            # melhor R no caminho (contra SL fixo)
            r_best = max(r_best, (hi - entry) / max((entry - sl), 1e-9))
            if lo <= sl: return 0, k, r_best, "SL"
            if hi >= tp: return 1, k, r_best, "TP"
        return -1, len(lows), r_best, "NONE"
    else:
        sl, tp = entry + m_sl*a, entry - m_tp*a
        r_best = 0.0
        for k, (lo, hi) in enumerate(zip(lows, highs), start=1):
            r_best = max(r_best, (entry - lo) / max((sl - entry), 1e-9))
            if hi >= sl: return 0, k, r_best, "SL"
            if lo <= tp: return 1, k, r_best, "TP"
        return -1, len(lows), r_best, "NONE"

def _time_feats(ts: pd.Timestamp) -> Dict[str,float]:
    h = int(ts.hour)
    d = int(ts.weekday())  # 0=Mon
    hour_sin = np.sin(2*np.pi*h/24.0); hour_cos = np.cos(2*np.pi*h/24.0)
    dow_sin  = np.sin(2*np.pi*d/7.0);  dow_cos  = np.cos(2*np.pi*d/7.0)
    return {
        "hour": h, "dow": d, "hour_sin": hour_sin, "hour_cos": hour_cos, "dow_sin": dow_sin, "dow_cos": dow_cos
    }

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Dataset v5-super — Core (live-safe) + extras de pesquisa.")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--tf-exec", default="M5")
    ap.add_argument("--tf-regime", default="H1")

    # Parâmetros de lógica/gates
    ap.add_argument("--donchian", type=int, default=16)
    ap.add_argument("--sr-win", type=int, default=24)
    ap.add_argument("--vol-lookback", type=int, default=20)
    ap.add_argument("--vol-mult", type=float, default=1.6)
    ap.add_argument("--min-break-atr", type=float, default=0.12)
    ap.add_argument("--room-atr", type=float, default=1.4)

    # Sessão (UTC)
    ap.add_argument("--hour-start", type=int, default=7)
    ap.add_argument("--hour-end", type=int, default=20)

    # SL/TP (trend) e (scalper)
    ap.add_argument("--atr-sl", type=float, default=1.6)
    ap.add_argument("--atr-tp", type=float, default=3.2)
    ap.add_argument("--scalper-atr-sl", type=float, default=1.0)
    ap.add_argument("--scalper-atr-tp", type=float, default=2.0)

    # Bollinger/RSI (scalper)
    ap.add_argument("--bb-len", type=int, default=20)
    ap.add_argument("--bb-k", type=float, default=2.0)
    ap.add_argument("--scalper-max-adx", type=float, default=18.0)
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--rsi-buy", type=float, default=30.0)
    ap.add_argument("--rsi-sell", type=float, default=70.0)

    # Horizonte e dados
    ap.add_argument("--ahead", type=int, default=24)        # label principal (y)
    ap.add_argument("--ahead-long", type=int, default=48)   # label auxiliar (y_long)
    ap.add_argument("--bars-exec", type=int, default=32000)
    ap.add_argument("--bars-regime", type=int, default=12000)

    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not mt5.initialize():
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")

    rows: List[Dict] = []
    try:
        for sym in args.symbols:
            log.info(f"[{sym}] baixando dados…")
            df_e = _fetch(sym, args.tf_exec, args.bars_exec)
            df_r = _fetch(sym, args.tf_regime, args.bars_regime)

            # sincroniza topo
            tmax = min(df_e["time"].iloc[-1], df_r["time"].iloc[-1])
            df_e = df_e[df_e["time"] <= tmax].reset_index(drop=True)
            df_r = df_r[df_r["time"] <= tmax].reset_index(drop=True)

            # filtro sessão [hs, he)
            hs, he = int(args.hour_start), int(args.hour_end)
            if hs < he:
                df_e = df_e[df_e["time"].dt.hour.between(hs, he-1, inclusive="both")].reset_index(drop=True)
            else:
                mask = (df_e["time"].dt.hour >= hs) | (df_e["time"].dt.hour < he)
                df_e = df_e[mask].reset_index(drop=True)

            # burn-in
            start = max(300, args.donchian + 30, args.bb_len + 5)
            end = len(df_e) - (max(args.ahead, args.ahead_long) + 3)
            if end <= start:
                log.warning(f"[{sym}] poucos dados após filtros. Pulando.")
                continue

            # -------- pré-cálculos M5 --------
            a_m5   = atr(df_e["h"], df_e["l"], df_e["c"], 14)
            ema20  = ema(df_e["c"], 20)
            ema50  = ema(df_e["c"], 50)
            ema200 = ema(df_e["c"], 200)

            # VWAP
            cumvol = df_e["v"].replace(0, np.nan).cumsum()
            vwap   = (df_e["c"] * df_e["v"]).cumsum() / cumvol
            vwap   = vwap.ffill().bfill()

            # Keltner (VK)
            ema20_k = ema(df_e["c"], 20)
            atr_k   = atr(df_e["h"], df_e["l"], df_e["c"], 10)
            k_mid   = ema20_k
            k_up    = k_mid + 1.8 * atr_k
            k_low   = k_mid - 1.8 * atr_k

            # SR/Volume (BV)
            sr_high = df_e["h"].rolling(args.sr_win, min_periods=args.sr_win).max()
            sr_low  = df_e["l"].rolling(args.sr_win, min_periods=args.sr_win).min()
            vol_avg = df_e["v"].rolling(args.vol_lookback, min_periods=args.vol_lookback).mean()

            # Bollinger/RSI (Scalper)
            bb_up, bb_mid, bb_lo = _bollinger(df_e["c"], args.bb_len, args.bb_k)
            rsi_m5 = _rsi(df_e["c"], args.rsi_len)

            # Extras
            ret1 = (df_e["c"] / df_e["c"].shift(1) - 1.0).fillna(0.0)
            ret5 = (df_e["c"] / df_e["c"].shift(5) - 1.0).fillna(0.0)
            rv12 = ret1.rolling(12, min_periods=12).std(ddof=0)
            rv12 = rv12.fillna(rv12.mean())

            # -------- loop temporal --------
            for i in range(start, end):
                t_i = df_e["time"].iloc[i]
                df_r_win = df_r[df_r["time"] <= t_i]
                if len(df_r_win) < 210:
                    continue

                # Regime H1
                adx_h1 = float(adx(df_r_win["h"], df_r_win["l"], df_r_win["c"], 14).iloc[-1])
                atr_h1_now = float(atr(df_r_win["h"], df_r_win["l"], df_r_win["c"], 14).iloc[-1])

                # M5 no i
                c0 = float(df_e["c"].iloc[i]); h0 = float(df_e["h"].iloc[i]); l0 = float(df_e["l"].iloc[i])
                a0 = float(a_m5.iloc[i])
                if not np.isfinite(a0) or a0 <= 0: 
                    continue

                ema20i, ema50i, ema200i = float(ema20.iloc[i]), float(ema50.iloc[i]), float(ema200.iloc[i])
                vw0 = float(vwap.iloc[i])
                ku0, km0, kl0 = float(k_up.iloc[i]), float(k_mid.iloc[i]), float(k_low.iloc[i])

                up_i, lo_i = _donchian(df_e["h"][:i+1], df_e["l"][:i+1], args.donchian)
                up0, lo0 = float(up_i.iloc[-1]), float(lo_i.iloc[-1])

                # SR/Volume
                sr_hi = float(sr_high.iloc[i]) if np.isfinite(sr_high.iloc[i]) else np.nan
                sr_lo = float(sr_low.iloc[i])  if np.isfinite(sr_low.iloc[i])  else np.nan
                v_now = float(df_e["v"].iloc[i])
                v_avg = float(vol_avg.iloc[i]) if np.isfinite(vol_avg.iloc[i]) else np.nan
                vol_ratio = (v_now / v_avg) if (v_avg and v_avg > 0) else 0.0

                # Bollinger/RSI
                bb_up0, bb_mid0, bb_lo0 = float(bb_up.iloc[i]), float(bb_mid.iloc[i]), float(bb_lo.iloc[i])
                rsi0 = float(rsi_m5.iloc[i]) if np.isfinite(rsi_m5.iloc[i]) else 50.0
                bb_std = (bb_up0 - bb_mid0) / max(args.bb_k, 1e-9)
                bb_z = (c0 - bb_mid0) / max(bb_std, 1e-9) if bb_std > 0 else 0.0
                bb_width = (bb_up0 - bb_lo0) / max(a0, 1e-9) if np.isfinite(bb_up0) and np.isfinite(bb_lo0) else 0.0

                # Gates
                margin = args.min_break_atr * a0
                above_vwap = c0 > vw0; below_vwap = c0 < vw0
                in_upper = (c0 >= km0) and (c0 <= ku0)
                in_lower = (c0 <= km0) and (c0 >= kl0)
                room_buy = (ku0 - c0) / max(a0, 1e-9)
                room_sel = (c0 - kl0) / max(a0, 1e-9)

                # VK
                buy_vk  = (adx_h1 >= 14) and (rsi0 > 50.0) and ( (above_vwap and (c0 > ku0 + margin or h0 > ku0 + margin)) or
                                                                  (above_vwap and in_upper and (room_buy >= args.room_atr)) )
                sell_vk = (adx_h1 >= 14) and (rsi0 < 50.0) and ( (below_vwap and (c0 < kl0 - margin or l0 < kl0 - margin)) or
                                                                  (below_vwap and in_lower and (room_sel >= args.room_atr)) )

                # DON
                buy_don  = (adx_h1 >= 18) and (ema20i > ema50i > ema200i) and (c0 >= up0 + margin)
                sell_don = (adx_h1 >= 18) and (ema20i < ema50i < ema200i) and (c0 <= lo0 - margin)

                # BV
                buy_bv  = (adx_h1 >= 12) and np.isfinite(sr_hi) and (c0 > sr_hi + margin) and (vol_ratio >= args.vol_mult)
                sell_bv = (adx_h1 >= 12) and np.isfinite(sr_lo) and (c0 < sr_lo - margin) and (vol_ratio >= args.vol_mult)

                # SCALPER
                lateral = (adx_h1 <= args.scalper_max_adx)
                buy_scl  = lateral and (rsi0 <= args.rsi_buy)  and (c0 <= bb_lo0 or l0 <= bb_lo0)
                sell_scl = lateral and (rsi0 >= args.rsi_sell) and (c0 >= bb_up0 or h0 >= bb_up0)

                families = []
                if buy_vk or sell_vk: families.append("vk")
                if buy_don or sell_don: families.append("don")
                if buy_bv or sell_bv: families.append("bv")
                if buy_scl or sell_scl: families.append("scl")
                if not families: 
                    continue

                # time extras
                tfe = _time_feats(t_i)

                for side in ("BUY","SELL"):
                    fire = ((side=="BUY" and (buy_vk or buy_don or buy_bv or buy_scl)) or
                            (side=="SELL" and (sell_vk or sell_don or sell_bv or sell_scl)))
                    if not fire:
                        continue

                    for fam in families:
                        # labels
                        if fam == "scl":
                            y, t_hit, r_best, htyp = _label_tb(df_e, i, side, args.scalper_atr_sl, args.scalper_atr_tp, args.ahead)
                            yL, _, _, _            = _label_tb(df_e, i, side, args.scalper_atr_sl, args.scalper_atr_tp, args.ahead_long)

                        else:
                            y, t_hit, r_best, htyp = _label_tb(df_e, i, side, args.atr_sl, args.atr_tp, args.ahead)
                            yL, _, _, _            = _label_tb(df_e, i, side, args.atr_sl, args.atr_tp, args.ahead_long)
                        if y < 0: 
                            continue

                        # core v5 (live-safe)
                        if fam == "vk":
                            feats = {
                                "adx_h1": adx_h1, "rsi_m5": rsi0, "atr_now": a0,
                                "c_kdist_up": (c0 - km0) / max(a0,1e-9),
                                "c_kdist_low": (km0 - c0) / max(a0,1e-9),
                                "near_vwap": float(abs(c0 - vw0) <= 0.35*a0),
                                "confirm_ema20": float(c0 > km0),
                                "room_atr": room_buy if side=="BUY" else room_sel,
                                "break_dist": (c0 - ku0)/max(a0,1e-9) if side=="BUY" else (kl0 - c0)/max(a0,1e-9),
                                "ema20_50": 0.0, "ema50_200": 0.0,
                                "vol_ratio": 0.0,
                                "bb_z": 0.0, "bb_width": 0.0,
                                "src_vk":1.0, "src_don":0.0, "src_bv":0.0, "src_scalper":0.0,
                            }
                        elif fam == "don":
                            feats = {
                                "adx_h1": adx_h1, "rsi_m5": 0.0, "atr_now": a0,
                                "c_kdist_up": 0.0, "c_kdist_low": 0.0,
                                "near_vwap": 0.0, "confirm_ema20": 0.0,
                                "room_atr": 0.0,
                                "break_dist": (c0 - up0)/max(a0,1e-9) if side=="BUY" else (lo0 - c0)/max(a0,1e-9),
                                "ema20_50": (ema20i - ema50i), "ema50_200": (ema50i - ema200i),
                                "vol_ratio": 0.0,
                                "bb_z": 0.0, "bb_width": 0.0,
                                "src_vk":0.0, "src_don":1.0, "src_bv":0.0, "src_scalper":0.0,
                            }
                        elif fam == "bv":
                            feats = {
                                "adx_h1": adx_h1, "rsi_m5": 0.0, "atr_now": a0,
                                "c_kdist_up": 0.0, "c_kdist_low": 0.0,
                                "near_vwap": 0.0, "confirm_ema20": 0.0,
                                "room_atr": 0.0,
                                "break_dist": (c0 - sr_hi)/max(a0,1e-9) if side=="BUY" else (sr_lo - c0)/max(a0,1e-9),
                                "ema20_50": 0.0, "ema50_200": 0.0,
                                "vol_ratio": (v_now / max(v_avg,1e-9)) if v_avg>0 else 0.0,
                                "bb_z": 0.0, "bb_width": 0.0,
                                "src_vk":0.0, "src_don":0.0, "src_bv":1.0, "src_scalper":0.0,
                            }
                        else:  # scalper
                            feats = {
                                "adx_h1": adx_h1, "rsi_m5": rsi0, "atr_now": a0,
                                "c_kdist_up": 0.0, "c_kdist_low": 0.0,
                                "near_vwap": 0.0, "confirm_ema20": 0.0,
                                "room_atr": 0.0,
                                "break_dist": (bb_mid0 - c0)/max(a0,1e-9) if side=="BUY" else (c0 - bb_mid0)/max(a0,1e-9),
                                "ema20_50": 0.0, "ema50_200": 0.0,
                                "vol_ratio": 0.0,
                                "bb_z": bb_z, "bb_width": bb_width,
                                "src_vk":0.0, "src_don":0.0, "src_bv":0.0, "src_scalper":1.0,
                            }

                        # extras
                        extras = {
                            "hour": tfe["hour"], "dow": tfe["dow"],
                            "hour_sin": tfe["hour_sin"], "hour_cos": tfe["hour_cos"],
                            "dow_sin": tfe["dow_sin"], "dow_cos": tfe["dow_cos"],
                            "atr_h1": atr_h1_now,
                            "atr_h1_ratio": (atr_h1_now / max(a0,1e-9)),
                            "ret1": float(ret1.iloc[i]), "ret5": float(ret5.iloc[i]), "rv12": float(rv12.iloc[i]),
                            "don10_pos": (c0 - float(_donchian(df_e['h'][:i+1], df_e['l'][:i+1], 10)[0].iloc[-1]))/max(a0,1e-9),
                            "don16_pos": (c0 - up0)/max(a0,1e-9),
                            "don24_pos": (c0 - float(_donchian(df_e['h'][:i+1], df_e['l'][:i+1], 24)[0].iloc[-1]))/max(a0,1e-9),
                            "v_ratio20": (v_now / max(v_avg,1e-9)) if v_avg>0 else 0.0,
                            "v_surge":  float(v_now >= 1.6*max(v_avg,1e-9)) if v_avg>0 else 0.0,
                            "range_atr": ((h0 - l0) / max(a0,1e-9)),
                            "r_room":    (room_buy if side=="BUY" else room_sel),
                        }

                        rows.append({
                            **feats, **extras,
                            "symbol": sym, "side": side, "family": fam,
                            "y": int(y), "y_short": int(y), "y_long": int(yL),
                            "r_best": float(r_best), "t_hit": int(t_hit), "hit_type": htyp,
                            "ts": df_e["time"].iloc[i],
                            "features_version": "v5-super",
                        })

            log.info(f"[{sym}] amostras acumuladas: {len(rows)}")

        if not rows:
            raise SystemExit("Sem amostras — ajuste sessão/parametros/gates.")

        df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower().endswith("parquet"):
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        log.info(f"Dataset salvo em: {out} | shape={df.shape}")

    finally:
        try: mt5.shutdown()
        except Exception: pass


if __name__ == "__main__":
    main()
