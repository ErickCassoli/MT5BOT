# fxbot/tools/build_dataset.py (v4)
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from core.logging import get_logger
from core.utils import atr, ema, adx, donchian as _donchian

log = get_logger(__name__)

_TF = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
}

def _fetch(sym: str, tf: str, n: int) -> pd.DataFrame:
    mt5.symbol_select(sym, True)
    rates = mt5.copy_rates_from_pos(sym, _TF[tf], 0, n)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos falhou p/ {sym}/{tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"open":"o","high":"h","low":"l","close":"c","tick_volume":"v"}, inplace=True)
    return df[["time","o","h","l","c","v"]]

def _label(exec_df: pd.DataFrame, i: int, side: str, m_sl: float, m_tp: float, ahead: int) -> int:
    """1 se TP antes de SL; 0 se SL antes de TP; -1 nenhum. Entrada na ABERTURA do (i+1); o candle (i+1) conta."""
    a = float(atr(exec_df["h"], exec_df["l"], exec_df["c"], 14).iloc[i])
    if not np.isfinite(a) or a <= 0: return -1
    j = i + 1
    if j >= len(exec_df): return -1
    entry = float(exec_df["o"].iloc[j])
    lows  = exec_df["l"].iloc[j:j+ahead]
    highs = exec_df["h"].iloc[j:j+ahead]
    if side == "BUY":
        sl, tp = entry - m_sl*a, entry + m_tp*a
        for lo, hi in zip(lows, highs):
            if lo <= sl: return 0
            if hi >= tp: return 1
    else:
        sl, tp = entry + m_sl*a, entry - m_tp*a
        for lo, hi in zip(lows, highs):
            if hi >= sl: return 0
            if lo <= tp: return 1
    return -1

def main():
    ap = argparse.ArgumentParser(description="Dataset v4 — features alinhadas ao live + src one-hots (VK/Don/BV).")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--tf-exec", default="M5")
    ap.add_argument("--tf-regime", default="H1")
    ap.add_argument("--donchian", type=int, default=16)
    ap.add_argument("--sr-win", type=int, default=24)
    ap.add_argument("--vol-lookback", type=int, default=20)
    ap.add_argument("--vol-mult", type=float, default=1.6)
    ap.add_argument("--adx-thr", type=float, default=18.0)
    ap.add_argument("--min-break-atr", type=float, default=0.12)
    ap.add_argument("--room-atr", type=float, default=1.40)
    ap.add_argument("--hour-start", type=int, default=7)   # UTC inclusive
    ap.add_argument("--hour-end", type=int, default=20)    # UTC exclusivo
    ap.add_argument("--atr-sl", type=float, default=1.6)
    ap.add_argument("--atr-tp", type=float, default=3.2)
    ap.add_argument("--ahead", type=int, default=24)
    ap.add_argument("--bars-exec", type=int, default=22000)
    ap.add_argument("--bars-regime", type=int, default=9000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not mt5.initialize():
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")

    rows: list[dict] = []
    try:
        for sym in args.symbols:
            log.info(f"[{sym}] baixando dados…")
            df_e = _fetch(sym, args.tf_exec, args.bars_exec)
            df_r = _fetch(sym, args.tf_regime, args.bars_regime)

            # sincroniza topos
            tmax = min(df_e["time"].iloc[-1], df_r["time"].iloc[-1])
            df_e = df_e[df_e["time"] <= tmax].reset_index(drop=True)
            df_r = df_r[df_r["time"] <= tmax].reset_index(drop=True)

            # filtro de sessão (UTC): [hs, he)
            hs, he = int(args.hour_start), int(args.hour_end)
            if hs < he:
                df_e = df_e[df_e["time"].dt.hour.between(hs, he-1, inclusive="both")].reset_index(drop=True)
            else:
                mask = (df_e["time"].dt.hour >= hs) | (df_e["time"].dt.hour < he)
                df_e = df_e[mask].reset_index(drop=True)

            # burn-in
            start = max(300, args.donchian + 30)
            end = len(df_e) - (args.ahead + 3)
            if end <= start:
                log.warning(f"[{sym}] poucos dados após filtros. Pulando.")
                continue

            # pré-cálculos M5
            a_m5   = atr(df_e["h"], df_e["l"], df_e["c"], 14)
            ema20  = ema(df_e["c"], 20)
            ema50  = ema(df_e["c"], 50)
            ema200 = ema(df_e["c"], 200)
            cumvol = df_e["v"].replace(0, np.nan).cumsum()
            vwap   = (df_e["c"] * df_e["v"]).cumsum() / cumvol
            vwap   = vwap.ffill().bfill()  # sem deprecation; garante valores no início também
            # Keltner (EMA20 de preço + ATR len=10, mult=1.8 — casa com VK)
            ema20_k = ema(df_e["c"], 20)
            atr_k   = atr(df_e["h"], df_e["l"], df_e["c"], 10)
            k_mid   = ema20_k
            k_up    = k_mid + 1.8 * atr_k
            k_low   = k_mid - 1.8 * atr_k

            # SR p/ BreakoutVolume
            sr_high = df_e["h"].rolling(args.sr_win, min_periods=args.sr_win).max()
            sr_low  = df_e["l"].rolling(args.sr_win, min_periods=args.sr_win).min()
            vol_avg = df_e["v"].rolling(args.vol_lookback if hasattr(args,'vol-lookback') else args.vol_lookback,
                                        min_periods=args.vol_lookback).mean()

            for i in range(start, end):
                t_i = df_e["time"].iloc[i]
                df_r_win = df_r[df_r["time"] <= t_i]
                if len(df_r_win) < 210:
                    continue

                # Regime H1
                adx_h1 = float(adx(df_r_win["h"], df_r_win["l"], df_r_win["c"], 14).iloc[-1])

                # M5 no i
                c0 = float(df_e["c"].iloc[i]); h0 = float(df_e["h"].iloc[i]); l0 = float(df_e["l"].iloc[i])
                a0 = float(a_m5.iloc[i]); 
                if not np.isfinite(a0) or a0 <= 0: 
                    continue

                ema20i, ema50i, ema200i = float(ema20.iloc[i]), float(ema50.iloc[i]), float(ema200.iloc[i])
                vw0 = float(vwap.iloc[i])
                ku0, km0, kl0 = float(k_up.iloc[i]), float(k_mid.iloc[i]), float(k_low.iloc[i])

                # Donchian (para a família Donchian)
                up_i, lo_i = _donchian(df_e["h"][:i+1], df_e["l"][:i+1], args.donchian)
                up0, lo0 = float(up_i.iloc[-1]), float(lo_i.iloc[-1])

                # BreakoutVolume auxiliares
                sr_hi = float(sr_high.iloc[i]) if np.isfinite(sr_high.iloc[i]) else np.nan
                sr_lo = float(sr_low.iloc[i])  if np.isfinite(sr_low.iloc[i]) else np.nan
                v_now = float(df_e["v"].iloc[i])
                v_avg = float(vol_avg.iloc[i]) if np.isfinite(vol_avg.iloc[i]) else np.nan
                vol_ratio = (v_now / v_avg) if (v_avg and v_avg > 0) else 0.0

                # Lógicas de “acendimento” por família
                # VK: acima/abaixo do VWAP + Keltner + margem e/ou pullback com room
                above_vwap = c0 > vw0
                below_vwap = c0 < vw0
                margin = args.min_break_atr * a0
                buy_break_vk  = above_vwap and (c0 > ku0 + margin or h0 > ku0 + margin)
                sell_break_vk = below_vwap and (c0 < kl0 - margin or l0 < kl0 - margin)
                in_upper = (c0 >= km0) and (c0 <= ku0)
                in_lower = (c0 <= km0) and (c0 >= kl0)
                room_buy = (ku0 - c0) / max(a0, 1e-9)
                room_sel = (c0 - kl0) / max(a0, 1e-9)
                buy_pull_vk  = above_vwap and in_upper and (room_buy >= args.room_atr)
                sell_pull_vk = below_vwap and in_lower and (room_sel >= args.room_atr)
                rsi_now = 50.0  # neutro por padrão (não usamos RSI real aqui)
                # (Se quiser, compute RSI aqui para dataset — no live VK usa RSI>50.)

                buy_vk  = (adx_h1 >= 14) and (buy_break_vk or buy_pull_vk)
                sell_vk = (adx_h1 >= 14) and (sell_break_vk or sell_pull_vk)

                # Donchian+ADX: break canal + filtro de médias
                buy_don  = (adx_h1 >= 18) and (ema20i > ema50i > ema200i) and (c0 >= up0 + margin)
                sell_don = (adx_h1 >= 18) and (ema20i < ema50i < ema200i) and (c0 <= lo0 - margin)

                # Breakout+Volume: rompe SR e volume alto
                buy_bv  = (adx_h1 >= 12) and np.isfinite(sr_hi) and (c0 > sr_hi + margin) and (vol_ratio >= args.vol_mult)
                sell_bv = (adx_h1 >= 12) and np.isfinite(sr_lo) and (c0 < sr_lo - margin) and (vol_ratio >= args.vol_mult)

                # Para cada família acionada, geramos uma amostra por lado elegível
                families = []
                if buy_vk or sell_vk:   families.append("vk")
                if buy_don or sell_don: families.append("don")
                if buy_bv or sell_bv:   families.append("bv")
                if not families: 
                    continue

                for side in ("BUY","SELL"):
                    fire = ((side=="BUY" and ((buy_vk) or (buy_don) or (buy_bv))) or
                            (side=="SELL" and ((sell_vk) or (sell_don) or (sell_bv))))
                    if not fire: 
                        continue

                    y = _label(df_e, i, side, args.atr_sl, args.atr_tp, args.ahead)
                    if y < 0: 
                        continue

                    for fam in families:
                        # schema fixo de features (igual ao live + src one-hots)
                        if fam == "vk":
                            feats = {
                                "adx_h1": adx_h1,
                                "rsi_m5": rsi_now,                    # no live VK usa RSI>50 (aqui neutro ou compute se quiser)
                                "atr_now": a0,
                                "c_kdist_up": (c0 - km0) / max(a0, 1e-9),
                                "c_kdist_low": (km0 - c0) / max(a0, 1e-9),
                                "near_vwap": float(abs(c0 - vw0) <= 0.35 * a0),
                                "confirm_ema20": float(c0 > ema20i),
                                "room_atr": room_buy if side=="BUY" else room_sel,
                                "break_dist": (c0 - ku0) / max(a0,1e-9) if side=="BUY" else (kl0 - c0) / max(a0,1e-9),
                                "ema20_50": 0.0, "ema50_200": 0.0,   # VK não envia isso no live
                                "vol_ratio": 0.0,
                                "src_vk":1.0, "src_don":0.0, "src_bv":0.0,
                            }
                        elif fam == "don":
                            feats = {
                                "adx_h1": adx_h1,
                                "rsi_m5": 0.0, "atr_now": a0,
                                "c_kdist_up": 0.0, "c_kdist_low": 0.0,
                                "near_vwap": 0.0, "confirm_ema20": 0.0,
                                "room_atr": 0.0,
                                "break_dist": (c0 - up0) / max(a0,1e-9) if side=="BUY" else (lo0 - c0) / max(a0,1e-9),
                                "ema20_50": (ema20i - ema50i), "ema50_200": (ema50i - ema200i),
                                "vol_ratio": 0.0,
                                "src_vk":0.0, "src_don":1.0, "src_bv":0.0,
                            }
                        else:  # bv
                            feats = {
                                "adx_h1": adx_h1,
                                "rsi_m5": 0.0, "atr_now": a0,
                                "c_kdist_up": 0.0, "c_kdist_low": 0.0,
                                "near_vwap": 0.0, "confirm_ema20": 0.0,
                                "room_atr": 0.0,
                                "break_dist": (c0 - sr_hi) / max(a0,1e-9) if side=="BUY" else (sr_lo - c0) / max(a0,1e-9),
                                "ema20_50": 0.0, "ema50_200": 0.0,
                                "vol_ratio": vol_ratio,
                                "src_vk":0.0, "src_don":0.0, "src_bv":1.0,
                            }

                        rows.append({
                            **feats,
                            "symbol": sym, "side": side, "y": int(y),
                            "ts": df_e["time"].iloc[i], "features_version": "v4",
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
