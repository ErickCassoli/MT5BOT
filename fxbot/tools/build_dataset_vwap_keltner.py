from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from core.utils import ema, atr, adx
from core.logging import get_logger

_TF = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
}


def fetch_rates_chunked(symbol: str, timeframe: str, total: int, chunk: int = 20000) -> pd.DataFrame:
    """
    Baixa 'total' barras em lotes para evitar 'Invalid params' do MT5.
    Padrão: pega dos candles mais recentes para trás (pos=0,20k,40k,...).
    """
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(
            f"symbol_select falhou para {symbol}: {mt5.last_error()}")

    tf = _TF[timeframe]
    got = []
    pos = 0
    remaining = int(total)
    while remaining > 0:
        cnt = min(chunk, remaining)
        rates = mt5.copy_rates_from_pos(symbol, tf, pos, cnt)
        if rates is None:
            err = mt5.last_error()
            # Se falhar logo no início, reporte erro direto
            if pos == 0:
                raise RuntimeError(
                    f"copy_rates_from_pos falhou: {symbol}/{timeframe} -> {err}")
            # Se falhar depois, sai do loop (não há mais dados)
            break
        if len(rates) == 0:
            break
        got.append(pd.DataFrame(rates))
        pos += len(rates)  # anda o cursor
        remaining -= len(rates)
        if len(rates) < cnt:
            # não há mais dados disponíveis
            break

    if not got:
        raise RuntimeError(
            f"Sem dados retornados para {symbol}/{timeframe}. Último erro: {mt5.last_error()}")

    df = pd.concat(got, ignore_index=True)
    # normaliza colunas e ordena do mais ANTIGO para o mais NOVO
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"open": "o", "high": "h", "low": "l",
              "close": "c", "tick_volume": "v"}, inplace=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def rsi(df_close: pd.Series, n: int = 14) -> pd.Series:
    delta = df_close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1.0/n, adjust=False).mean()
    loss = dn.ewm(alpha=1.0/n, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def vwap_session(df: pd.DataFrame) -> pd.Series:
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    vol = df["v"] if "v" in df.columns else pd.Series(1.0, index=df.index)
    if "time" in df.columns:
        day = pd.to_datetime(df["time"]).dt.date
        pv = (tp * vol).groupby(day).cumsum()
        vv = vol.groupby(day).cumsum().replace(0, np.nan)
        return pv / vv
    pv = (tp * vol).cumsum()
    vv = vol.cumsum().replace(0, np.nan)
    return pv / vv


def keltner(df: pd.DataFrame, ema_len: int, atr_len: int, mult: float):
    mid = ema(df["c"], ema_len)
    a = atr(df["h"], df["l"], df["c"], atr_len)
    up = mid + mult * a
    lo = mid - mult * a
    return mid, up, lo, a


def label_triple_barrier(exec_df: pd.DataFrame, i: int, side: str,
                         atr_mult_sl: float, atr_mult_tp: float,
                         max_bars_ahead: int) -> int:
    a = atr(exec_df["h"], exec_df["l"], exec_df["c"], 14).iloc[i]
    if pd.isna(a) or a <= 0:
        return -1
    entry_idx = i + 1
    if entry_idx >= len(exec_df):
        return -1
    entry = exec_df["o"].iloc[entry_idx]
    if side == "BUY":
        sl = entry - atr_mult_sl * a
        tp = entry + atr_mult_tp * a
        lo_seq = exec_df["l"].iloc[entry_idx+1: entry_idx+1+max_bars_ahead]
        hi_seq = exec_df["h"].iloc[entry_idx+1: entry_idx+1+max_bars_ahead]
        for lo, hi in zip(lo_seq, hi_seq):
            if lo <= sl:
                return 0
            if hi >= tp:
                return 1
    else:
        sl = entry + atr_mult_sl * a
        tp = entry - atr_mult_tp * a
        lo_seq = exec_df["l"].iloc[entry_idx+1: entry_idx+1+max_bars_ahead]
        hi_seq = exec_df["h"].iloc[entry_idx+1: entry_idx+1+max_bars_ahead]
        for hi, lo in zip(hi_seq, lo_seq):
            if hi >= sl:
                return 0
            if lo <= tp:
                return 1
    return -1


log = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser(
        description="Build dataset VKBP (VWAP + Keltner) com triple-barrier.")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--tf-exec", default="M5")
    ap.add_argument("--tf-regime", default="H1")

    ap.add_argument("--k-ema", type=int, default=20)
    ap.add_argument("--k-atr", type=int, default=10)
    ap.add_argument("--k-mult", type=float, default=1.8)
    ap.add_argument("--adx-thr", type=float, default=12.0)
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--rsi-trig", type=float, default=50.0)
    ap.add_argument("--near-vwap", type=float, default=0.35)

    ap.add_argument("--atr-sl", type=float, default=1.6)
    ap.add_argument("--atr-tp", type=float, default=3.2)
    ap.add_argument("--ahead", type=int, default=72)

    ap.add_argument("--bars-exec", type=int, default=150000)
    ap.add_argument("--bars-regime", type=int, default=60000)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=20000,
                    help="tamanho do lote p/ download de barras")
    ap.add_argument("--out", required=True)

    args = ap.parse_args()
    stride = args.stride or args.ahead

    if not mt5.initialize():
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")

    rows = []
    for sym in args.symbols:
        log.info(f"[{sym}] baixando dados…")
        df_e = fetch_rates_chunked(
            sym, args.tf_exec, args.bars_exec, args.chunk_size)
        df_r = fetch_rates_chunked(
            sym, args.tf_regime, args.bars_regime, args.chunk_size)

        min_time = min(df_e["time"].iloc[-1], df_r["time"].iloc[-1])
        df_e = df_e[df_e["time"] <= min_time].reset_index(drop=True)
        df_r = df_r[df_r["time"] <= min_time].reset_index(drop=True)

        start = max(200, args.k_ema + 50)
        end = len(df_e) - (args.ahead + 3)
        if end <= start:
            log.warning(f"[{sym}] poucos dados úteis (start={start}, end={end}). Pulando.")
            continue

        log.info(f"[{sym}] varrendo M5: {((end-start)//stride)+1} amostras…")
        for i in range(start, end, stride):
            t_i = df_e["time"].iloc[i]
            df_r_win = df_r[df_r["time"] <= t_i].copy()
            if len(df_r_win) < 210:
                continue

            ema50_h1 = ema(df_r_win["c"], 50)
            ema200_h1 = ema(df_r_win["c"], 200)
            adx_h1 = adx(df_r_win["h"], df_r_win["l"], df_r_win["c"], 14)
            uptrend = (df_r_win["c"].iloc[-1] > ema200_h1.iloc[-1]) and (
                ema50_h1.iloc[-1] > ema200_h1.iloc[-1]) and (adx_h1.iloc[-1] >= args.adx_thr)
            downtrend = (df_r_win["c"].iloc[-1] < ema200_h1.iloc[-1]) and (
                ema50_h1.iloc[-1] < ema200_h1.iloc[-1]) and (adx_h1.iloc[-1] >= args.adx_thr)
            if not (uptrend or downtrend):
                continue

            dfw = df_e.iloc[:i+1].copy()
            mid, kup, klo, a = keltner(
                dfw, args.k_ema, args.k_atr, args.k_mult)
            a0 = float(a.iloc[-1])
            if a0 <= 0 or np.isnan(a0):
                continue
            vwap = vwap_session(dfw)
            rsi_ser = rsi(dfw["c"], args.rsi_len)

            c0 = float(dfw["c"].iloc[-1])
            up0 = float(kup.iloc[-1])
            lo0 = float(klo.iloc[-1])
            vwap0 = float(vwap.iloc[-1])
            rsi0 = float(rsi_ser.iloc[-1])
            ema20_now = float(ema(dfw["c"], 20).iloc[-1])

            ema20_ok = (c0 > ema20_now) if uptrend else (c0 < ema20_now)
            near_thr = args.near_vwap * a0
            near_vwap = abs(c0 - vwap0) <= near_thr

            long_break = uptrend and (c0 > up0) and ema20_ok and (c0 > vwap0)
            short_break = downtrend and (
                c0 < lo0) and ema20_ok and (c0 < vwap0)

            long_pull = uptrend and near_vwap and (
                rsi0 >= args.rsi_trig) and (c0 > vwap0) and ema20_ok
            short_pull = downtrend and near_vwap and (
                rsi0 <= (100-args.rsi_trig)) and (c0 < vwap0) and ema20_ok

            def add_row(side: str, trigger: str):
                y = label_triple_barrier(
                    df_e, i, side, args.atr_sl, args.atr_tp, args.ahead)
                if y < 0:
                    return
                rows.append({
                    "symbol": sym,
                    "ts": dfw["time"].iloc[-1],
                    "side": side,
                    "y": int(y),
                    "atr_m5": a0,
                    "dist_k_up": float(max(0.0, up0 - c0)),
                    "dist_k_lo": float(max(0.0, c0 - lo0)),
                    "z_vwap": float((c0 - vwap0) / max(a0, 1e-12)),
                    "above_vwap": float(1.0 if c0 > vwap0 else 0.0),
                    "adx_h1": float(adx_h1.iloc[-1]),
                    "trigger_break": float(1.0 if trigger == "break" else 0.0),
                    "trigger_pull": float(1.0 if trigger == "pullback" else 0.0),
                })

            if long_break:
                add_row("BUY",  "break")
            if short_break:
                add_row("SELL", "break")
            if long_pull:
                add_row("BUY",  "pullback")
            if short_pull:
                add_row("SELL", "pullback")

        log.info(f"[{sym}] amostras acumuladas: {len(rows)}")

    if not rows:
        raise SystemExit(
            "Sem amostras geradas — ajuste parâmetros (k, near-vwap, adx, ahead/stride/barras).")

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    log.info(f"Dataset VKBP salvo em: {out} | shape={df.shape}")


if __name__ == "__main__":
    main()
