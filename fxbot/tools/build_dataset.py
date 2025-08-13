from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import MetaTrader5 as mt5

# nossas features e utils (rodando como módulo do pacote fxbot)
from fxbot.ml.features import compute_features
from fxbot.core.utils import atr, ema, adx

_TF = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
}


def fetch_rates(symbol: str, timeframe: str, n: int) -> pd.DataFrame:
    """Baixa N barras mais recentes do MT5 e padroniza colunas."""
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, _TF[timeframe], 0, n)
    if rates is None:
        raise RuntimeError(
            f"copy_rates_from_pos falhou para {symbol}/{timeframe}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'o', 'high': 'h', 'low': 'l',
              'close': 'c', 'tick_volume': 'v'}, inplace=True)
    return df


def label_triple_barrier(exec_df: pd.DataFrame, i: int, side: str,
                         atr_mult_sl: float, atr_mult_tp: float,
                         max_bars_ahead: int = 36) -> int:
    """
    Rótulo: 1 se atinge TP antes de SL, 0 se SL antes de TP, -1 se nenhum até o horizonte.
    Entrada simulada na ABERTURA da barra seguinte (i+1).
    """
    a = atr(exec_df['h'], exec_df['l'], exec_df['c'], 14).iloc[i]
    if pd.isna(a) or a <= 0:
        return -1
    # preço de entrada: abertura da barra seguinte
    entry_idx = i + 1
    if entry_idx >= len(exec_df):
        return -1
    entry = exec_df['o'].iloc[entry_idx]

    if side == "BUY":
        sl = entry - atr_mult_sl * a
        tp = entry + atr_mult_tp * a
        lo_seq = exec_df['l'].iloc[entry_idx+1: entry_idx+1+max_bars_ahead]
        hi_seq = exec_df['h'].iloc[entry_idx+1: entry_idx+1+max_bars_ahead]
        for lo, hi in zip(lo_seq, hi_seq):
            if lo <= sl:
                return 0
            if hi >= tp:
                return 1
    else:  # SELL
        sl = entry + atr_mult_sl * a
        tp = entry - atr_mult_tp * a
        lo_seq = exec_df['l'].iloc[entry_idx+1: entry_idx+1+max_bars_ahead]
        hi_seq = exec_df['h'].iloc[entry_idx+1: entry_idx+1+max_bars_ahead]
        for lo, hi in zip(lo_seq, hi_seq):
            if hi >= sl:
                return 0
            if lo <= tp:
                return 1
    return -1


def main():
    ap = argparse.ArgumentParser(
        description="Gera dataset rotulado (features + triple-barrier) a partir do MT5.")
    ap.add_argument("--symbols", nargs="+", required=True,
                    help="Ex.: EURUSD GBPUSD USDJPY")
    ap.add_argument("--tf-exec", default="M5")
    ap.add_argument("--tf-regime", default="H1")
    ap.add_argument("--donchian", type=int, default=16)
    ap.add_argument("--near-ratio", type=float, default=0.10)
    ap.add_argument("--adx-thr", type=float, default=18.0)
    ap.add_argument("--atr-sl", type=float, default=1.7)
    ap.add_argument("--atr-tp", type=float, default=3.4)
    ap.add_argument("--ahead", type=int, default=36,
                    help="Horizonte de barras M5 p/ avaliar TP/SL")
    ap.add_argument("--bars-exec", type=int, default=12000,
                    help="Barras M5 a baixar")
    ap.add_argument("--bars-regime", type=int,
                    default=5000, help="Barras H1 a baixar")
    ap.add_argument("--out", required=True,
                    help="Caminho de saída (.parquet ou .csv)")
    args = ap.parse_args()

    if not mt5.initialize():
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")

    all_rows = []
    for sym in args.symbols:
        print(f"[{sym}] baixando dados…")
        df_e = fetch_rates(sym, args.tf_exec, args.bars_exec)   # M5
        df_r = fetch_rates(sym, args.tf_regime, args.bars_regime)  # H1

        # sincroniza pelo menor timestamp final (segurança)
        min_time = min(df_e['time'].iloc[-1], df_r['time'].iloc[-1])
        df_e = df_e[df_e['time'] <= min_time].reset_index(drop=True)
        df_r = df_r[df_r['time'] <= min_time].reset_index(drop=True)

        # burn-in para ter indicadores estáveis (M5 e H1)
        start = max(args.donchian + 30, 300)  # ~mín. para donch/ema/atr no M5
        end = len(df_e) - (args.ahead + 3)
        if end <= start:
            print(
                f"[{sym}] poucos dados após burn-in/horizonte (start={start}, end={end}). Pulando.")
            continue

        print(f"[{sym}] varrendo janelas M5: {end - start} amostras candidatas…")
        for i in range(start, end):
            t_i = df_e['time'].iloc[i]

            # janela H1 alinhada no tempo (NÃO olha o futuro)
            df_r_win = df_r[df_r['time'] <= t_i].copy()
            if len(df_r_win) < 210:  # precisa de histórico p/ ema200
                continue

            ema50_h1 = ema(df_r_win['c'], 50)
            ema200_h1 = ema(df_r_win['c'], 200)
            adx_h1 = adx(df_r_win['h'], df_r_win['l'], df_r_win['c'], 14)

            uptrend = (df_r_win['c'].iloc[-1] > ema200_h1.iloc[-1]) and (
                ema50_h1.iloc[-1] > ema200_h1.iloc[-1]) and (adx_h1.iloc[-1] >= args.adx_thr)
            downtrend = (df_r_win['c'].iloc[-1] < ema200_h1.iloc[-1]) and (
                ema50_h1.iloc[-1] < ema200_h1.iloc[-1]) and (adx_h1.iloc[-1] >= args.adx_thr)

            # janela M5 até i (NÃO olha a barra seguinte)
            df_e_win = df_e.iloc[:i+1].copy()
            feats = compute_features(
                df_e_win, df_r_win, args.donchian, args.near_ratio)

            # mesma lógica de candidatura do live (near-break + confirmação EMA20)
            buy_cond = uptrend and (feats["buy_break_hl"] == 1.0 or (
                feats["frac_to_up"] <= 1.0 and feats["above_ema20"] == 1.0))
            sell_cond = downtrend and (feats["sell_break_hl"] == 1.0 or (
                feats["frac_to_low"] <= 1.0 and feats["above_ema20"] == 0.0))

            if buy_cond:
                y = label_triple_barrier(
                    df_e, i, "BUY", args.atr_sl, args.atr_tp, args.ahead)
                if y >= 0:
                    all_rows.append(
                        {**feats, "symbol": sym, "side": "BUY", "y": int(y), "ts": df_e['time'].iloc[i]})

            if sell_cond:
                y = label_triple_barrier(
                    df_e, i, "SELL", args.atr_sl, args.atr_tp, args.ahead)
                if y >= 0:
                    all_rows.append(
                        {**feats, "symbol": sym, "side": "SELL", "y": int(y), "ts": df_e['time'].iloc[i]})

        print(f"[{sym}] amostras coletadas até agora: {len(all_rows)}")

    if not all_rows:
        raise SystemExit(
            "Sem amostras geradas — ajuste parâmetros (donchian/near-ratio/adx-thr/ahead/barras).")

    df = pd.DataFrame(all_rows).sort_values("ts").reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        # requer pyarrow ou fastparquet
        df.to_parquet(out, index=False)

    print(f"Dataset salvo em: {out} | shape={df.shape}")


if __name__ == "__main__":
    main()
