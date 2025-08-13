from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import importlib
import MetaTrader5 as mt5

import pandas as pd

from fxbot.core.config import load_config
from fxbot.adapters.mt5 import MT5Broker
from fxbot.core.utils import atr, donchian, ema


def import_from_path(path: str):
    mod, cls = path.rsplit(".", 1)
    # tenta com prefixo fxbot. e, se não achar, usa o path “cru”
    try:
        m = importlib.import_module(f"fxbot.{mod}")
    except ModuleNotFoundError:
        m = importlib.import_module(mod)
    return getattr(m, cls)


def get_server_time(sym: str):
    t = mt5.symbol_info_tick(sym)
    return datetime.utcfromtimestamp(int(t.time)) if t else None


def main():
    base = Path(__file__).resolve().parents[1]  # .../fxbot
    cfg = load_config(str(base / "config.yaml"))

    if not mt5.initialize():
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    broker = MT5Broker()
    broker.initialize()

    # Strategy & ML
    Strat = import_from_path(cfg.strategy.class_path)
    ml = None
    if cfg.ml_model:
        MLCls = import_from_path(cfg.ml_model["class_path"])
        ml = MLCls(**cfg.ml_model.get("params", {}))
    strat = Strat(cfg.strategy.params, ml_model=ml)
    ml_thr = getattr(ml, "threshold", 0.5) if ml else 0.5

    print(f"[SELFTEST] ml_thr={ml_thr:.3f}  window={cfg.session.start_hour}-{cfg.session.end_hour}h  "
          f"spread_cap={cfg.spread.hard_cap_points}  atr_ratio={cfg.spread.max_atr_ratio}")

    for symbol in cfg.symbols:
        print("\n---", symbol, "---")
        now_utc = datetime.now(timezone.utc)
        srv = get_server_time(symbol)
        in_window = (cfg.session.start_hour <=
                     now_utc.hour < cfg.session.end_hour)

        print(
            f"time_utc={now_utc} | server_time≈{srv} | gate_window={in_window}")

        df_e = broker.get_rates(symbol, cfg.timeframe_exec, 600)
        df_r = broker.get_rates(symbol, cfg.timeframe_regime, 600)
        a = float(atr(df_e["h"], df_e["l"], df_e["c"], 14).iloc[-1])

        spr_pts = broker.get_spread_points(symbol)
        spr_price = spr_pts * broker.get_point(symbol)
        dyn_cap = cfg.spread.max_atr_ratio * max(a, 1e-12)
        abs_ok = spr_pts <= cfg.spread.hard_cap_points
        dyn_ok = spr_price <= dyn_cap
        print(
            f"spread_pts={spr_pts} abs_ok={abs_ok} dyn_ok={dyn_ok} spr_price={spr_price:.6f} atr={a:.6f} dyn_cap={dyn_cap:.6f}")

        open_mine = [p for p in broker.positions(
            symbol) if p.magic == cfg.magic]
        print(
            f"gate_openpos={'BLOCK' if open_mine else 'OK'} open_count={len(open_mine)}")

        # Checagem de proximidade/break
        win = cfg.strategy.params.get("donchian", 16)
        up, lo = donchian(df_e["h"], df_e["l"], win)
        c0 = float(df_e["c"].iloc[-1])
        dist_up = float(max(0.0, up.iloc[-1] - c0))
        dist_low = float(max(0.0, c0 - lo.iloc[-1]))
        near_thr = float(cfg.strategy.params.get(
            "near_by_atr_ratio", 0.10) * a)
        above_ema20 = c0 > float(ema(df_e["c"], 20).iloc[-1])
        print(
            f"dist_up={dist_up:.6f} dist_low={dist_low:.6f} near_thr={near_thr:.6f} above_ema20={above_ema20}")

        # Sinal (sem enviar ordem)
        sig = strat.generate_signal(symbol, df_e, df_r)
        if sig is None:
            print("signal=None (regras da estratégia não passaram)")
            continue
        prob = float(sig.confidence)
        print(f"signal={sig.side.value} prob={prob:.3f} ml_thr={ml_thr:.3f} atr={sig.atr:.6f} adx_h1={sig.meta.get('adx_h1') if sig.meta else None}")
        print("pass_ml_filter=", (prob >= ml_thr))


if __name__ == "__main__":
    main()
