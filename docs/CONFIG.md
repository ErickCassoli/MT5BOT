
# Guia do config.yaml

```yaml
symbols: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "EURJPY"]

timeframe_exec: "M5"       # timeframe de execução (sinais/ordens)
timeframe_regime: "H1"     # timeframe para filtros de regime (tendência, ADX)
cooldown_minutes: 6        # anti-reamorque em mesma direção/símbolo
magic: 5120812             # ID do robô nas posições
deviation_points: 20       # slippage permitido
log_every_bar: true        # prints a cada candle

strategy:
  class_path: "strategies.ensemble_router.EnsembleRouter"
  params:
    mode: "best_conf"
    children:
      - class_path: "strategies.vwap_keltner.VWAPKeltner"
        params: { k_ema_len: 20, k_atr_len: 10, k_mult: 1.8, adx_thr: 14, rsi_len: 14, rsi_trig: 50, near_vwap_by_atr: 0.35, confirm_ema20: true, allow_break_close: true, min_bars: 150, donchian: 16 }
      - class_path: "strategies.donchian_adx.DonchianADX"
        params: { donchian: 16, adx_thr: 18, ema_fast: 50, ema_slow: 200, min_bars: 150, allow_close_break: true, boost_strong_adx: 5 }
      - class_path: "strategies.breakout_volume.BreakoutVolume"
        params: { sr_win: 24, vol_lookback: 20, vol_mult: 1.6, adx_thr: 12, allow_retest: true, min_bars: 200 }
      - class_path: "strategies.scalper_rsi_bb.ScalperRSIBB"
        params: { rsi_len: 14, rsi_buy: 30, rsi_sell: 70, bb_len: 20, bb_k: 2.0, max_adx_h1: 18, min_bars: 120 }

broker:
  class_path: "adapters.mt5.MT5Broker"
  params: {}

session:
  hours: 24
  start_hour: 0
  end_hour: 24
  profit_target_pct: 10.0
  loss_limit_pct: 5.0
  continue_after_target: true
  max_concurrent_trades: 4
  allow_pyramiding: true
  max_stack_per_symbol: 3
  min_stack_increase_r: 0.6
  pyramiding_risk_scale: 0.5

risk:
  risk_per_trade_pct: 0.6
  risk_per_trade_pct_after_target: 0.25
  atr_mult_sl: 1.7
  atr_mult_tp: 3.4
  atr_trail_mult: 1.1
  partial_at_1r: 0.5

spread:
  hard_cap_points: 35
  max_atr_ratio: 0.90      # aceita spread <= 0.9×ATR (afrouxado; após validar, reduza para 0.75–0.85)

ml_model:
  class_path: "ml.xgb_model.XGBModel"
  params:
    path: "models/xgb.pkl"
    min_prob: 0.41         # sobrescreve threshold do arquivo
````

### Notas rápidas

* **`min_prob`**: se muitos sinais estiverem sendo barrados (`ML-filtered`), reduza de 0.41 → 0.38–0.40.
* **`max_atr_ratio`**: controla filtro de spread dinâmico; em horários ruins suba temporariamente, depois volte.
* **Pyramid**: só empilha se estiver em lucro ≥ `min_stack_increase_r` (em R) e na mesma direção.
* **Sessão 24/7**: com `continue_after_target: true`, ao bater a meta o risco cai para `risk_per_trade_pct_after_target`.