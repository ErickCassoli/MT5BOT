# Troubleshooting

### MT5: copy_rates_from_pos falhou: (-2, 'Terminal: Invalid params')
- MT5 não está aberto/logado OU símbolo/timeframe não existem.
- Garanta que o **Market Watch** mostra os pares e que o **MT5 está aberto**.

### Retcode 10016 (Invalid stops)
- SL/TP muito próximos. Use `atr_mult_sl`/`tp` maiores ou respeite `SYMBOL_TRADE_STOPS_LEVEL`.
- Verifique `point` e número de dígitos do ativo.

### Retcode 10009 (Request executed) mas sem movimento
- Ordem enviada com sucesso. Acompanhe em *Terminal > Negociação*.  
- Confirme se `volume` ≥ mínimo do símbolo.

### `skip: spread filter`
- Spread atual > limites (`hard_cap_points` ou `max_atr_ratio` × ATR).  
- Horários de **rollover** amplificam isso. Ajuste `max_atr_ratio` temporariamente se necessário.

### `ML-filtered | p=... < thr=...`
- Probabilidade do modelo ficou abaixo do `min_prob`.  
- Para mais fluxo, teste `min_prob: 0.38–0.45`. Para mais qualidade, suba.

### Parquet: `ImportError: pyarrow`
```
pip install pyarrow   # ou fastparquet
```

### Dataset builder muito pequeno
- Aumente `--bars-exec`/`--bars-regime` e inclua mais símbolos.  
- Em M5, 150k barras ≈ ~2 anos (depende do broker).

### Nada de ordens por muito tempo
- Veja `python -m fxbot.tools.selftest_live` para diagnóstico rápido (portas, spreads, distâncias, ADX).
- Diminua thresholds (ex.: `adx_thr`, `k_mult`, `min_prob`) e/ou afrouxe spread.