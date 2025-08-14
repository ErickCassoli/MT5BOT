# FXBOT — Bot de Forex modular (Python + MT5)

Framework modular para trading em Forex com **MetaTrader 5** (conta demo ou real), focado em **M1/M5** e pronto para:
- Estratégias plugáveis (trend, breakout, scalper, ensemble).
- Filtros de **Machine Learning** (XGBoost já integrado).
- Gestão de risco (ATR SL/TP, parcial a 1R, trailing).
- Logs estruturados (CSV + JSON) e sessão 24/7.

> ⚠️ Aviso: Trading envolve risco. Use primeiro em **conta demo**.

---

## ✨ Features
- **Arquitetura limpa**: `adapters/` (broker), `strategies/`, `risk/`, `ml/`, `core/`, `tools/`.
- **Estratégias**: VWAP+Keltner, Donchian+ADX, Breakout+Volume, Scalper RSI+Bollinger, Ensemble.
- **MT5 Adapter**: envio de ordens, SL/TP, trailing, histórico e posições.
- **ML**: builder de dataset e treino de XGBoost; filtro por probabilidade.
- **Observabilidade**: `logs/trades.csv` (ordens/sinais) e `logs/session.json` (resumo).

---

## 📦 Requisitos
- Windows 10/11 (recomendado) com **MetaTrader 5** instalado e **logado**.
- Python 3.10+ (3.11 recomendado).
- Conta **demo** para testes.

---

## 🚀 Quickstart (5 minutos)

```bash
# 1) clonar
git clone https://github.com/ErickCassoli/MT5BOT.git
cd MT5BOT

# 2) ambiente
python -m venv .venv
.venv\Scripts\activate  # (Windows)

# 3) deps
pip install -r requirements.txt

# 4) configurar MT5
#   Abra o MT5, faça login em conta demo e mantenha o terminal aberto.
#   Siga docs/SETUP_MT5.md (habilitar "Algo Trading" e "Allow DLL imports").

# 5) configs
insira os parametros desejaveis em config.yaml

# 6) rodar ao vivo (conta demo)
python -m fxbot.run_live
````

Você deve ver algo como:

```
[ML] loaded=fxbot.ml.xgb_model.XGBModel thr=0.41
[SESSION] start=... end=... baseline=...
Live engine up.
[EURUSD] spread=... | ... | dyn_ok=True | ...
...
```

**Logs** serão gravados em:

* `logs/trades.csv` – cada sinal/ordem, retcodes, SL/TP.
* `logs/session.json` – resumo final (P\&L, winrate, PF).

---

## 🧠 (Opcional) Treinar o modelo de ML

1. **Gerar dataset** (VWAP+Keltner):

```bash
python -m fxbot.tools.build_dataset_vwap_keltner ^
  --symbols EURUSD GBPUSD USDJPY AUDUSD USDCAD EURJPY ^
  --tf-exec M5 --tf-regime H1 ^
  --k-ema 20 --k-atr 10 --k-mult 1.8 ^
  --adx-thr 12 --rsi-len 14 --rsi-trig 50 --near-vwap 0.35 ^
  --atr-sl 1.6 --atr-tp 3.2 --ahead 72 --stride 72 ^
  --bars-exec 150000 --bars-regime 60000 ^
  --out fxbot\data\vkbp.parquet
```

2. **Treinar XGBoost**:

```bash
python -m fxbot.ml.train_xgb --data fxbot\data\vkbp.parquet --out models\xgb.pkl --cv 5 --rr 2.0
# Saída esperada:
# CV ROC-AUC: ...
# Modelo salvo em: models\xgb.pkl | threshold_sugerido=0.33x
```

No `config.yaml`, a seção `ml_model` aponta para `models/xgb.pkl`. Você pode sobrescrever o threshold com `min_prob`.

---

## 🧩 Estratégias disponíveis

* `strategies.vwap_keltner.VWAPKeltner` — tendência intraday com bandas de Keltner + VWAP, filtro ADX/RSI.
* `strategies.donchian_adx.DonchianADX` — rompimento Donchian, regime H1 por EMA50/EMA200 e ADX.
* `strategies.breakout_volume.BreakoutVolume` — S/R (Donchian) com confirmação de volume (tick volume).
* `strategies.scalper_rsi_bb.ScalperRSIBB` — mean-reversion curto (RSI + Bollinger) evitando ADX alto.
* `strategies.ensemble_router.EnsembleRouter` — orquestra várias estratégias e escolhe melhor sinal.

Veja `docs/CONFIG.md` para parâmetros e exemplos.

---

## ⚙️ Configuração

O arquivo `config.yaml` controla símbolos, janelas, risco, regras de spread, sessão e ML.
Exemplos em `examples/configs/`. Documentação detalhada em `docs/CONFIG.md`.

---

## 🛠️ Troubleshooting

Problemas comuns (e soluções) em `docs/TROUBLESHOOTING.md`:

* `(-2, 'Terminal: Invalid params')` ao baixar candles.
* `10016 Invalid stops` ao enviar ordem.
* `pyarrow` ausente ao salvar `.parquet`.
* `skip: spread filter` (horários de spread alto).
* `ML-filtered | p=... < thr=...` (threshold alto).

---

## 📚 Roadmap curto

* Notebook `examples/notebooks/Intro.ipynb` (visão, backtest rápido, gráficos).
* Export de métricas em Prometheus e painel Grafana.
* Backtester interno com fills realistas.

---

## 🤝 Contribuição

PRs são bem-vindos! Abra issues com sugestões/bugs.
Padrão de código: PEP8, type hints, docstrings curtas nas estratégias.

---

## ⚖️ Licença

MIT. Veja `LICENSE`.

---


