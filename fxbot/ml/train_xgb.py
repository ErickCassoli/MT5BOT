# fxbot/ml/train_xgb.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import warnings


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def ev_for_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float, rr: float, cost_r: float) -> float:
    """
    EV em unidades de 'R' por trade, assumindo:
      - ganho = +RR se y=1 (TP)
      - perda = -1 - cost_r se y=0 (SL + custos)
    Considera apenas trades com p >= thr.
    """
    mask = y_prob >= thr
    if mask.sum() == 0:
        return -1e9  # evita escolher threshold sem trades
    y_sel = y_true[mask]
    # EV médio = mean( y*RR - (1-y)*(1+cost_r) )
    gains = y_sel * rr
    losses = (1.0 - y_sel) * (1.0 + cost_r)
    ev = float(np.mean(gains - losses))
    return ev


def main():
    ap = argparse.ArgumentParser(description="Treino XGB calibrado para o MT5BOT (com seleção de threshold por EV).")
    ap.add_argument("--data", required=True, help="Caminho do dataset (.parquet ou .csv) com coluna 'ts' e 'y'")
    ap.add_argument("--out", required=True, help="Caminho de saída do modelo .pkl")
    ap.add_argument("--cv", type=int, default=5, help="n_splits do TimeSeriesSplit")
    ap.add_argument("--rr", type=float, default=2.0, help="Risk-Reward (TP/SL) p/ EV e limiar teórico")
    ap.add_argument("--cost-r", type=float, default=0.05, help="Custos em R por trade (slip/fees). Ex.: 0.05")
    ap.add_argument("--thr-min", type=float, default=0.40)
    ap.add_argument("--thr-max", type=float, default=0.70)
    ap.add_argument("--thr-step", type=float, default=0.01)
    ap.add_argument("--features", nargs="+", default=None,
                    help="Lista de features a usar (em ordem). Se omitido, usa colunas padrão conhecidas.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features-version", type=str, default=None,
                    help="Tag opcional (ex.: v5) salva em meta.features_version")
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    # Leitura
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
    else:
        df = pd.read_parquet(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])

    # Ordena temporalmente
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)

    if "y" not in df.columns:
        raise ValueError("Dataset precisa ter a coluna 'y' (rótulo 0/1).")

    # Features default (v5 sugerido)
    default_feats = [
        "adx_h1", "rsi_m5", "atr_now",
        "c_kdist_up", "c_kdist_low", "near_vwap", "confirm_ema20",
        "room_atr", "break_dist",
        "ema20_50", "ema50_200",
        "vol_ratio",
        "bb_z", "bb_width",
        "src_vk", "src_don", "src_bv", "src_scalper",
    ]
    feats = args.features or default_feats

    # Checagem e montagem de X, y
    missing = [f for f in feats if f not in df.columns]
    if missing:
        warnings.warn(f"As features a seguir não estão no dataset e serão preenchidas com 0: {missing}")
        for m in missing:
            df[m] = 0.0

    X = df[feats].astype(float).values
    y = df["y"].astype(int).values

    # Modelo base
    base = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        tree_method="hist",
        random_state=args.seed,
        # balanceamento simples
        scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1))
    )

    # CV temporal
    tscv = TimeSeriesSplit(n_splits=args.cv)
    aucs, aps, briers = [], [], []
    thr_grid = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)
    best_thrs = []

    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # Treina + calibra (cv interno para calibração; evita cv='prefit')
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(Xtr, ytr)

        p = model.predict_proba(Xva)[:, 1]
        aucs.append(roc_auc_score(yva, p))
        aps.append(average_precision_score(yva, p))
        briers.append(brier_score(yva, p))

        # escolhe threshold por EV no fold
        evs = [ev_for_threshold(yva, p, t, args.rr, args.cost_r) for t in thr_grid]
        best_t = float(thr_grid[int(np.argmax(evs))])
        best_thrs.append(best_t)

    print(f"CV ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"CV PR-AUC : {np.mean(aps):.3f} ± {np.std(aps):.3f}")
    print(f"CV Brier  : {np.mean(briers):.4f} ± {np.std(briers):.4f}")

    # Threshold por EV (mediana dos folds)
    thr_ev = float(np.median(best_thrs)) if best_thrs else float(1.0 / (1.0 + args.rr))
    print(f"Threshold ótimo por EV (mediana dos folds): {thr_ev:.3f}  (grid={args.thr_min:.2f}-{args.thr_max:.2f})")

    # Treino full + calibração final
    final_model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    final_model.fit(X, y)

    payload = {
        "model": final_model,
        "feature_names": feats,
        "threshold": thr_ev,
        "meta": {
            "features_version": args.features_version
        }
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    print(f"Modelo salvo em: {out}")
    print(f"Threshold salvo (por EV): {thr_ev:.3f}")


if __name__ == "__main__":
    main()
