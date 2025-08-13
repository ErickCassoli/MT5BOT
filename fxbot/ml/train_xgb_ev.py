from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib

FEATURES = [
    "ema20_slope", "atr_m5", "dist_up", "dist_low", "near_thr",
    "frac_to_up", "frac_to_low", "above_ema20",
    "buy_break_hl", "sell_break_hl", "buy_close", "sell_close",
    "adx_h1", "ema50_over_200_h1", "trend_up_h1", "trend_dn_h1"
]


def eval_ev(y_true: np.ndarray, p: np.ndarray, rr: float, cost_r: float, thr: float) -> float:
    """EV por trade em unidades de R para exemplos com p>=thr."""
    sel = p >= thr
    if sel.sum() == 0:
        return -999.0
    y = y_true[sel]
    # payoff: +RR se acerta, -1 se erra, -cost_r em todos
    ev = (y * rr - (1 - y) * 1.0 - cost_r).mean()
    return float(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="fxbot/data/dataset.parquet ou .csv")
    ap.add_argument("--out", required=True, help="fxbot/models/xgb.pkl")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--rr", type=float, default=2.0,
                    help="Risk-Reward (TP/SL)")
    ap.add_argument("--cost-r", type=float, default=0.05,
                    help="custo por trade em R (spread+comissão / SL)")
    ap.add_argument("--thr-min", type=float, default=0.30)
    ap.add_argument("--thr-max", type=float, default=0.80)
    ap.add_argument("--thr-step", type=float, default=0.01)
    args = ap.parse_args()

    path = Path(args.data)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["ts"])
    else:
        df = pd.read_parquet(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    X = df[FEATURES].astype(float).values
    y = df["y"].astype(int).values

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos = (neg / max(pos, 1.0))

    base = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        scale_pos_weight=scale_pos
    )

    # TimeSeries CV: métricas padrão + busca do melhor threshold por EV
    tscv = TimeSeriesSplit(n_splits=args.cv)
    aucs, aps = [], []
    thrs = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)
    evs = np.zeros_like(thrs, dtype=float)

    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = base.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        aucs.append(roc_auc_score(yva, p))
        aps.append(average_precision_score(yva, p))
        for i, t in enumerate(thrs):
            evs[i] += eval_ev(yva, p, rr=args.rr,
                              cost_r=args.cost_r, thr=float(t))

    auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))
    ap_mean, ap_std = float(np.mean(aps)), float(np.std(aps))
    evs_mean = evs / args.cv
    best_idx = int(np.argmax(evs_mean))
    best_thr = float(thrs[best_idx])
    best_ev = float(evs_mean[best_idx])

    print(f"CV ROC-AUC: {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"CV PR-AUC : {ap_mean:.3f} ± {ap_std:.3f}")
    print(
        f"EV-threshold search [{args.thr_min:.2f},{args.thr_max:.2f}] step={args.thr_step}:")
    print(f" -> best_thr={best_thr:.3f} | EV={best_ev:.4f} R/trade (avg CV)")

    # treina full + calibração e salva threshold ótimo
    model = base.fit(X, y)
    calib = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    calib.fit(X, y)

    payload = {
        "model": calib,
        "feature_names": FEATURES,
        "threshold": best_thr,
        "meta": {
            "cv_roc_auc_mean": auc_mean, "cv_roc_auc_std": auc_std,
            "cv_pr_auc_mean": ap_mean, "cv_pr_auc_std": ap_std,
            "rr": args.rr, "cost_r": args.cost_r, "best_ev": best_ev,
            "thr_grid": {"min": args.thr_min, "max": args.thr_max, "step": args.thr_step}
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    print(f"Modelo salvo em: {out} | threshold_EV={best_thr:.3f}")


if __name__ == "__main__":
    main()
