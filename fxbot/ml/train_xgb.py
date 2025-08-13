from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib

FEATURES = [
    "ema20_slope", "atr_m5", "dist_up", "dist_low", "near_thr",
    "frac_to_up", "frac_to_low", "above_ema20",
    "buy_break_hl", "sell_break_hl", "buy_close", "sell_close",
    "adx_h1", "ema50_over_200_h1", "trend_up_h1", "trend_dn_h1"
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="fxbot/data/dataset.parquet/.csv")
    ap.add_argument("--out", required=True, help="models/xgb.pkl")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--rr", type=float, default=2.0,
                    help="Risk-Reward (TP/SL) p/ threshold teórico (ex.: 2.0)")
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

    # class weight approx
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    scale_pos = (neg / max(pos, 1.0))

    base = XGBClassifier(
        n_estimators=400,
        max_depth=4,
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

    # time series CV
    tscv = TimeSeriesSplit(n_splits=args.cv)
    aucs, aps = [], []
    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = base.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        aucs.append(roc_auc_score(yva, p))
        aps.append(average_precision_score(yva, p))

    print(f"CV ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"CV PR-AUC : {np.mean(aps):.3f} ± {np.std(aps):.3f}")

    # treina full + calibração Platt (melhora probas)
    model = base.fit(X, y)
    calib = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    calib.fit(X, y)

    # threshold teórico p/ RR: entrar se p > 1/(1+RR)
    thr = 1.0/(1.0 + float(args.rr))

    payload = {
        "model": calib,
        "feature_names": FEATURES,
        "threshold": thr
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    print(f"Modelo salvo em: {out} | threshold_sugerido={thr:.3f}")


if __name__ == "__main__":
    main()
