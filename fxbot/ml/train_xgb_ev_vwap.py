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

FEATURES = [
    "atr_m5","dist_k_up","dist_k_lo","z_vwap","above_vwap",
    "adx_h1","trigger_break","trigger_pull"
]

def ev_per_trade(y: np.ndarray, p: np.ndarray, rr: float, cost_r: float, thr: float) -> float:
    sel = p >= thr
    if sel.sum() == 0:
        return -999.0
    y_sel = y[sel]
    return float((y_sel * rr - (1 - y_sel) * 1.0 - cost_r).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--cost-r", type=float, default=0.05)
    ap.add_argument("--thr-min", type=float, default=0.35)
    ap.add_argument("--thr-max", type=float, default=0.65)
    ap.add_argument("--thr-step", type=float, default=0.01)
    args = ap.parse_args()

    path = Path(args.data)
    if path.suffix.lower()==".csv":
        df = pd.read_csv(path, parse_dates=["ts"])
    else:
        df = pd.read_parquet(path)
        if "ts" in df.columns: df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    X = df[FEATURES].astype(float).values
    y = df["y"].astype(int).values

    pos = (y==1).sum(); neg = (y==0).sum()
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

    tscv = TimeSeriesSplit(n_splits=args.cv)
    aucs, aps = [], []
    thrs = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)
    ev_accum = np.zeros_like(thrs, dtype=float)

    for tr, va in tscv.split(X):
        m = base.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:,1]
        aucs.append(roc_auc_score(y[va], p))
        aps.append(average_precision_score(y[va], p))
        for i, t in enumerate(thrs):
            ev_accum[i] += ev_per_trade(y[va], p, rr=args.rr, cost_r=args.cost_r, thr=float(t))

    auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))
    ap_mean, ap_std = float(np.mean(aps)), float(np.std(aps))
    ev_mean = ev_accum / args.cv
    best_idx = int(np.argmax(ev_mean)); best_thr = float(thrs[best_idx]); best_ev = float(ev_mean[best_idx])

    print(f"CV ROC-AUC: {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"CV PR-AUC : {ap_mean:.3f} ± {ap_std:.3f}")
    print(f"EV grid [{args.thr_min:.2f},{args.thr_max:.2f}] step={args.thr_step}: best_thr={best_thr:.3f}, EV={best_ev:.4f} R/trade")

    # treina full + calibração
    m = base.fit(X, y)
    calib = CalibratedClassifierCV(m, method="sigmoid", cv=3)
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
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    print(f"Modelo salvo em: {out} | threshold_EV={best_thr:.3f}")

if __name__ == "__main__":
    main()
