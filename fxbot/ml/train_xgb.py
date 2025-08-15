from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib


RESERVED_COLS = {"y", "ts", "symbol", "side", "features_version"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Treino XGB + calibração + threshold por EV (alinhado ao live).")
    ap.add_argument("--data", required=True, help="datasets/v3_m5_h1_day.parquet (ou .csv)")
    ap.add_argument("--out", required=True, help="models/xgb_vkbp_v3.pkl")
    ap.add_argument("--cv", type=int, default=5, help="n_splits do TimeSeriesSplit")
    ap.add_argument("--rr", type=float, default=2.0, help="Risk/Reward (TP/SL). Ex.: 2.0")
    ap.add_argument("--cost-r", type=float, default=0.05, help="Custo médio por trade em unidades de R")
    ap.add_argument("--thr-min", type=float, default=0.40)
    ap.add_argument("--thr-max", type=float, default=0.70)
    ap.add_argument("--thr-step", type=float, default=0.01)
    ap.add_argument("--features", nargs="+", default=None,
                    help="Lista explícita de features (recomendado). "
                         "Ex.: adx_h1 rsi_m5 atr_now c_kdist_up c_kdist_low near_vwap confirm_ema20 room_atr break_dist ema20_50 ema50_200 vol_ratio")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["ts"]) if "ts" in open(path, "r", encoding="utf-8").readline() else pd.read_csv(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
    else:
        df = pd.read_parquet(path)
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def pick_features(df: pd.DataFrame, feats: Sequence[str] | None) -> List[str]:
    if feats is not None:
        missing = [f for f in feats if f not in df.columns]
        if missing:
            raise SystemExit(f"Features ausentes no dataset: {missing}")
        return list(feats)
    # fallback: tudo que não é reservado e é numérico
    cand = [c for c in df.columns if c not in RESERVED_COLS]
    num = [c for c in cand if np.issubdtype(df[c].dtype, np.number)]
    return num


def ev_per_trade(winrate: float, rr: float, cost_r: float) -> float:
    # EV (em R) quando um trade é executado
    # R/R -> ganho = rr*1, perda = 1
    return rr * winrate - (1.0 - winrate) - cost_r


def grid_threshold_by_ev(y_true: np.ndarray, p: np.ndarray, thr_grid: np.ndarray,
                         rr: float, cost_r: float) -> Tuple[float, Dict[float, float]]:
    best_thr = thr_grid[0]
    best_ev = -np.inf
    ev_map: Dict[float, float] = {}
    for thr in thr_grid:
        sel = p >= thr
        if not np.any(sel):
            ev_map[float(thr)] = -np.inf
            continue
        wr = float(np.mean(y_true[sel] == 1))
        ev = ev_per_trade(wr, rr, cost_r)
        ev_map[float(thr)] = ev
        if ev > best_ev:
            best_ev, best_thr = ev, thr
    return float(best_thr), ev_map


def main() -> None:
    args = parse_args()
    path = Path(args.data)
    df = load_dataset(path)

    feats = pick_features(df, args.features)
    X = df[feats].astype(float).values
    y = df["y"].astype(int).values

    # balanceamento aproximado
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

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
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
    )

    tscv = TimeSeriesSplit(n_splits=args.cv)

    aucs, aps, briers, best_thrs = [], [], [], []
    thr_grid = np.arange(args.thr_min, args.thr_max + 1e-12, args.thr_step)

    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # calibrado sem usar cv='prefit' (evita FutureWarning)
        calib = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calib.fit(Xtr, ytr)

        p = calib.predict_proba(Xva)[:, 1]
        try:
            aucs.append(roc_auc_score(yva, p))
        except Exception:
            aucs.append(np.nan)
        aps.append(average_precision_score(yva, p))
        briers.append(brier_score_loss(yva, p))

        thr_star, _ = grid_threshold_by_ev(yva, p, thr_grid, rr=args.rr, cost_r=args.cost_r)
        best_thrs.append(thr_star)

    auc_mean, auc_std = np.nanmean(aucs), np.nanstd(aucs)
    ap_mean, ap_std = np.mean(aps), np.std(aps)
    br_mean, br_std = np.mean(briers), np.std(briers)
    thr_med = float(np.median(best_thrs)) if best_thrs else float(args.thr_min)

    print(f"CV ROC-AUC: {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"CV PR-AUC : {ap_mean:.3f} ± {ap_std:.3f}")
    print(f"CV Brier  : {br_mean:.4f} ± {br_std:.4f}")
    print(f"Threshold ótimo por EV (mediana dos folds): {thr_med:.3f}  (grid={args.thr_min:.2f}-{args.thr_max:.2f})")

    # treino final + calibração
    calib_final = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calib_final.fit(X, y)

    payload = {
        "model": calib_final,
        "feature_names": feats,
        "threshold": thr_med,
        "meta": {
            "cv_auc_mean": float(auc_mean),
            "cv_pr_mean": float(ap_mean),
            "cv_brier_mean": float(br_mean),
            "rr": float(args.rr),
            "cost_r": float(args.cost_r),
            "features_version": "v3",
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    print(f"Modelo salvo em: {out}")
    print(f"Threshold salvo (por EV): {thr_med:.3f}")


if __name__ == "__main__":
    main()
