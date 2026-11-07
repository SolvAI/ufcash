"""Experiment 4: stats-only (no market) model zoo, yearly walk-forward. OOF preds cached per variant.

usage: python exp4_stats.py VARIANT [VARIANT...] [--years dev|all]
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import PROC, design, fighter_cols, load, logit, side_frame, metrics

warnings.filterwarnings("ignore")
OOF_DIR = PROC / "oof"
OOF_DIR.mkdir(exist_ok=True)

MKT = {"mkt_close", "lg_mkt_close", "mkt_open", "lg_mkt_open", "line_move", "close_spread", "overround",
       "A_dec_imp", "B_dec_imp", "A_sub_imp", "B_sub_imp", "A_ko_imp", "B_ko_imp"}


def cols_all(X):
    return [c for c in X.columns if c not in MKT]


def cols_diff(X):
    keep = []
    for c in X.columns:
        if c in MKT:
            continue
        if c.startswith("D_") or c.startswith("M_") or c.startswith("lg_p_") or c in ("corner", "southpaw_vs_orth", "sched_rounds", "title", "women", "wc_limit"):
            keep.append(c)
    # a few absolute values matter non-linearly
    keep += [c for c in X.columns if c in ("A_age", "B_age", "A_n_fights", "B_n_fights", "A_pro_n", "B_pro_n", "A_rank", "B_rank")]
    return keep


def make_model(kind, params):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**{**dict(objective="binary", verbose=-1, n_jobs=4, subsample_freq=1), **params})
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(**{**dict(tree_method="hist", n_jobs=4, eval_metric="logloss"), **params})
    if kind == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**{**dict(verbose=0, thread_count=4, random_seed=0, allow_writing_files=False), **params})
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **params))
    raise ValueError(kind)


VARIANTS = {
    "lgb_small": ("lgb", cols_all, dict(learning_rate=0.015, num_leaves=7, min_child_samples=150, subsample=0.8, colsample_bytree=0.3, reg_lambda=5.0, n_estimators=500)),
    "lgb_small_diff": ("lgb", cols_diff, dict(learning_rate=0.015, num_leaves=7, min_child_samples=150, subsample=0.8, colsample_bytree=0.3, reg_lambda=5.0, n_estimators=500)),
    "lgb_stump_diff": ("lgb", cols_diff, dict(learning_rate=0.02, num_leaves=3, min_child_samples=200, subsample=0.7, colsample_bytree=0.3, reg_lambda=10.0, n_estimators=900)),
    "lgb_slow_diff": ("lgb", cols_diff, dict(learning_rate=0.005, num_leaves=7, min_child_samples=200, subsample=0.7, colsample_bytree=0.2, reg_lambda=10.0, n_estimators=1500)),
    "lgb_d4_diff": ("lgb", cols_diff, dict(learning_rate=0.01, max_depth=4, num_leaves=15, min_child_samples=100, subsample=0.7, colsample_bytree=0.25, reg_lambda=10.0, n_estimators=900)),
    "xgb_diff": ("xgb", cols_diff, dict(learning_rate=0.01, max_depth=3, min_child_weight=50, subsample=0.7, colsample_bytree=0.3, reg_lambda=10.0, n_estimators=900)),
    "cat_diff": ("cat", cols_diff, dict(learning_rate=0.03, depth=4, l2_leaf_reg=10, iterations=900, rsm=0.3)),
    "cat_all": ("cat", cols_all, dict(learning_rate=0.03, depth=4, l2_leaf_reg=10, iterations=900, rsm=0.2)),
    "lr_diff": ("lr", cols_diff, dict(C=0.003)),
    "resid_diff": ("resid", cols_diff, dict(learning_rate=0.01, num_leaves=4, min_child_samples=300, subsample=0.7, colsample_bytree=0.3, reg_lambda=20.0, n_estimators=250)),
    "resid_diff_long": ("resid", cols_diff, dict(learning_rate=0.005, num_leaves=4, min_child_samples=300, subsample=0.7, colsample_bytree=0.2, reg_lambda=30.0, n_estimators=600)),
    "resid_all": ("resid", cols_all, dict(learning_rate=0.01, num_leaves=4, min_child_samples=300, subsample=0.7, colsample_bytree=0.2, reg_lambda=20.0, n_estimators=250)),
    "lr_diff_C01": ("lr", cols_diff, dict(C=0.01)),
}


def fit_predict_year(d, fcols, kind, cols_fn, params, year, train_from):
    tr = d[(d.date >= train_from) & (d.date < f"{year}-01-01")]
    te = d[(d.date >= f"{year}-01-01") & (d.date < f"{year + 1}-01-01")]
    X, y, _ = design(tr, fcols)
    cols = cols_fn(X)
    if kind == "resid":  # boost from the market: learns only what the closing line misses
        import lightgbm as lgb
        ok = X.lg_mkt_close.notna().values
        m = lgb.LGBMClassifier(**{**dict(objective="binary", verbose=-1, n_jobs=4, subsample_freq=1), **params})
        m.fit(X[cols][ok], y[ok], init_score=X.lg_mkt_close.values[ok])
        def prr(Z):
            return 1 / (1 + np.exp(-(m.predict_proba(Z[cols], raw_score=True) + Z.lg_mkt_close.values)))
        Xo = side_frame(te, "r", "b", fcols, False)
        Xs = side_frame(te, "b", "r", fcols, True)
        return te.index.values, 0.5 * (prr(Xo) + 1 - prr(Xs))
    med = X[cols].median().fillna(0.0)
    m = make_model(kind, params)
    Xf = X[cols].fillna(med) if kind == "lr" else X[cols]
    m.fit(Xf, y)
    def pr(Z):
        Z = Z[cols].fillna(med) if kind == "lr" else Z[cols]
        return m.predict_proba(Z)[:, 1]
    Xo = side_frame(te, "r", "b", fcols, False)
    Xs = side_frame(te, "b", "r", fcols, True)
    return te.index.values, 0.5 * (pr(Xo) + 1 - pr(Xs))


def run(name, d, fcols, years, train_from="2001-01-01"):
    kind, cols_fn, params = VARIANTS[name]
    res = Parallel(n_jobs=min(len(years), 7))(delayed(fit_predict_year)(d, fcols, kind, cols_fn, params, y, train_from) for y in years)
    p = pd.Series(np.nan, index=d.index)
    for idx, v in res:
        p.loc[idx] = v
    pd.DataFrame({"fight_id": d.fight_id, name: p.values}).to_parquet(OOF_DIR / f"{name}.parquet")
    return p


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    years = list(range(2010, 2027)) if "--all" in sys.argv else list(range(2016, 2024))
    d = load()
    fcols = fighter_cols(d)
    for name in args:
        p = run(name, d, fcols, years)
        m = d.date.dt.year.isin(range(2016, 2024)) & p.notna() & d.mkt_close.notna()
        r = metrics(p[m], d.y[m])
        mk = metrics(d.mkt_close[m], d.y[m])
        print(f"{name:18s} dev acc {r['acc']:.4f} ll {r['logloss']:.4f} n {r['n']} | mkt acc {mk['acc']:.4f} ll {mk['logloss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
