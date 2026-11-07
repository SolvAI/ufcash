"""Experiment 2: stats model variants + market combination, walk-forward on DEV years."""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import design, fighter_cols, load, logit, side_frame
from walk import DEV_YEARS, summarize, walk_forward

warnings.filterwarnings("ignore")

MKT = {"mkt_close", "lg_mkt_close", "mkt_open", "lg_mkt_open", "line_move", "close_spread", "overround",
       "A_dec_imp", "B_dec_imp", "A_sub_imp", "B_sub_imp", "A_ko_imp", "B_ko_imp"}


def sym(fn, te, fcols):
    Xo = side_frame(te, "r", "b", fcols, False)
    Xs = side_frame(te, "b", "r", fcols, True)
    return 0.5 * (fn(Xo) + 1 - fn(Xs))


def lgb_fp(cols_fn, params, fcols, init_market=False, weight_halflife=None):
    def fp(tr, te):
        X, y, idx = design(tr, fcols)
        cols = cols_fn(X)
        w = None
        if weight_halflife:
            age = (tr.date.max() - tr.date.iloc[idx]).dt.days.values / 365.25
            w = 0.5 ** (age / weight_halflife)
        if init_market:
            ok = X.mkt_close.notna().values
            X, y = X[ok], y[ok]
            w = w[ok] if w is not None else None
            m = lgb.LGBMClassifier(**params).fit(X[cols], y, sample_weight=w, init_score=X.lg_mkt_close.values)
            fn = lambda Z: 1 / (1 + np.exp(-(m.predict_proba(Z[cols], raw_score=True) + Z.lg_mkt_close.values)))
            p = sym(fn, te, fcols)
            mk = te.mkt_close.values
            return np.where(np.isnan(mk), np.nan, p)
        m = lgb.LGBMClassifier(**params).fit(X[cols], y, sample_weight=w)
        return sym(lambda Z: m.predict_proba(Z[cols])[:, 1], te, fcols)
    return fp


def lr_fp(cols_fn, C, fcols):
    def fp(tr, te):
        X, y, _ = design(tr, fcols)
        cols = cols_fn(X)
        med = X[cols].median()
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[cols].fillna(med), y)
        return sym(lambda Z: lr.predict_proba(Z[cols].fillna(med))[:, 1], te, fcols)
    return fp


def main():
    d = load()
    fcols = fighter_cols(d)
    no_mkt = lambda X: [c for c in X.columns if c not in MKT]
    diffs_only = lambda X: [c for c in X.columns if (c.startswith("D_") or c.startswith("lg_p_") or c in ("corner", "southpaw_vs_orth")) and c not in MKT and c != "D_rank"]
    base = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.5, reg_lambda=5.0, n_estimators=600, verbose=-1, n_jobs=16)
    small = dict(base, num_leaves=7, min_child_samples=150, n_estimators=500, learning_rate=0.015, colsample_bytree=0.3)
    resid = dict(objective="binary", learning_rate=0.01, num_leaves=5, min_child_samples=200, subsample=0.7, subsample_freq=1,
                 colsample_bytree=0.3, reg_lambda=20.0, n_estimators=150, verbose=-1, n_jobs=16)
    variants = {
        "mkt": None,
        "lgb_nomkt": lgb_fp(no_mkt, base, fcols),
        "lgb_nomkt_small": lgb_fp(no_mkt, small, fcols),
        "lgb_diffs_small": lgb_fp(diffs_only, small, fcols),
        "lgb_nomkt_small_w4": lgb_fp(no_mkt, small, fcols, weight_halflife=4),
        "lr_diffs_C.01": lr_fp(diffs_only, 0.01, fcols),
        "lr_diffs_C.001": lr_fp(diffs_only, 0.001, fcols),
        "resid_lgb": lgb_fp(no_mkt, resid, fcols, init_market=True),
    }
    only = sys.argv[1:] or list(variants)
    rows, preds = [], {}
    for name in only:
        fp = variants[name]
        p = d.mkt_close.copy() if fp is None else walk_forward(d, fp, DEV_YEARS, fcols=fcols)
        preds[name] = p
        rows.append(summarize(name, p, d, DEV_YEARS))
        print(rows[-1], flush=True)
    pd.DataFrame(preds).assign(fight_id=d.fight_id).to_parquet("../data/processed/exp2_dev_preds.parquet")
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
