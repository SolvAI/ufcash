"""Experiment 3: generate OOF stats-model predictions (yearly walk-forward), then stack with market signals."""
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import PROC, design, fighter_cols, load, logit, side_frame
from walk import DEV_YEARS, TEST_YEARS, summarize, walk_forward, per_year

warnings.filterwarnings("ignore")

MKT = {"mkt_close", "lg_mkt_close", "mkt_open", "lg_mkt_open", "line_move", "close_spread", "overround",
       "A_dec_imp", "B_dec_imp", "A_sub_imp", "B_sub_imp", "A_ko_imp", "B_ko_imp"}
OOF_YEARS = list(range(2008, 2027))


def sym(fn, te, fcols):
    Xo = side_frame(te, "r", "b", fcols, False)
    Xs = side_frame(te, "b", "r", fcols, True)
    return 0.5 * (fn(Xo) + 1 - fn(Xs))


def stats_lgb(fcols, params):
    def fp(tr, te):
        X, y, _ = design(tr, fcols)
        cols = [c for c in X.columns if c not in MKT]
        m = lgb.LGBMClassifier(**params).fit(X[cols], y)
        return sym(lambda Z: m.predict_proba(Z[cols])[:, 1], te, fcols)
    return fp


def gen_oof(d, fcols):
    path = PROC / "oof_stats.parquet"
    if path.exists() and "--regen" not in sys.argv:
        return pd.read_parquet(path).set_index("fight_id").reindex(d.fight_id).values[:, 0]
    small = dict(objective="binary", learning_rate=0.015, num_leaves=7, min_child_samples=150, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.3, reg_lambda=5.0, n_estimators=500, verbose=-1, n_jobs=16)
    p = walk_forward(d, stats_lgb(fcols, small), OOF_YEARS, fcols=fcols)
    pd.DataFrame({"fight_id": d.fight_id, "p_stats": p.values}).to_parquet(path)
    return p.values


def main():
    d = load()
    fcols = fighter_cols(d)
    d["p_stats"] = gen_oof(d, fcols)
    # market signals (red perspective)
    d["lg_close"] = logit(d.n_close_r)
    d["lg_ult"] = logit(d.mkt_p_red)
    d["lg_open"] = logit(d.mkt_open)
    d["lg_mkt"] = logit(d.mkt_close)            # close, fallback ult
    d["lg_ult_f"] = d.lg_ult.fillna(d.lg_mkt)
    d["lg_open_f"] = d.lg_open.fillna(d.lg_mkt)
    d["lg_stats"] = logit(d.p_stats)
    d["corner"] = np.where(d.date >= "2010-01-01", 1.0, 0.0)
    d["lg_elo"] = logit(d.p_elo_str)

    def stack(cols, C=1.0, fit_intercept=False):
        # orientation-symmetric LR: features are antisymmetric under swap, so augment with negation
        def fp(tr, te):
            tr = tr.dropna(subset=cols)
            Xt = tr[cols].values
            X = np.vstack([Xt, -Xt])
            y = np.concatenate([tr.y.values, 1 - tr.y.values])
            lr = LogisticRegression(C=C, fit_intercept=fit_intercept, max_iter=2000).fit(X, y)
            Z = te[cols].values
            ok = ~np.isnan(Z).any(axis=1)
            out = np.full(len(te), np.nan)
            out[ok] = lr.predict_proba(Z[ok])[:, 1]
            return out
        return fp

    combos = {
        "mkt": ["lg_mkt"],
        "close+ult": ["lg_mkt", "lg_ult_f"],
        "close+ult+open": ["lg_mkt", "lg_ult_f", "lg_open_f"],
        "mkt+corner": ["lg_mkt", "corner"],
        "mkt+stats": ["lg_mkt", "lg_stats"],
        "mkt3+stats": ["lg_mkt", "lg_ult_f", "lg_open_f", "lg_stats"],
        "mkt3+stats+corner": ["lg_mkt", "lg_ult_f", "lg_open_f", "lg_stats", "corner"],
        "mkt3+stats+elo+corner": ["lg_mkt", "lg_ult_f", "lg_open_f", "lg_stats", "lg_elo", "corner"],
    }
    rows = []
    years = DEV_YEARS + (TEST_YEARS if "--test" in sys.argv else [])
    for name, cols in combos.items():
        p = walk_forward(d, stack(cols), years, train_from="2010-01-01")
        rows.append(summarize(name, p, d, DEV_YEARS))
        if "--test" in sys.argv:
            rows[-1].update({f"T_{k}": v for k, v in summarize(name, p, d, TEST_YEARS).items() if k in ("acc_odds", "ll_odds", "mkt_acc", "n_odds")})
    r = summarize("stats_only", pd.Series(d.p_stats, index=d.index), d, DEV_YEARS)
    rows.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
