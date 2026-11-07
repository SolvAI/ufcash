"""The final model: stats ensemble (no market input) + market stacker.

  stats ensemble   5 models (LightGBM x2, XGBoost, CatBoost, logistic regression) on leakage-free pre-fight
                   features, trained on both corner orientations; output = mean logit  ->  lg_ens
  stacker          average (in logit space) of
                     (a) logistic regression on [logit(closing line), lg_ens]   (no intercept, antisymmetric)
                     (b) LightGBM boosted from the closing-line logit (learns systematic market errors)
                   fights without odds fall back to the stats ensemble.

Selection of every choice here was done on walk-forward 2016-2023; 2024-2026 is the hold-out.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import design, logit, side_frame

warnings.filterwarnings("ignore")

MKT = {"mkt_close", "lg_mkt_close", "mkt_open", "lg_mkt_open", "line_move", "close_spread", "overround",
       "A_dec_imp", "B_dec_imp", "A_sub_imp", "B_sub_imp", "A_ko_imp", "B_ko_imp"}


def cols_all(X):
    return [c for c in X.columns if c not in MKT]


def cols_diff(X):
    keep = [c for c in X.columns if c not in MKT and (c.startswith(("D_", "M_", "lg_p_")) or c in
            ("corner", "southpaw_vs_orth", "sched_rounds", "title", "women", "wc_limit"))]
    keep += [c for c in X.columns if c in ("A_age", "B_age", "A_n_fights", "B_n_fights", "A_pro_n", "B_pro_n", "A_rank", "B_rank")]
    return keep


STATS_VARIANTS = {
    "lgb_small": ("lgb", cols_all, dict(learning_rate=0.015, num_leaves=7, min_child_samples=150, subsample=0.8, colsample_bytree=0.3, reg_lambda=5.0, n_estimators=500)),
    "lgb_d4_diff": ("lgb", cols_diff, dict(learning_rate=0.01, max_depth=4, num_leaves=15, min_child_samples=100, subsample=0.7, colsample_bytree=0.25, reg_lambda=10.0, n_estimators=900)),
    "xgb_diff": ("xgb", cols_diff, dict(learning_rate=0.01, max_depth=3, min_child_weight=50, subsample=0.7, colsample_bytree=0.3, reg_lambda=10.0, n_estimators=900)),
    "lr_diff": ("lr", cols_diff, dict(C=0.003)),
    "cat_all": ("cat", cols_all, dict(learning_rate=0.03, depth=4, l2_leaf_reg=10, iterations=900, rsm=0.2)),
}
RESID_PARAMS = dict(learning_rate=0.01, num_leaves=4, min_child_samples=300, subsample=0.7, colsample_bytree=0.5, reg_lambda=20.0, n_estimators=200)
ANTI = ["lg_ens", "lg_mkt", "lg_ult_f", "lg_open_f", "corner", "D_n_fights", "D_pro_n", "D_age", "D_welo", "D_reach", "D_rank", "D_days_since",
        "D_c_adj_net", "D_e_sdiff", "D_elo_str", "D_card_pos_mean", "D_mk_surprise", "D_mk_mean_imp", "D_mk_last_imp", "D_mk_surprise3"]
SYM = ["women", "wc_limit", "sched_rounds", "title", "close_spread", "card_pos", "year"]


def make_model(kind, params, n_jobs=4):
    if kind == "lgb":
        return lgb.LGBMClassifier(**{**dict(objective="binary", verbose=-1, n_jobs=n_jobs, subsample_freq=1), **params})
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(**{**dict(tree_method="hist", n_jobs=n_jobs, eval_metric="logloss"), **params})
    if kind == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**{**dict(verbose=0, thread_count=n_jobs, random_seed=0, allow_writing_files=False), **params})
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **params))
    raise ValueError(kind)


# ------------------------------------------------------------------ stats ensemble
def fit_stats_one(tr, fcols, name, n_jobs=4):
    kind, cols_fn, params = STATS_VARIANTS[name]
    X, y, _ = design(tr, fcols)
    cols = cols_fn(X)
    med = X[cols].median().fillna(0.0)
    m = make_model(kind, params, n_jobs)
    m.fit(X[cols].fillna(med) if kind == "lr" else X[cols], y)
    return {"name": name, "kind": kind, "model": m, "cols": cols, "med": med}


def predict_stats_one(fm, df, fcols):
    def pr(Z):
        Z = Z[fm["cols"]]
        if fm["kind"] == "lr":
            Z = Z.fillna(fm["med"])
        return fm["model"].predict_proba(Z)[:, 1]
    Xo = side_frame(df, "r", "b", fcols, False)
    Xs = side_frame(df, "b", "r", fcols, True)
    return 0.5 * (pr(Xo) + 1 - pr(Xs))


def lg_ens_from(preds):
    return np.mean([logit(p) for p in preds], axis=0)


# ------------------------------------------------------------------ stacker
def prep_market(d):
    d = d.copy()
    d["lg_mkt"] = logit(d.mkt_close)
    d["lg_ult_f"] = logit(d.mkt_p_red).fillna(d.lg_mkt)
    d["lg_open_f"] = logit(d.mkt_open).fillna(d.lg_mkt)
    d["corner"] = np.where(d.date >= "2010-01-01", 1.0, 0.0)
    return d


def _sym_frame(d, flip):
    X = {}
    for c in ANTI:
        v = (d["r_" + c[2:]] - d["b_" + c[2:]]).values.astype(float) if c.startswith("D_") else d[c].values.astype(float)
        X[c] = -v if flip else v
    for c in SYM:
        if c == "year":
            X[c] = (d.date.dt.year + d.date.dt.dayofyear / 366).values
        elif c == "close_spread":
            X[c] = d["n_close_spread"].values
        else:
            X[c] = d[c].astype(float).values
    X["A_n_fights"] = (d.b_n_fights if flip else d.r_n_fights).values
    X["B_n_fights"] = (d.r_n_fights if flip else d.b_n_fights).values
    return pd.DataFrame(X)


def fit_stacker(tr):
    """tr: completed fights with columns lg_ens (out-of-fold!) + market columns."""
    tr = prep_market(tr)
    tr = tr[tr.lg_mkt.notna() & tr.lg_ens.notna()]
    y = np.concatenate([tr.y.values, 1 - tr.y.values])
    # (a) LR on [lg_mkt, lg_ens]
    cols = ["lg_mkt", "lg_ens"]
    Xt = tr[cols].values
    lr = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000).fit(np.vstack([Xt, -Xt]), y)
    # (b) residual LightGBM from the market logit
    X = pd.concat([_sym_frame(tr, False), _sym_frame(tr, True)], ignore_index=True)
    init = np.concatenate([tr.lg_mkt.values, -tr.lg_mkt.values])
    rl = lgb.LGBMClassifier(**{**dict(objective="binary", verbose=-1, n_jobs=8, subsample_freq=1), **RESID_PARAMS}).fit(X, y, init_score=init)
    return {"lr": lr, "lr_cols": cols, "resid": rl}


def predict_final(st, df):
    """Returns P(red wins). df needs lg_ens + market columns."""
    df = prep_market(df)
    Z = df[st["lr_cols"]].values
    ok = ~np.isnan(Z).any(axis=1)
    za = np.full(len(df), np.nan)
    za[ok] = st["lr"].decision_function(Z[ok])
    ra = st["resid"].predict_proba(_sym_frame(df, False), raw_score=True) + df.lg_mkt.values
    rb = st["resid"].predict_proba(_sym_frame(df, True), raw_score=True) - df.lg_mkt.values
    pr = 0.5 * (1 / (1 + np.exp(-ra)) + 1 - 1 / (1 + np.exp(-rb)))
    p = 1 / (1 + np.exp(-0.5 * (za + logit(pr))))
    stats_only = 1 / (1 + np.exp(-df.lg_ens.values))
    return np.where(np.isnan(p), stats_only, p)
