"""Experiment 5: ensemble OOF stats models, stack with market (LR + residual-LGB). Walk-forward by year."""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import PROC, load, logit, metrics
from walk import DEV_YEARS, TEST_YEARS, per_year, summarize, walk_forward

warnings.filterwarnings("ignore")
STATS_MODELS = ["lgb_small", "lgb_d4_diff", "xgb_diff", "lr_diff", "cat_all"]


def add_oof(d, names):
    for n in names:
        o = pd.read_parquet(PROC / "oof" / f"{n}.parquet")
        d[n] = d.fight_id.map(dict(zip(o.fight_id, o[n])))
    d["lg_ens"] = np.mean([logit(d[n]) for n in names], axis=0)
    return d


def prep(d):
    d["lg_close"] = logit(d.n_close_r)
    d["lg_ult"] = logit(d.mkt_p_red)
    d["lg_mkt"] = logit(d.mkt_close)
    d["lg_ult_f"] = d.lg_ult.fillna(d.lg_mkt)
    d["lg_open_f"] = logit(d.mkt_open).fillna(d.lg_mkt)
    d["corner"] = np.where(d.date >= "2010-01-01", 1.0, 0.0)
    d["lg_mkt_avg"] = 0.5 * (d.lg_mkt + d.lg_ult_f)
    return d


# antisymmetric (flip sign under corner swap) and symmetric (unchanged) columns for the residual LGB
ANTI = ["lg_ens", "lg_mkt", "lg_ult_f", "lg_open_f", "corner", "D_n_fights", "D_pro_n", "D_age", "D_welo", "D_reach", "D_rank", "D_days_since",
        "D_c_adj_net", "D_e_sdiff", "D_elo_str", "D_card_pos_mean", "D_mk_surprise", "D_mk_mean_imp", "D_mk_last_imp", "D_mk_surprise3"]
SYM = ["women", "wc_limit", "sched_rounds", "title", "close_spread", "card_pos", "year"]


def sym_frame(d, flip):
    X = {}
    for c in ANTI:
        if c.startswith("D_"):
            base = c[2:]
            v = (d["r_" + base] - d["b_" + base]).values.astype(float)
        else:
            v = d[c].values.astype(float)
        X[c] = -v if flip else v
    for c in SYM:
        if c == "year":
            X[c] = (d.date.dt.year + d.date.dt.dayofyear / 366).values
        elif c == "close_spread":
            X[c] = d["n_close_spread"].values
        else:
            X[c] = d[c].astype(float).values
    # also absolute experience of the side
    X["A_n_fights"] = (d.b_n_fights if flip else d.r_n_fights).values
    X["B_n_fights"] = (d.r_n_fights if flip else d.b_n_fights).values
    return pd.DataFrame(X)


def lr_stack(cols, C=1.0, weight_hl=None):
    def fp(tr, te):
        tr = tr.dropna(subset=cols)
        Xt = tr[cols].values
        X = np.vstack([Xt, -Xt])
        y = np.concatenate([tr.y.values, 1 - tr.y.values])
        w = None
        if weight_hl:
            age = (tr.date.max() - tr.date).dt.days.values / 365.25
            w = np.tile(0.5 ** (age / weight_hl), 2)
        lr = LogisticRegression(C=C, fit_intercept=False, max_iter=5000).fit(X, y, sample_weight=w)
        Z = te[cols].values
        ok = ~np.isnan(Z).any(axis=1)
        out = np.full(len(te), np.nan)
        out[ok] = lr.predict_proba(Z[ok])[:, 1]
        return out
    return fp


def resid_lgb(params, base_col="lg_mkt"):
    def fp(tr, te):
        tr = tr[tr[base_col].notna() & tr.lg_ens.notna()]
        Xa, Xb = sym_frame(tr, False), sym_frame(tr, True)
        X = pd.concat([Xa, Xb], ignore_index=True)
        y = np.concatenate([tr.y.values, 1 - tr.y.values])
        init = np.concatenate([tr[base_col].values, -tr[base_col].values])
        m = lgb.LGBMClassifier(**{**dict(objective="binary", verbose=-1, n_jobs=8, subsample_freq=1), **params}).fit(X, y, init_score=init)
        def pr(Z, base):
            return 1 / (1 + np.exp(-(m.predict_proba(Z, raw_score=True) + base)))
        pa = pr(sym_frame(te, False), te[base_col].values)
        pb = pr(sym_frame(te, True), -te[base_col].values)
        p = 0.5 * (pa + 1 - pb)
        return np.where(te[base_col].isna() | te.lg_ens.isna(), np.nan, p)
    return fp


def main():
    d = prep(add_oof(load(), STATS_MODELS))
    test = "--test" in sys.argv
    years = DEV_YEARS + (TEST_YEARS if test else [])
    rp = dict(learning_rate=0.01, num_leaves=4, min_child_samples=300, subsample=0.7, colsample_bytree=0.5, reg_lambda=20.0, n_estimators=200)
    variants = {
        "mkt": lambda: pd.Series(d.mkt_close.values, index=d.index),
        "lr[mkt,ens]": lambda: walk_forward(d, lr_stack(["lg_mkt", "lg_ens"]), years, train_from="2010-01-01"),
        "lr[mkt,ult,ens,corner]": lambda: walk_forward(d, lr_stack(["lg_mkt", "lg_ult_f", "lg_ens", "corner"]), years, train_from="2010-01-01"),
        "lr[mktavg,ens,corner]": lambda: walk_forward(d, lr_stack(["lg_mkt_avg", "lg_ens", "corner"]), years, train_from="2010-01-01"),
        "lr[mktavg,ens,corner] w5": lambda: walk_forward(d, lr_stack(["lg_mkt_avg", "lg_ens", "corner"], weight_hl=5), years, train_from="2010-01-01"),
        "resid_lgb200": lambda: walk_forward(d, resid_lgb(rp), years, train_from="2010-01-01"),
        "resid_lgb400": lambda: walk_forward(d, resid_lgb(dict(rp, n_estimators=400)), years, train_from="2010-01-01"),
        "resid_lgb200_avg": lambda: walk_forward(d, resid_lgb(rp, "lg_mkt_avg"), years, train_from="2010-01-01"),
    }
    for n in STATS_MODELS + ["ens"]:
        col = n if n != "ens" else None
        p = pd.Series(d[col].values if col else 1 / (1 + np.exp(-d.lg_ens.values)), index=d.index)
        r = summarize(n, p, d, DEV_YEARS)
        print(f"stats {n:12s} dev acc {r['acc']:.4f} ll {r['ll']:.4f}" + (f" | test acc {summarize(n, p, d, TEST_YEARS)['acc']:.4f}" if test else ""))
    rows = []
    preds = {}
    for name, fn in variants.items():
        p = fn()
        preds[name] = p
        if name == "resid_lgb200":
            a, b = logit(preds["lr[mkt,ens]"]), logit(p)
            bl = pd.Series(1 / (1 + np.exp(-0.5 * (a + b))), index=d.index)
            # fights without odds: fall back to the stats ensemble
            bl = bl.fillna(pd.Series(1 / (1 + np.exp(-d.lg_ens.values)), index=d.index))
            preds["BLEND"] = bl
            rr = summarize("BLEND", bl, d, DEV_YEARS)
            if test:
                t = summarize("BLEND", bl, d, TEST_YEARS)
                rr.update({"T_acc": t["acc_odds"], "T_ll": t["ll_odds"], "T_mkt": t["mkt_acc"], "T_n": t["n_odds"], "T_acc_all": t["acc"], "T_n_all": t["n"]})
            rows.append(rr)
        r = summarize(name, p, d, DEV_YEARS)
        if test:
            t = summarize(name, p, d, TEST_YEARS)
            r.update({"T_acc": t["acc_odds"], "T_ll": t["ll_odds"], "T_mkt": t["mkt_acc"], "T_n": t["n_odds"]})
        rows.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).drop(columns=["acc", "ll", "n"]).round(4).to_string())
    if "--peryear" in sys.argv:
        for name in sys.argv[sys.argv.index("--peryear") + 1:]:
            print(name)
            print(per_year(preds[name], d, years).round(4).to_string())


if __name__ == "__main__":
    main()
