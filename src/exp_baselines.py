"""First experiments: market baseline, logistic regression, LightGBM. Tuning only on validation (2022-2023)."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import TRAIN_END, VALID_END, design, fighter_cols, load, metrics, sym_predict

warnings.filterwarnings("ignore")


def report(name, p, d, mask_odds=True):
    out = {"model": name}
    for split, m in (("valid", (d.date >= TRAIN_END) & (d.date < VALID_END)), ("test", d.date >= VALID_END)):
        r = metrics(p[m], d.y[m])
        out[f"{split}_acc"] = r["acc"]
        out[f"{split}_ll"] = r["logloss"]
        mo = m & d.mkt_close.notna()
        out[f"{split}_acc_odds"] = metrics(p[mo], d.y[mo])["acc"]
    return out


def main():
    d = load()
    fcols = fighter_cols(d)
    print("fighter feature cols", len(fcols))
    res = []
    mkt = d.mkt_close.fillna(0.5).values
    res.append(report("market_close", mkt, d))

    tr = d[(d.date < TRAIN_END) & (d.date >= "2001-01-01")]
    X, y, _ = design(tr, fcols)

    # LR on rating probs only (no odds)
    for cols, name in (
        (["lg_p_elo32", "lg_p_elo_mov", "lg_p_elo_str", "lg_p_glicko", "corner"], "LR ratings"),
        (["lg_mkt_close", "corner"], "LR market"),
    ):
        sub = X[cols]
        ok = sub.notna().all(axis=1)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(sub[ok], y[ok])
        fn = lambda Z, lr=lr, cols=cols: lr.predict_proba(Z[cols].fillna(0))[:, 1]
        res.append(report(name, sym_predict(fn, d, fcols), d))

    # LightGBM, all features, with and without market
    mkt_cols = [c for c in X.columns if "mkt" in c or c in ("line_move", "close_spread", "overround") or c.endswith("_imp")]
    for name, drop in (("LGB no-odds", mkt_cols), ("LGB all", [])):
        cols = [c for c in X.columns if c not in drop]
        params = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.5, reg_lambda=5.0, n_estimators=600, verbose=-1)
        m = lgb.LGBMClassifier(**params).fit(X[cols], y)
        fn = lambda Z, m=m, cols=cols: m.predict_proba(Z[cols])[:, 1]
        res.append(report(name, sym_predict(fn, d, fcols), d))
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print(name, "top features:", imp.head(15).round(0).to_dict())
    print(pd.DataFrame(res).round(4).to_string())


if __name__ == "__main__":
    main()
