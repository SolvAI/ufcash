"""Walk-forward (expanding window, yearly retrain) evaluation helpers."""
import numpy as np
import pandas as pd

from common import design, fighter_cols, metrics, side_frame

DEV_YEARS = list(range(2016, 2024))    # model selection
TEST_YEARS = [2024, 2025, 2026]       # final hold-out (only looked at for chosen configs)


def walk_forward(d, fit_predict, years, train_from="2001-01-01", fcols=None):
    """fit_predict(train_df, test_df) -> P(red) for test_df. Returns Series of OOF preds indexed like d."""
    fcols = fcols or fighter_cols(d)
    out = pd.Series(np.nan, index=d.index)
    for y in years:
        tr = d[(d.date >= train_from) & (d.date < f"{y}-01-01")]
        te = d[(d.date >= f"{y}-01-01") & (d.date < f"{y + 1}-01-01")]
        if len(te) == 0:
            continue
        out.loc[te.index] = fit_predict(tr, te)
    return out


def summarize(name, p, d, years):
    m = d.date.dt.year.isin(years) & p.notna()
    r = metrics(p[m], d.y[m])
    mo = m & d.mkt_close.notna()
    ro = metrics(p[mo], d.y[mo])
    mk = metrics(d.mkt_close[mo], d.y[mo])
    return {"model": name, "acc": r["acc"], "ll": r["logloss"], "n": r["n"], "acc_odds": ro["acc"], "ll_odds": ro["logloss"],
            "mkt_acc": mk["acc"], "mkt_ll": mk["logloss"], "n_odds": ro["n"]}


def per_year(p, d, years):
    rows = []
    for y in years:
        m = (d.date.dt.year == y) & p.notna()
        mo = m & d.mkt_close.notna()
        rows.append({"year": y, "acc": metrics(p[m], d.y[m])["acc"], "acc_odds": metrics(p[mo], d.y[mo])["acc"],
                     "mkt_acc": metrics(d.mkt_close[mo], d.y[mo])["acc"], "n": int(m.sum())})
    return pd.DataFrame(rows)
