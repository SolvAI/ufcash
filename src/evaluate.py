"""Reproduce the headline evaluation: yearly walk-forward (retrain every Jan 1 on all prior fights).

  stats ensemble OOF for 2010-2026  ->  stacker walk-forward for 2016-2026
  dev  = 2016-2023 (all model selection happened here)
  test = 2024-01-01 .. latest (hold-out)

Writes results/walkforward_predictions.csv and results/metrics.json
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import PROC, ROOT, fighter_cols, load, logit, metrics
from final_model import STATS_VARIANTS, fit_stacker, fit_stats_one, predict_final, predict_stats_one

warnings.filterwarnings("ignore")
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
OOF_YEARS = list(range(2010, 2027))
EVAL_YEARS = list(range(2016, 2027))
DEV = list(range(2016, 2024))
TEST = [2024, 2025, 2026]


def oof_year(d, fcols, name, year):
    tr = d[(d.date >= "2001-01-01") & (d.date < f"{year}-01-01")]
    te = d[(d.date >= f"{year}-01-01") & (d.date < f"{year + 1}-01-01")]
    fm = fit_stats_one(tr, fcols, name, n_jobs=4)
    return name, te.index.values, predict_stats_one(fm, te, fcols)


def main():
    d = load()
    fcols = fighter_cols(d)
    jobs = [(n, y) for n in STATS_VARIANTS for y in OOF_YEARS]
    print(f"fitting {len(jobs)} stats models (5 variants x {len(OOF_YEARS)} years)...", flush=True)
    out = Parallel(n_jobs=7)(delayed(oof_year)(d, fcols, n, y) for n, y in jobs)
    for n in STATS_VARIANTS:
        d[n] = np.nan
    for n, idx, p in out:
        d.loc[idx, n] = p
    d["lg_ens"] = np.mean([logit(d[n]) for n in STATS_VARIANTS], axis=0)
    d["p_stats"] = 1 / (1 + np.exp(-d.lg_ens))
    # out-of-fold stats-ensemble logits (each year predicted by models trained on earlier years) for src/betting.py
    d.loc[d.lg_ens.notna(), ["fight_id", "date", "lg_ens"] + list(STATS_VARIANTS)].to_parquet(PROC / "wf_stats_oof.parquet", index=False)

    d["p_final"] = np.nan
    for y in EVAL_YEARS:
        tr = d[(d.date >= "2010-01-01") & (d.date < f"{y}-01-01")]
        te = d[(d.date.dt.year == y)]
        st = fit_stacker(tr)
        d.loc[te.index, "p_final"] = predict_final(st, te)

    ev = d[d.date.dt.year.isin(EVAL_YEARS)].copy()
    ev["p_red"] = np.where(ev.date >= "2010-01-01", 0.5 + 1e-9, np.nan)  # "always pick red corner"
    ev["p_mkt"] = ev.mkt_close

    def block(years, only_odds=False):
        m = ev.date.dt.year.isin(years)
        if only_odds:
            m &= ev.mkt_close.notna()
        g = ev[m]
        r = {"n": int(len(g))}
        for k, c in (("always_red", "p_red"), ("world_elo", "p_welo"), ("stats_only", "p_stats"), ("market_close", "p_mkt"), ("final", "p_final")):
            ok = g[c].notna()
            mm = metrics(g[c][ok], g.y[ok])
            r[k] = {"acc": round(mm["acc"], 4), "logloss": round(mm["logloss"], 4) if k != "always_red" else None, "n": mm["n"]}
        return r

    res = {"dev_2016_2023": block(DEV), "test_2024_2026_all": block(TEST), "test_2024_2026_with_odds": block(TEST, True),
           "test_period": [str(ev[ev.date.dt.year.isin(TEST)].date.min().date()), str(ev.date.max().date())]}
    per = []
    for y in EVAL_YEARS:
        g = ev[ev.date.dt.year == y]
        per.append({"year": y, "n": len(g), "final": metrics(g.p_final, g.y)["acc"],
                    "market": metrics(g.p_mkt.dropna(), g.y[g.p_mkt.notna()])["acc"], "stats_only": metrics(g.p_stats, g.y)["acc"]})
    res["per_year"] = per
    # calibration on test
    t = ev[ev.date.dt.year.isin(TEST)]
    bins = pd.cut(np.maximum(t.p_final, 1 - t.p_final), [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0])
    pick_right = np.where(t.p_final > 0.5, t.y == 1, t.y == 0)
    cal = pd.DataFrame({"bin": bins, "conf": np.maximum(t.p_final, 1 - t.p_final), "right": pick_right}).groupby("bin", observed=True).agg(
        n=("right", "size"), mean_conf=("conf", "mean"), hit_rate=("right", "mean")).round(3)
    res["calibration_test"] = cal.reset_index().astype({"bin": str}).to_dict(orient="records")

    cols = ["fight_id", "date", "event", "r_name", "b_name", "weight_class", "y", "mkt_close", "p_stats", "p_final"]
    ev[cols].rename(columns={"y": "red_won", "mkt_close": "market_p_red", "p_stats": "stats_p_red", "p_final": "final_p_red"}).to_csv(
        RES / "walkforward_predictions.csv", index=False)
    json.dump(res, open(RES / "metrics.json", "w"), indent=2)

    def show(title, r):
        print(f"\n{title} (n={r['n']})")
        for k in ("always_red", "world_elo", "stats_only", "market_close", "final"):
            v = r[k]
            print(f"  {k:13s} acc {v['acc']:.4f}" + (f"   logloss {v['logloss']:.4f}" if v["logloss"] is not None else "") + f"   (n={v['n']})")
    show("DEV 2016-2023", res["dev_2016_2023"])
    show("TEST 2024-2026, all fights", res["test_2024_2026_all"])
    show("TEST 2024-2026, fights with odds", res["test_2024_2026_with_odds"])
    print("\nper year:\n", pd.DataFrame(per).round(4).to_string(index=False))
    print("\ncalibration (test):\n", cal.to_string())


if __name__ == "__main__":
    main()
