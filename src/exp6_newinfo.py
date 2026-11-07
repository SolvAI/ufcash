"""Experiment 6: does new information (canon_extra: pre-UFC stats, media scorecards, judge rounds) help?
Full pipeline (5-model stats ensemble OOF 2010-2023 -> stacker walk-forward 2016-2023), DEV ONLY.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import PROC, fighter_cols, load, logit, metrics
from final_model import STATS_VARIANTS, fit_stacker, fit_stats_one, predict_final, predict_stats_one

warnings.filterwarnings("ignore")
YEARS = list(range(2010, 2024))


def oof(d, fcols, name, year):
    tr = d[(d.date >= "2001-01-01") & (d.date < f"{year}-01-01")]
    te = d[d.date.dt.year == year]
    return name, te.index.values, predict_stats_one(fit_stats_one(tr, fcols, name, n_jobs=4), te, fcols)


def run(d, fcols, tag):
    out = Parallel(n_jobs=7)(delayed(oof)(d, fcols, n, y) for n in STATS_VARIANTS for y in YEARS)
    d = d.copy()
    for n in STATS_VARIANTS:
        d[n] = np.nan
    for n, idx, p in out:
        d.loc[idx, n] = p
    d["lg_ens"] = np.mean([logit(d[n]) for n in STATS_VARIANTS], axis=0)
    d["p_final"] = np.nan
    for y in range(2016, 2024):
        st = fit_stacker(d[(d.date >= "2010-01-01") & (d.date < f"{y}-01-01")])
        te = d[d.date.dt.year == y]
        d.loc[te.index, "p_final"] = predict_final(st, te)
    g = d[(d.date >= "2016-01-01") & (d.date < "2024-01-01")]
    ms, mf = metrics(1 / (1 + np.exp(-g.lg_ens)), g.y), metrics(g.p_final, g.y)
    print(f"{tag:10s} stats acc {ms['acc']:.4f} ll {ms['logloss']:.4f} | final acc {mf['acc']:.4f} ll {mf['logloss']:.4f}", flush=True)
    d[["fight_id", "lg_ens", "p_final"]].to_parquet(PROC.parent / f"exp6_{tag}.parquet")
    return d


if __name__ == "__main__":
    d = load().merge(pd.read_parquet(PROC / "canon_extra.parquet"), on="fight_id", how="left")
    d = d[d.date < "2024-01-01"].reset_index(drop=True)  # dev only: the hold-out is not even loaded
    allc = fighter_cols(d)
    new = [c for c in allc if c.startswith(("xs_", "med_", "jr_"))]
    base = [c for c in allc if c not in new]
    cfg = {"base": base, "all_new": allc, "xs": base + [c for c in new if c.startswith("xs_")],
           "med_jr": base + [c for c in new if not c.startswith("xs_")]}
    for tag in sys.argv[1:] or cfg:
        run(d, cfg[tag], tag)
