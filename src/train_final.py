"""Fit the production model on every completed fight and save it to models/final.joblib.

stats ensemble: 5 models fit on all completed fights since 2001
stacker:        fit on out-of-fold stats predictions (yearly walk-forward, 2010+) so its weights are honest
stacker_open:   the same with the opening line as the market input (the value-bet model of src/betting.py)
"""
import warnings

import joblib
import numpy as np
from joblib import Parallel, delayed

from betting import open_frame
from common import ROOT, fighter_cols, load, logit
from evaluate import OOF_YEARS, oof_year
from final_model import STATS_VARIANTS, fit_stacker, fit_stats_one

warnings.filterwarnings("ignore")


def main():
    d = load()
    fcols = fighter_cols(d)
    print(f"completed fights: {len(d)}  ({d.date.min().date()} .. {d.date.max().date()})")
    jobs = [(n, y) for n in STATS_VARIANTS for y in OOF_YEARS]
    out = Parallel(n_jobs=7)(delayed(oof_year)(d, fcols, n, y) for n, y in jobs)
    for n in STATS_VARIANTS:
        d[n] = np.nan
    for n, idx, p in out:
        d.loc[idx, n] = p
    d["lg_ens"] = np.mean([logit(d[n]) for n in STATS_VARIANTS], axis=0)
    stacker = fit_stacker(d[d.date >= "2010-01-01"])
    print("stacker LR weights [lg_mkt, lg_stats]:", stacker["lr"].coef_[0].round(3))
    # opening-line model (src/betting.py): the same stacker with the OPENING line as its only market input -> value bets at the open
    do = open_frame(d[d.date >= "2010-01-01"])
    stacker_open = fit_stacker(do[do.mkt_close.notna()])
    print("opening stacker LR weights [lg_open, lg_stats]:", stacker_open["lr"].coef_[0].round(3))
    tr = d[d.date >= "2001-01-01"]
    stats = Parallel(n_jobs=5)(delayed(fit_stats_one)(tr, fcols, n, 4) for n in STATS_VARIANTS)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"stats": stats, "stacker": stacker, "stacker_open": stacker_open, "fcols": fcols, "trained_through": str(d.date.max().date())},
                ROOT / "models" / "final.joblib")
    print("saved models/final.joblib")


if __name__ == "__main__":
    main()
