"""Predict the bouts listed in data/upcoming.csv with models/final.joblib.

Run the feature pipeline first (./run_pipeline.sh does it all), then:  python src/predict.py
"""
import warnings

import joblib
import numpy as np
import pandas as pd

from common import PROC, ROOT
from final_model import lg_ens_from, predict_final, predict_stats_one

warnings.filterwarnings("ignore")


def main():
    bundle = joblib.load(ROOT / "models" / "final.joblib")
    d = pd.read_parquet(PROC / "features.parquet")
    up = d[d.outcome == "UPCOMING"].copy().reset_index(drop=True)
    if up.empty:
        print("no upcoming bouts (add rows to data/upcoming.csv and rerun the pipeline)")
        return
    preds = [predict_stats_one(fm, up, bundle["fcols"]) for fm in bundle["stats"]]
    up["lg_ens"] = lg_ens_from(preds)
    up["p_stats"] = 1 / (1 + np.exp(-up.lg_ens))
    up["p_final"] = predict_final(bundle["stacker"], up)
    out = pd.DataFrame({
        "date": up.date.dt.date, "red": up.r_name, "blue": up.b_name, "weight_class": up.weight_class,
        "market_p_red": up.mkt_close.round(3), "stats_p_red": up.p_stats.round(3), "model_p_red": up.p_final.round(3),
        "pick": np.where(up.p_final > 0.5, up.r_name, up.b_name),
        "confidence": np.maximum(up.p_final, 1 - up.p_final).round(3),
        "red_ufc_fights": up.r_n_fights, "blue_ufc_fights": up.b_n_fights,
    })
    pd.set_option("display.width", 250)
    print(f"model trained through {bundle['trained_through']}")
    print(out.to_string(index=False))
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "upcoming_predictions.csv", index=False)


if __name__ == "__main__":
    main()
