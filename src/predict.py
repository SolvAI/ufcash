"""Predict the bouts listed in data/upcoming.csv with models/final.joblib: picks, and value bets at the price you entered.

Run the feature pipeline first (./ufcash card does it all), then:  python src/predict.py

Value bets use the opening-line model (src/betting.py: +24 % on 226 hold-out bets 2024-26 at BestFightOdds opening prices).
The odds in data/upcoming.csv are read as the price you can bet NOW, early in the week (the opening line). Rule, fixed on 2016-23:
bet the side whose edge = p * decimal odds - 1 is above 10 %; skip edges above 30 % (almost always a bad line) and odds pairs whose
overround is outside 0-20 %. Suggested stake: quarter Kelly, capped at 5 % of the bankroll.
"""
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from betting import open_frame
from common import PROC, ROOT
from final_model import lg_ens_from, predict_final, predict_stats_one

warnings.filterwarnings("ignore")
EDGE_MIN, EDGE_MAX, KELLY, CAP = 0.10, 0.30, 0.25, 0.05


def dec(ml):
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml > 0, 1 + ml / 100, np.where(ml < 0, 1 + 100 / -ml, np.nan))


def value_bets(up, p_open, red_odds, blue_odds):
    dr, db = dec(red_odds), dec(blue_odds)
    over = 1 / dr + 1 / db
    ev_r, ev_b = p_open * dr - 1, (1 - p_open) * db - 1
    side_red = ev_r >= ev_b
    edge = np.where(side_red, ev_r, ev_b)
    odds = np.where(side_red, dr, db)
    ok = (edge > EDGE_MIN) & (edge <= EDGE_MAX) & (over >= 1.0) & (over <= 1.2)
    kelly = np.clip(KELLY * edge / (odds - 1), 0, CAP)
    return pd.DataFrame({"bet": np.where(ok, np.where(side_red, up.r_name, up.b_name), ""), "decimal_odds": np.round(odds, 2),
                         "edge": np.round(edge, 3), "stake_pct_bankroll": np.where(ok, np.round(kelly * 100, 1), 0.0)})


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
    odds = pd.read_parquet(PROC / "upcoming_odds.parquet").set_index("fight_id").reindex(up.fight_id)
    out = pd.DataFrame({
        "date": up.date.dt.date, "red": up.r_name, "blue": up.b_name, "weight_class": up.weight_class,
        "market_p_red": up.mkt_close.round(3), "stats_p_red": up.p_stats.round(3), "model_p_red": up.p_final.round(3),
        "pick": np.where(up.p_final > 0.5, up.r_name, up.b_name),
        "confidence": np.maximum(up.p_final, 1 - up.p_final).round(3),
        "red_ufc_fights": up.r_n_fights, "blue_ufc_fights": up.b_n_fights,
    })
    if "stacker_open" in bundle:
        uo = up.copy()
        uo["mkt_open"] = uo.mkt_close  # the price entered = the price you can bet now, read as the opening line
        p_open = np.where(uo.mkt_open.notna(), predict_final(bundle["stacker_open"], open_frame(uo)), np.nan)
        out["open_model_p_red"] = np.round(p_open, 3)
        out = pd.concat([out, value_bets(up, p_open, odds.red_odds.values, odds.blue_odds.values)], axis=1)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "upcoming_predictions.csv", index=False)
    show(out, bundle["trained_through"])


def show(out, trained):
    color = sys.stdout.isatty()
    g = (lambda t: f"\033[32;1m{t}\033[0m") if color else str
    b = (lambda t: f"\033[1m{t}\033[0m") if color else str
    print(b(f"UFCash card  ") + f"(model trained on every UFC fight up to {trained})\n")
    for r in out.itertuples():
        mk = f"market {r.market_p_red * 100:4.1f}%" if r.market_p_red == r.market_p_red else "no odds   "
        print(f"  {r.date}  {b(f'{r.red} vs {r.blue}')}  ({r.weight_class})")
        print(f"      pick {b(r.pick)} {r.confidence * 100:.0f}%   red win: model {r.model_p_red * 100:4.1f}%, stats-only {r.stats_p_red * 100:4.1f}%, {mk}")
        if getattr(r, "bet", ""):
            print(f"      {g('VALUE BET')}  {b(r.bet)} at {r.decimal_odds:.2f}  edge {g(f'+{r.edge * 100:.0f}%')}  "
                  f"stake {r.stake_pct_bankroll:.1f}% of bankroll (quarter Kelly)")
    n = int((out.get("bet", pd.Series(dtype=str)) != "").sum()) if "bet" in out else 0
    print(f"\n{n} value bet{'s' if n != 1 else ''}. Saved to results/upcoming_predictions.csv")
    if n == 0 and out.market_p_red.isna().all():
        print("Add the moneylines (red_odds, blue_odds) to data/upcoming.csv: value bets need the price you can get.")


if __name__ == "__main__":
    main()
