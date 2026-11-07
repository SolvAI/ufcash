"""Attach betting odds / rankings (ultimate_ufc_dataset) to ufcstats fight ids.

Output: data/processed/odds.parquet with one row per fight_id, oriented to the ufcstats red corner.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def american_to_prob(o):
    o = pd.to_numeric(o, errors="coerce")
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def load_ultimate():
    u = pd.read_csv(ROOT / "data" / "raw" / "ultimate" / "ufc-master.csv")
    u["date"] = pd.to_datetime(u.date)
    out = pd.DataFrame({"date": u.date, "R_fighter": u.R_fighter, "B_fighter": u.B_fighter})
    pr, pb = american_to_prob(u.R_odds), american_to_prob(u.B_odds)
    out["u_r_imp"], out["u_b_imp"] = pr, pb
    for k in ("dec", "sub", "ko"):
        out[f"u_r_{k}_imp"] = american_to_prob(u[f"r_{k}_odds"])
        out[f"u_b_{k}_imp"] = american_to_prob(u[f"b_{k}_odds"])
    out["u_r_rank"] = u.R_match_weightclass_rank
    out["u_b_rank"] = u.B_match_weightclass_rank
    out["u_r_p4p"] = u["R_Pound-for-Pound_rank"]
    out["u_b_p4p"] = u["B_Pound-for-Pound_rank"]
    return out


def match_to_fights(src, fights, r_col="R_fighter", b_col="B_fighter", day_tol=1):
    """Fuzzy-join a source table (date, red name, blue name) to ufcstats fights; returns rows with fight_id + swapped flag."""
    rows = []
    f_by_date = {d: g for d, g in fights.groupby("date")}
    for i, r in src.iterrows():
        cands = [f_by_date[d] for d in (r.date + pd.Timedelta(days=k) for k in range(-day_tol, day_tol + 1)) if d in f_by_date]
        if not cands:
            continue
        c = pd.concat(cands)
        same = np.array([fuzz.token_sort_ratio(r[r_col], a) + fuzz.token_sort_ratio(r[b_col], b) for a, b in zip(c.r_name, c.b_name)])
        swap = np.array([fuzz.token_sort_ratio(r[r_col], b) + fuzz.token_sort_ratio(r[b_col], a) for a, b in zip(c.r_name, c.b_name)])
        best = np.maximum(same, swap)
        j = best.argmax()
        if best[j] < 150:
            continue
        rows.append({"src_idx": i, "fight_id": c.fight_id.iloc[j], "swapped": bool(swap[j] > same[j]), "score": best[j]})
    m = pd.DataFrame(rows).sort_values("score", ascending=False).drop_duplicates("fight_id")
    return m


def main():
    fights = pd.read_parquet(PROC / "fights.parquet")
    u = load_ultimate()
    m = match_to_fights(u, fights)
    j = m.join(u, on="src_idx")
    out = pd.DataFrame({"fight_id": j.fight_id})
    sw = j.swapped.values
    for a, b in [("u_r_imp", "u_b_imp"), ("u_r_dec_imp", "u_b_dec_imp"), ("u_r_sub_imp", "u_b_sub_imp"),
                 ("u_r_ko_imp", "u_b_ko_imp"), ("u_r_rank", "u_b_rank"), ("u_r_p4p", "u_b_p4p")]:
        ra, rb = j[a].values, j[b].values
        out[a.replace("u_r_", "r_")] = np.where(sw, rb, ra)
        out[b.replace("u_b_", "b_")] = np.where(sw, ra, rb)
    s = out.r_imp + out.b_imp
    out["mkt_p_red"] = out.r_imp / s  # vig removed
    out["overround"] = s
    print("ultimate matched", len(out), "of", len(u), "| with odds", out.mkt_p_red.notna().sum(), "| swapped", sw.sum())

    # ---- nlbair/ufc-data (BestFightOdds opening + closing range, official rankings), keyed by ufcstats ids
    w = pd.read_parquet(ROOT / "data" / "raw" / "nlbair" / "ufc_all_fights_wide.parquet")
    w = w.merge(fights[["fight_id", "r_id", "b_id", "r_name", "b_name"]], on="fight_id", how="inner")
    # orientation: f1 == ufcstats red in >99.5%; detect swaps by id, then by name
    same = (w.f1_id == w.r_id) | ((w.f1_id != w.b_id) & (w.f1_name == w.r_name))
    n = pd.DataFrame({"fight_id": w.fight_id})
    def orient(c1, c2, prob=False):
        a, b = w[c1].values, w[c2].values
        return np.where(same, a, b), np.where(same, b, a)
    n["n_open_r"], n["n_open_b"] = orient("f1_open_prob_novig", "f2_open_prob_novig")
    n["n_close_r"], n["n_close_b"] = orient("f1_close_prob_novig", "f2_close_prob_novig")
    for k in ("open", "close_low", "close_high"):
        n[f"n_r_{k}_ml"], n[f"n_b_{k}_ml"] = orient(f"f1_{k}", f"f2_{k}")
    n["n_r_rank"], n["n_b_rank"] = orient("f1_rank", "f2_rank")
    n["n_r_ranked"], n["n_b_ranked"] = orient("f1_is_ranked", "f2_is_ranked")
    n["n_r_p4p"], n["n_b_p4p"] = orient("f1_p4p_rank", "f2_p4p_rank")
    n["n_close_overround"] = american_to_prob((n.n_r_close_low_ml + n.n_r_close_high_ml) / 2) + american_to_prob((n.n_b_close_low_ml + n.n_b_close_high_ml) / 2)
    # closing range width in implied-prob space (book disagreement)
    n["n_close_spread"] = np.abs(american_to_prob(n.n_r_close_low_ml) - american_to_prob(n.n_r_close_high_ml))
    n = n.drop(columns=[c for c in n.columns if c.endswith("_ml")])
    n["n_r_ranked"] = n.n_r_ranked.astype(float)
    n["n_b_ranked"] = n.n_b_ranked.astype(float)
    print("nlbair joined", len(n), "| with close", n.n_close_r.notna().sum(), "| swapped", (~same).sum())

    out = out.merge(n, on="fight_id", how="outer")

    # ---- fight-canon-db (BestFightOdds open + closing range; updated daily) as a third source / gap filler
    import sqlite3
    c = sqlite3.connect(ROOT / "data" / "raw" / "canon" / "canon.db")
    co = pd.read_sql("""select fa.source_id as fight_id, xa.source_id as fighter_uid, o.open, o.close_low, o.close_high
                        from odds o join fight_aliases fa on fa.fight_id=o.fight_id and fa.source='ufcstats'
                        join fighter_aliases xa on xa.fighter_id=o.fighter_id and xa.source='ufcstats'
                        where o.book='all'""", c)
    c.close()
    co = co.drop_duplicates(["fight_id", "fighter_uid"])
    co = co.merge(fights[["fight_id", "r_id", "b_id"]], on="fight_id")
    def mid(lo, hi):
        return (american_to_prob(lo) + american_to_prob(hi)) / 2
    co["pc"] = mid(co.close_low, co.close_high)
    co["po"] = american_to_prob(co.open)
    r = co[co.fighter_uid == co.r_id].set_index("fight_id")
    b = co[co.fighter_uid == co.b_id].set_index("fight_id")
    cc = pd.DataFrame({"c_close_r": r.pc / (r.pc + b.reindex(r.index).pc), "c_open_r": r.po / (r.po + b.reindex(r.index).po)}).reset_index()
    print("canon odds fights", cc.c_close_r.notna().sum())
    out = out.merge(cc, on="fight_id", how="outer")
    out["odds_source"] = np.where(out.n_close_r.notna(), "nlbair", np.where(out.c_close_r.notna(), "canon", np.where(out.mkt_p_red.notna(), "ultimate", "none")))
    # best available closing-ish market probability (red)
    upo = PROC / "upcoming_odds.parquet"
    if upo.exists():
        uo = pd.read_parquet(upo)
        if len(uo):
            pr_, pb_ = american_to_prob(uo.red_odds), american_to_prob(uo.blue_odds)
            uo["u_close_r"] = pr_ / (pr_ + pb_)
            out = out.merge(uo[["fight_id", "u_close_r"]], on="fight_id", how="outer")
    if "u_close_r" not in out:
        out["u_close_r"] = np.nan
    out["mkt_close"] = out.n_close_r.fillna(out.c_close_r).fillna(out.mkt_p_red).fillna(out.u_close_r)
    out["mkt_open"] = out.n_open_r.fillna(out.c_open_r)
    out["line_move"] = out.mkt_close - out.mkt_open
    # rank: 0 champ .. 15, 16 = unranked (board known), NaN = no ranking info
    for c in ("r", "b"):
        out[f"rank_{c}"] = np.where(out[f"n_{c}_ranked"] == 1, out[f"n_{c}_rank"], np.where(out[f"n_{c}_ranked"] == 0, 16.0, np.nan))
        out[f"p4p_{c}"] = np.where(out[f"n_{c}_p4p"].notna(), out[f"n_{c}_p4p"], np.where(out[f"n_{c}_ranked"].notna(), 16.0, np.nan))
    out.to_parquet(PROC / "odds.parquet", index=False)
    print(out.odds_source.value_counts().to_dict())


if __name__ == "__main__":
    main()
