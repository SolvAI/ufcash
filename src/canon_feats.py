"""Full pro-career features from fight-canon-db (Sherdog records of every pro MMA fight, all promotions).

For every ufcstats bout, each fighter's state is taken from fights on dates strictly before the bout date.
Also computes a world Elo across all ~370k pro fights (rates debutants by the quality of their regional opposition).
Output: data/processed/canon_feats.parquet keyed by ufcstats fight_id (columns r_pro_*, b_pro_*, p_welo*).
"""
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("UFC_CANON_DB", ROOT / "data" / "raw" / "canon" / "canon.db"))
PROC = Path(os.environ.get("UFC_PROC", ROOT / "data" / "processed"))  # overridable for the leakage audit

MAJOR = ("UFC", "Bellator", "PFL", "Professional Fighters League", "ONE", "Strikeforce", "PRIDE", "WEC", "DREAM",
         "Rizin", "RIZIN", "KSW", "Absolute Championship", "M-1", "CWFC", "Cage Warriors", "LFA", "Legacy Fighting",
         "Invicta", "Contender", "DWCS", "Oktagon", "OKTAGON", "Affliction", "EliteXC", "World Series of Fighting")


def mcat(m):
    m = str(m or "")
    if m.startswith("Draw"):
        return "draw"
    if "Decision" in m:
        return "dec"
    if "Submission" in m and "Submission to Punches" not in m:
        return "sub"
    if "KO" in m:
        return "ko"
    return "other"


def main():
    c = sqlite3.connect(DB)
    q = lambda s: pd.read_sql(s, c)
    fights = q("""select f.fight_id, e.date, e.promotion, e.location, f.method, f.no_contest, f.overturned
                  from fights f join events e using(event_id) where e.date is not null""")
    parts = q("select fight_id, fighter_id, corner, result from fight_participants")
    fa = q("select fight_id, source_id from fight_aliases where source='ufcstats'")
    fta = q("select fighter_id, source_id from fighter_aliases where source='ufcstats'")
    dob = q("select fighter_id, dob from fighters").set_index("fighter_id").dob
    names = q("select fighter_id, name from fighter_aliases where source='sherdog'")
    c.close()

    fights["date"] = pd.to_datetime(fights.date, errors="coerce")
    fights = fights.dropna(subset=["date"])
    fights["mc"] = fights.method.map(mcat)
    fights["major"] = fights.promotion.fillna("").map(lambda p: any(k in p for k in MAJOR))
    fights["is_ufc"] = fights.promotion.fillna("").str.startswith("UFC")
    fights["country"] = fights.location.fillna("").str.split(",").str[-1].str.strip().replace({"United States": "USA", "US": "USA", "England": "United Kingdom", "UK": "United Kingdom"})
    p0 = parts[parts.corner == 0].set_index("fight_id")
    p1 = parts[parts.corner == 1].set_index("fight_id")
    fights = fights.set_index("fight_id")
    fights = fights.join(p0[["fighter_id", "result"]].rename(columns={"fighter_id": "a", "result": "ra"}), how="inner")
    fights = fights.join(p1[["fighter_id", "result"]].rename(columns={"fighter_id": "b", "result": "rb"}), how="inner")
    fights = fights.reset_index().sort_values("date")
    # drop near-duplicate entries of the same pairing (<=3 days apart): keep the last, so no copy leaks forward
    fights["lo"] = fights[["a", "b"]].min(axis=1)
    fights["hi"] = fights[["a", "b"]].max(axis=1)
    nxt = fights.groupby(["lo", "hi"]).date.shift(-1)
    dup = (nxt - fights.date).dt.days <= 3
    fights = fights[~dup]
    print("canon fights", len(fights), "dropped dups", int(dup.sum()))

    # ---- query: ufcstats bouts -> canon fighter ids
    ours = pd.read_parquet(PROC / "fights.parquet")[["fight_id", "date", "r_id", "b_id", "location"]]
    ours["country"] = ours.location.fillna("").str.split(",").str[-1].str.strip()
    fid_map = dict(zip(fta.source_id, fta.fighter_id))
    ours["cr"] = ours.r_id.map(fid_map)
    ours["cb"] = ours.b_id.map(fid_map)
    # fallback via fight alias + corner
    alias = dict(zip(fa.source_id, fa.fight_id))
    ours["canon_fight"] = ours.fight_id.map(alias)
    pa = parts.set_index(["fight_id", "corner"]).fighter_id
    for col, corner in (("cr", 0), ("cb", 1)):
        miss = ours[col].isna() & ours.canon_fight.notna()
        ours.loc[miss, col] = [pa.get((int(f), corner), np.nan) for f in ours.loc[miss, "canon_fight"]]
    # fighters ufcstats doesn't know yet (e.g. UFC debutants in upcoming bouts): unique exact Sherdog name match
    nm = names.drop_duplicates().groupby("name").fighter_id.agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan).dropna()
    for col, nmcol in (("cr", "r_id"), ("cb", "b_id")):
        miss = ours[col].isna() & ours[nmcol].astype(str).str.startswith("name:")
        ours.loc[miss, col] = ours.loc[miss, nmcol].str[5:].map(nm)
    print("ufcstats bouts mapped:", ours.cr.notna().mean().round(4), ours.cb.notna().mean().round(4))
    q_by_date = {d: g for d, g in ours.groupby("date")}

    # ---- state
    S = defaultdict(lambda: {"n": 0, "w": 0, "l": 0, "d": 0, "nc": 0, "kow": 0, "subw": 0, "decw": 0, "kol": 0, "subl": 0,
                             "decl": 0, "major": 0, "ufc": 0, "res": [], "dates": [], "first": None, "opp_elo": [], "beat_elo": [],
                             "countries": defaultdict(int), "elo_hist": []})
    H2H = defaultdict(list)  # (me, opp) -> list of results from me's perspective
    elo = defaultdict(lambda: 1500.0)
    elo_n = defaultdict(int)
    elo_m = defaultdict(lambda: 1500.0)  # MOV-scored

    def K(f, base=32.0):
        n = elo_n[f]
        return base * (2.5 if n < 3 else 1.6 if n < 8 else 1.0)

    def feats(f, date, country=None, opp=None):
        s = S[f]
        n = s["n"]
        eh = s["elo_hist"]
        o = {"pro_n": n, "pro_w": s["w"], "pro_l": s["l"], "pro_wr": (s["w"] + 1) / (s["w"] + s["l"] + 2),
             "pro_kow_rate": (s["kow"] + 0.5) / (n + 1.5), "pro_subw_rate": (s["subw"] + 0.3) / (n + 1.5),
             "pro_finw_rate": (s["kow"] + s["subw"] + 0.8) / (n + 1.5),
             "pro_kol_rate": (s["kol"] + 0.2) / (n + 1.5), "pro_subl_rate": (s["subl"] + 0.2) / (n + 1.5),
             "pro_major": s["major"], "pro_nonufc": n - s["ufc"], "pro_major_nonufc": s["major"] - s["ufc"],
             "welo": elo[f], "welo_mov": elo_m[f], "welo_n": elo_n[f],
             "welo_mom3": elo[f] - eh[-3] if len(eh) >= 3 else 0.0,
             "welo_peak_gap": (max(eh) - elo[f]) if eh else 0.0,
             "home_frac": (s["countries"].get(country, 0) + 0.1) / (n + 1) if country else np.nan}
        h2h = H2H.get((f, opp), [])
        o["h2h_n"] = len(h2h)
        o["h2h_net"] = sum(1 if r == 1 else -1 if r == 0 else 0 for r in h2h if r is not None)
        res = s["res"]
        ws = ls = 0
        for r in reversed(res):
            if r == 1 and ls == 0:
                ws += 1
            elif r == 0 and ws == 0:
                ls += 1
            else:
                break
        o["pro_ws"], o["pro_ls"] = ws, ls
        last = [r for r in res[-5:] if r is not None]
        o["pro_last5_wr"] = (sum(1 for r in last if r == 1) + 0.5) / (len(last) + 1)
        ds = s["dates"]
        o["pro_days_since"] = (date - ds[-1]).days if ds else np.nan
        o["pro_career_days"] = (date - s["first"]).days if s["first"] is not None else 0
        o["pro_n_2y"] = sum(1 for x in ds if (date - x).days < 730)
        o["pro_n_1y"] = sum(1 for x in ds if (date - x).days < 365)
        oe = s["opp_elo"]
        o["pro_opp_elo"] = float(np.mean(oe)) if oe else np.nan
        o["pro_opp_elo5"] = float(np.mean(oe[-5:])) if oe else np.nan
        be = s["beat_elo"]
        o["pro_beat_elo_max"] = max(be) if be else np.nan
        o["pro_beat_elo_top3"] = float(np.mean(sorted(be)[-3:])) if be else np.nan
        db = dob.get(f)
        if db and s["first"] is not None:
            try:
                o["pro_age_debut"] = (s["first"] - pd.Timestamp(db)).days / 365.25
            except Exception:
                o["pro_age_debut"] = np.nan
        else:
            o["pro_age_debut"] = np.nan
        return o

    out = []

    def emit(date):
        for r in q_by_date[date].itertuples(index=False):
            rec = {"fight_id": r.fight_id}
            for side, f in (("r", r.cr), ("b", r.cb)):
                if pd.isna(f):
                    continue
                opp = r.cb if side == "r" else r.cr
                for k, v in feats(int(f), date, r.country, None if pd.isna(opp) else int(opp)).items():
                    rec[f"{side}_{k}"] = v
            if pd.notna(r.cr) and pd.notna(r.cb):
                rec["p_welo"] = 1 / (1 + 10 ** ((elo[int(r.cb)] - elo[int(r.cr)]) / 400))
                rec["p_welo_mov"] = 1 / (1 + 10 ** ((elo_m[int(r.cb)] - elo_m[int(r.cr)]) / 400))
            out.append(rec)

    done = set()
    for date, day in fights.groupby("date", sort=True):
        # query dates with no canon fight on that exact day (e.g. upcoming bouts) are evaluated before the next date
        for qd in sorted(x for x in q_by_date if x < date and x not in done):
            emit(qd)
            done.add(qd)
        if date in q_by_date:
            done.add(date)
            emit(date)
        # update with this date's fights
        de, dm, upd = defaultdict(float), defaultdict(float), []
        for r in day.itertuples(index=False):
            a, b = int(r.a), int(r.b)
            if r.no_contest or r.ra == "NC" or r.overturned:
                sa = None
            elif r.ra == "W":
                sa = 1.0
            elif r.ra == "L":
                sa = 0.0
            elif r.ra == "D":
                sa = 0.5
            else:
                sa = None
            if sa is not None:
                ea = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
                de[a] += K(a) * (sa - ea)
                de[b] += K(b) * ((1 - sa) - (1 - ea))
                mv = sa if sa == 0.5 else ({"ko": 1.0, "sub": 1.0, "dec": 0.8}.get(r.mc, 0.8) if sa == 1 else 1 - {"ko": 1.0, "sub": 1.0, "dec": 0.8}.get(r.mc, 0.8))
                em = 1 / (1 + 10 ** ((elo_m[b] - elo_m[a]) / 400))
                dm[a] += K(a, 40) * (mv - em)
                dm[b] += K(b, 40) * ((1 - mv) - (1 - em))
            upd.append((a, b, sa, r))
        for a, b, sa, r in upd:
            for me, op, s_ in ((a, b, sa), (b, a, None if sa is None else 1 - sa)):
                st = S[me]
                st["n"] += 1
                st["dates"].append(date)
                if st["first"] is None:
                    st["first"] = date
                st["major"] += int(r.major)
                st["ufc"] += int(r.is_ufc)
                st["opp_elo"].append(elo[op])
                st["countries"][r.country] += 1
                H2H[(me, op)].append(s_)
                if s_ is None:
                    st["nc"] += 1
                    st["res"].append(None)
                elif s_ == 0.5:
                    st["d"] += 1
                    st["res"].append(0.5)
                elif s_ == 1:
                    st["w"] += 1
                    st["res"].append(1)
                    st["beat_elo"].append(elo[op])
                    st[{"ko": "kow", "sub": "subw"}.get(r.mc, "decw")] += 1
                else:
                    st["l"] += 1
                    st["res"].append(0)
                    st[{"ko": "kol", "sub": "subl"}.get(r.mc, "decl")] += 1
        for f, v in de.items():
            S[f]["elo_hist"].append(elo[f])
            elo[f] += v
            elo_n[f] += 1
        for f, v in dm.items():
            elo_m[f] += v
    for qd in sorted(x for x in q_by_date if x not in done):
        emit(qd)
    res = pd.DataFrame(out)
    res.to_parquet(PROC / "canon_feats.parquet", index=False)
    print(res.shape, res.isna().mean().sort_values().tail(5).round(3).to_dict())


if __name__ == "__main__":
    main()
