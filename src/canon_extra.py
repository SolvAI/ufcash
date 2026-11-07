"""Extra pre-fight features from fight-canon-db that the main builders do not use:

  xs_*   striking/grappling rates from NON-UFC pro fights with round stats (DWCS, PRIDE, WEC, Strikeforce, PFL, ACA,
         Cage Warriors, Oktagon, KSW, ONE, Rizin, ...): a stats picture of UFC debutants before their first UFC fight
  med_*  media scorecards (MMA Decisions): how the press scored the fighter's past decisions -> "deserved" record,
         decision luck (official result minus media verdict)
  jr_*   judges' round share in past scored fights

Every value for a bout on date D uses only fights dated strictly before D (as-of join, exact dates excluded).
Output: data/processed/canon_extra.parquet keyed by ufcstats fight_id (columns r_xs_*, b_med_*, ...).
"""
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("UFC_CANON_DB", ROOT / "data" / "raw" / "canon" / "canon.db"))
PROC = Path(os.environ.get("UFC_PROC", ROOT / "data" / "processed"))
SUM = ["sl", "sa", "ol", "oa", "tdl", "tda", "otdl", "otda", "ctrl", "octrl", "kd", "okd", "sub", "osub", "head_abs", "min",
       "xs_n", "med_n", "med_share", "med_res", "med_robbed", "med_gifted", "jr_won", "jr_n"]


def mmss(t):
    try:
        m, s = str(t).split(":")
        return int(m) + int(s) / 60
    except Exception:
        return np.nan


def main():
    c = sqlite3.connect(DB)
    q = lambda s: pd.read_sql(s, c)
    fights = q("""select f.fight_id, e.date, coalesce(e.promotion, '') as promotion, f.end_round, f.end_time
                  from fights f join events e using(event_id) where e.date is not null""")
    parts = q("select fight_id, fighter_id, result from fight_participants")
    rs = q("""select fight_id, fighter_id, sum(sig_str_land) sl, sum(sig_str_att) sa, sum(td_land) tdl, sum(td_att) tda,
                     sum(ctrl_sec) ctrl, sum(knockdowns) kd, sum(sub_att) sub, sum(head_land) head, count(distinct round) nr
              from round_stats group by fight_id, fighter_id""")
    med = q("select fight_id, fighter_id, writer, score from media_scores where score is not null")
    js = q("select fight_id, fighter_id, round, judge, score from judge_scores where score is not null")
    fta = q("select fighter_id, source_id from fighter_aliases where source='ufcstats'")
    fa = q("select fight_id, source_id from fight_aliases where source='ufcstats'")
    pc = q("select fight_id, fighter_id, corner from fight_participants")
    c.close()

    fights["date"] = pd.to_datetime(fights.date, errors="coerce")
    fights = fights.dropna(subset=["date"]).set_index("fight_id")
    fights["ufc"] = fights.promotion.str.startswith("UFC")

    # ---- non-UFC round stats, joined with the opponent's line
    rs = rs.merge(rs, on="fight_id", suffixes=("", "_o"))
    rs = rs[rs.fighter_id != rs.fighter_id_o]
    rs = rs.join(fights[["date", "ufc", "end_round", "end_time"]], on="fight_id")
    rs = rs[~rs.ufc.fillna(True).astype(bool)]
    mins = (pd.to_numeric(rs.end_round, errors="coerce") - 1) * 5 + rs.end_time.map(mmss)
    rs["min"] = mins.where(mins > 0, rs.nr * 5.0).clip(0.5, 60)
    x = pd.DataFrame({"fighter_id": rs.fighter_id, "date": rs.date, "sl": rs.sl, "sa": rs.sa, "ol": rs.sl_o, "oa": rs.sa_o,
                      "tdl": rs.tdl, "tda": rs.tda, "otdl": rs.tdl_o, "otda": rs.tda_o, "ctrl": rs.ctrl, "octrl": rs.ctrl_o,
                      "kd": rs.kd, "okd": rs.kd_o, "sub": rs["sub"], "osub": rs.sub_o, "head_abs": rs.head_o, "min": rs["min"], "xs_n": 1.0})

    # ---- media scorecards: per writer, who did they have winning
    m2 = med.merge(med, on=["fight_id", "writer"], suffixes=("", "_o"))
    m2 = m2[m2.fighter_id != m2.fighter_id_o]
    m2["vote"] = np.sign(m2.score - m2.score_o) * 0.5 + 0.5
    mv = m2.groupby(["fight_id", "fighter_id"]).vote.mean().rename("share").reset_index()
    mv = mv.merge(parts, on=["fight_id", "fighter_id"]).join(fights[["date"]], on="fight_id")
    res = mv.result.map({"W": 1.0, "L": 0.0, "D": 0.5})
    mv = mv[res.notna()].assign(res=res[res.notna()])
    y = pd.DataFrame({"fighter_id": mv.fighter_id, "date": mv.date, "med_n": 1.0, "med_share": mv.share, "med_res": mv.res,
                      "med_robbed": ((mv.res == 0) & (mv.share > 0.5)).astype(float),
                      "med_gifted": ((mv.res == 1) & (mv.share < 0.5)).astype(float)})

    # ---- judges: share of judge-rounds won
    j2 = js.merge(js, on=["fight_id", "round", "judge"], suffixes=("", "_o"))
    j2 = j2[j2.fighter_id != j2.fighter_id_o]
    j2["won"] = np.sign(j2.score - j2.score_o) * 0.5 + 0.5
    jr = j2.groupby(["fight_id", "fighter_id"]).won.agg(["sum", "size"]).reset_index().join(fights[["date"]], on="fight_id")
    z = pd.DataFrame({"fighter_id": jr.fighter_id, "date": jr.date, "jr_won": jr["sum"], "jr_n": jr["size"]})

    ev = pd.concat([x, y, z], ignore_index=True).dropna(subset=["date"])
    for col in SUM:
        if col not in ev:
            ev[col] = 0.0
    ev[SUM] = ev[SUM].fillna(0.0)
    cum = ev.groupby(["fighter_id", "date"])[SUM].sum().groupby(level=0).cumsum().reset_index().sort_values("date")

    # ---- ufcstats bouts -> canon fighter ids (alias table, then fight alias + corner)
    ours = pd.read_parquet(PROC / "fights.parquet")[["fight_id", "date", "r_id", "b_id"]]
    fid_map = dict(zip(fta.source_id, fta.fighter_id))
    ours["cr"] = ours.r_id.map(fid_map)
    ours["cb"] = ours.b_id.map(fid_map)
    alias = dict(zip(fa.source_id, fa.fight_id))
    pa = pc.set_index(["fight_id", "corner"]).fighter_id
    cf = ours.fight_id.map(alias)
    for col, corner in (("cr", 0), ("cb", 1)):
        miss = ours[col].isna() & cf.notna()
        ours.loc[miss, col] = [pa.get((int(f), corner), np.nan) for f in cf[miss]]

    # priors: pooled non-UFC rates from fights before 2014 (fixed constants, no future information)
    e0 = x[x.date < "2014-01-01"]
    M = e0["min"].sum()
    pri = {"slpm": e0.sl.sum() / M, "sacc": e0.sl.sum() / e0.sa.sum(), "td15": 15 * e0.tdl.sum() / M,
           "tdacc": e0.tdl.sum() / max(e0.tda.sum(), 1), "ctrl": e0.ctrl.sum() / (60 * M), "kd15": 15 * e0.kd.sum() / M,
           "sub15": 15 * e0["sub"].sum() / M, "head_pm": e0.head_abs.sum() / M}
    K = 15.0  # prior minutes

    out = ours[["fight_id"]].copy()
    for side, col in (("r", "cr"), ("b", "cb")):
        qq = ours[["fight_id", "date", col]].dropna().rename(columns={col: "fighter_id"}).sort_values("date")
        qq["fighter_id"] = qq.fighter_id.astype(int)
        cum["fighter_id"] = cum.fighter_id.astype(int)
        s = pd.merge_asof(qq, cum, on="date", by="fighter_id", allow_exact_matches=False, direction="backward")
        s[SUM] = s[SUM].fillna(0.0)
        m = s["min"]
        f = pd.DataFrame({"fight_id": s.fight_id})
        f["xs_n"] = s.xs_n
        f["xs_min"] = m
        f["xs_slpm"] = (s.sl + K * pri["slpm"]) / (m + K)
        f["xs_sapm"] = (s.ol + K * pri["slpm"]) / (m + K)
        f["xs_net"] = f.xs_slpm - f.xs_sapm
        f["xs_sacc"] = (s.sl + 20 * pri["sacc"]) / (s.sa + 20)
        f["xs_sdef"] = 1 - (s.ol + 20 * pri["sacc"]) / (s.oa + 20)
        f["xs_td15"] = 15 * (s.tdl + K * pri["td15"] / 15) / (m + K)
        f["xs_tdabs15"] = 15 * (s.otdl + K * pri["td15"] / 15) / (m + K)
        f["xs_tdacc"] = (s.tdl + 4 * pri["tdacc"]) / (s.tda + 4)
        f["xs_tddef"] = 1 - (s.otdl + 4 * pri["tdacc"]) / (s.otda + 4)
        f["xs_ctrl"] = (s.ctrl / 60 + K * pri["ctrl"]) / (m + K)
        f["xs_ctrl_abs"] = (s.octrl / 60 + K * pri["ctrl"]) / (m + K)
        f["xs_kd15"] = 15 * (s.kd + K * pri["kd15"] / 15) / (m + K)
        f["xs_kdabs15"] = 15 * (s.okd + K * pri["kd15"] / 15) / (m + K)
        f["xs_sub15"] = 15 * (s["sub"] + K * pri["sub15"] / 15) / (m + K)
        f["xs_headabs_pm"] = (s.head_abs + K * pri["head_pm"]) / (m + K)
        f["med_n"] = s.med_n
        f["med_share"] = (s.med_share + 1.0) / (s.med_n + 2)
        f["med_luck"] = (s.med_res - s.med_share) / (s.med_n + 2)
        f["med_robbed"] = s.med_robbed
        f["med_gifted"] = s.med_gifted
        f["jr_share"] = (s.jr_won + 3) / (s.jr_n + 6)
        f["jr_n"] = s.jr_n
        out = out.merge(f.rename(columns={k: f"{side}_{k}" for k in f.columns if k != "fight_id"}), on="fight_id", how="left")
    out.to_parquet(PROC / "canon_extra.parquet", index=False)
    print(out.shape, "coverage r_xs_n>0:", round(float((out.r_xs_n > 0).mean()), 3), "r_med_n>0:", round(float((out.r_med_n > 0).mean()), 3))


if __name__ == "__main__":
    main()
