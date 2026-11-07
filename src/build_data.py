"""Clean the ufcstats mirror (Greco1899/scrape_ufc_stats) into tidy tables.

Outputs (data/processed/):
  fights.parquet       one row per bout (red = first listed fighter on ufcstats)
  fighter_fights.parquet one row per (bout, fighter) with totals + opponent totals
  fighters.parquet     physical attributes per fighter key
"""
import re
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "greco"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

WEIGHT_LIMITS = {
    "Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
    "Lightweight": 155, "Welterweight": 170, "Middleweight": 185,
    "Light Heavyweight": 205, "Heavyweight": 265, "Super Heavyweight": 300,
    "Catch Weight": np.nan, "Open Weight": np.nan,
}


def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def parse_height(h):
    m = re.match(r"(\d+)' (\d+)\"", str(h))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def parse_reach(r):
    m = re.match(r"([\d.]+)\"", str(r))
    return float(m.group(1)) if m else np.nan


def parse_weight(w):
    m = re.match(r"(\d+)", str(w))
    return float(m.group(1)) if m else np.nan


def of_split(x):
    m = re.match(r"(\d+) of (\d+)", str(x))
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)


def mmss(x):
    m = re.match(r"(\d+):(\d+)", str(x))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan


def weightclass_info(wc):
    wc = clean(wc)
    women = wc.startswith("Women's")
    title = "Title" in wc
    base = None
    for k in sorted(WEIGHT_LIMITS, key=len, reverse=True):
        if k in wc:
            base = k
            break
    if base is None:
        base = "Catch Weight" if "Catch" in wc else ("Open Weight" if "Open" in wc else "Other")
    return base, women, title


def main():
    ev = pd.read_csv(RAW / "ufc_event_details.csv")
    ev["EVENT"] = ev.EVENT.map(clean)
    ev["DATE"] = pd.to_datetime(ev.DATE)
    fr = pd.read_csv(RAW / "ufc_fight_results.csv")
    fs = pd.read_csv(RAW / "ufc_fight_stats.csv")
    tott = pd.read_csv(RAW / "ufc_fighter_tott.csv")

    for df in (fr, fs):
        df["EVENT"] = df.EVENT.map(clean)
        df["BOUT"] = df.BOUT.map(clean)

    # ---- fighters ----
    tott["FIGHTER"] = tott.FIGHTER.map(clean)
    tott["height"] = tott.HEIGHT.map(parse_height)
    tott["reach"] = tott.REACH.map(parse_reach)
    tott["weight"] = tott.WEIGHT.map(parse_weight)
    tott["stance"] = tott.STANCE.fillna("Unknown").map(clean)
    tott["dob"] = pd.to_datetime(tott.DOB, errors="coerce", format="%b %d, %Y")
    tott["fid"] = tott.URL.str.extract(r"fighter-details/(\w+)")[0]
    name_counts = tott.FIGHTER.value_counts()
    dup_names = set(name_counts[name_counts > 1].index)

    # ---- fights ----
    # event names sometimes differ between files ("Noche UFC: X" vs "UFC Fight Night: X")
    known = set(ev.EVENT)
    suffix = ev.assign(sfx=ev.EVENT.str.split(":").str[-1].str.strip()).drop_duplicates("sfx", keep=False)
    sfx_map = dict(zip(suffix.sfx, suffix.EVENT))
    def fix_event(e):
        if e in known:
            return e
        return sfx_map.get(e.split(":")[-1].strip(), e)
    for df in (fr, fs):
        df["EVENT"] = df.EVENT.map(fix_event)
    # the mirror sometimes lists the same event under two names -> exact duplicate rows
    fr = fr.drop_duplicates("URL").copy()
    fs = fs.drop_duplicates(["EVENT", "BOUT", "ROUND", "FIGHTER"]).copy()
    fr = fr.merge(ev[["EVENT", "DATE", "LOCATION"]], on="EVENT", how="left")
    if fr.DATE.isna().any():
        print("dropping fights with unknown event date:", fr[fr.DATE.isna()].EVENT.value_counts().to_dict())
        fr = fr[fr.DATE.notna()].copy()
    fr["fight_id"] = fr.URL.str.extract(r"fight-details/(\w+)")[0]
    names = fr.BOUT.str.split(" vs. ", n=1, expand=True)
    fr["r_name"] = names[0].map(clean)
    fr["b_name"] = names[1].map(clean)
    oc = fr.OUTCOME.map(clean)
    fr["outcome"] = oc
    fr["winner"] = np.select([oc == "W/L", oc == "L/W"], [1, 0], default=-1)  # 1 red, 0 blue, -1 draw/nc
    fr["method"] = fr.METHOD.map(clean)
    fr["end_round"] = pd.to_numeric(fr.ROUND, errors="coerce")
    fr["end_time"] = fr.TIME.map(mmss)
    fmt = fr["TIME FORMAT"].map(clean)
    fr["sched_rounds"] = pd.to_numeric(fmt.str.extract(r"^(\d+) Rnd")[0], errors="coerce")
    # duration: sum of completed rounds (using format) + end time
    def duration(row):
        m = re.search(r"\(([\d-]+)\)", str(row["TIME FORMAT"]))
        if not m or pd.isna(row.end_round):
            return np.nan
        lens = [int(x) * 60 for x in m.group(1).split("-")]
        r = int(row.end_round)
        return sum(lens[: r - 1]) + (row.end_time if pd.notna(row.end_time) else 0)
    fr["duration"] = fr.apply(duration, axis=1)
    wci = fr.WEIGHTCLASS.map(weightclass_info)
    fr["weight_class"] = [x[0] for x in wci]
    fr["women"] = [x[1] for x in wci]
    fr["title"] = [x[2] for x in wci]
    fr["wc_limit"] = fr.weight_class.map(WEIGHT_LIMITS)
    fr["referee"] = fr.REFEREE.fillna("").map(clean)
    fr = fr.rename(columns={"EVENT": "event", "DATE": "date", "LOCATION": "location"})

    # disambiguate duplicate names by weight closeness
    dup_tott = tott[tott.FIGHTER.isin(dup_names)]
    name_to_fid = tott[~tott.FIGHTER.isin(dup_names)].drop_duplicates("FIGHTER").set_index("FIGHTER").fid.to_dict()

    tott_names = list(name_to_fid)

    def resolve(name, wc_limit):
        if name in name_to_fid:
            return name_to_fid[name]
        if name in dup_names:
            c = dup_tott[dup_tott.FIGHTER == name]
            if pd.isna(wc_limit) or c.weight.isna().all():
                return c.fid.iloc[0]
            return c.loc[(c.weight - wc_limit).abs().fillna(1e9).idxmin(), "fid"]
        m = process.extractOne(name, tott_names, scorer=fuzz.token_sort_ratio, score_cutoff=88)
        if m is not None and m[0] not in dup_names:
            return name_to_fid[m[0]]
        return "name:" + name  # fighter without a tott record
    fr["r_id"] = [resolve(n, w) for n, w in zip(fr.r_name, fr.wc_limit)]
    fr["b_id"] = [resolve(n, w) for n, w in zip(fr.b_name, fr.wc_limit)]

    # order: date, then event order (file is newest-first; within an event first row = main event)
    fr["row"] = np.arange(len(fr))
    fr["card_pos"] = fr.row - fr.groupby("event").row.transform("min")  # 0 = main event
    fr = fr.sort_values(["date", "row"], ascending=[True, False]).reset_index(drop=True)
    fr["order"] = np.arange(len(fr))

    fights = fr[["fight_id", "order", "date", "event", "location", "r_id", "b_id", "r_name", "b_name",
                 "winner", "outcome", "method", "end_round", "end_time", "duration", "sched_rounds", "weight_class",
                 "wc_limit", "women", "title", "referee", "card_pos", "DETAILS"]].rename(columns={"DETAILS": "details"})

    # ---- upcoming bouts to predict (data/upcoming.csv): flow through the same pipeline with outcome UPCOMING
    up_path = ROOT / "data" / "upcoming.csv"
    up_odds = []
    if up_path.exists():
        up = pd.read_csv(up_path, comment="#")
        if len(up):
            up["date"] = pd.to_datetime(up.date)
            last = fights.date.max()
            if (up.date <= last).any():
                print(f"warning: upcoming bouts dated on/before last completed event {last.date()} are skipped")
                up = up[up.date > last]
            rows = []
            for i, r in enumerate(up.itertuples(index=False)):
                base, women, _ = weightclass_info(str(r.weight_class))
                wl = WEIGHT_LIMITS.get(base, np.nan)
                fid = f"up_{i:04d}"
                rows.append({"fight_id": fid, "order": len(fights) + i, "date": r.date, "event": "UPCOMING",
                             "location": getattr(r, "location", "") if isinstance(getattr(r, "location", ""), str) else "",
                             "r_name": clean(r.red), "b_name": clean(r.blue), "r_id": resolve(clean(r.red), wl), "b_id": resolve(clean(r.blue), wl),
                             "winner": -1, "outcome": "UPCOMING", "method": "", "end_round": np.nan, "end_time": np.nan, "duration": np.nan,
                             "sched_rounds": float(getattr(r, "rounds", 3) or 3), "weight_class": base, "wc_limit": wl, "women": women,
                             "title": bool(int(getattr(r, "title", 0) or 0)), "referee": "",
                             "card_pos": float(getattr(r, "card_pos", i) if pd.notna(getattr(r, "card_pos", np.nan)) else i), "details": ""})
                up_odds.append({"fight_id": fid, "red_odds": getattr(r, "red_odds", np.nan), "blue_odds": getattr(r, "blue_odds", np.nan)})
            fights = pd.concat([fights, pd.DataFrame(rows)], ignore_index=True)
            print(f"upcoming bouts: {len(rows)}; unresolved names -> no UFC history:",
                  [n for n, i_ in zip(fights.r_name.tolist() + fights.b_name.tolist(), fights.r_id.tolist() + fights.b_id.tolist())
                   if str(i_).startswith("name:") and n in set(up.red.map(clean)) | set(up.blue.map(clean))])
    pd.DataFrame(up_odds, columns=["fight_id", "red_odds", "blue_odds"]).to_parquet(OUT / "upcoming_odds.parquet", index=False)
    fights.to_parquet(OUT / "fights.parquet", index=False)

    # ---- per fighter per fight totals ----
    fs = fs.dropna(subset=["FIGHTER"]).copy()
    fs["FIGHTER"] = fs.FIGHTER.map(clean)
    cols_of = ["SIG.STR.", "TOTAL STR.", "TD", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"]
    rec = {"EVENT": fs.EVENT, "BOUT": fs.BOUT, "FIGHTER": fs.FIGHTER}
    for c in cols_of:
        sp = fs[c].map(of_split)
        key = re.sub(r"[^a-z]+", "_", c.lower()).strip("_")
        rec[key + "_l"] = [a for a, b in sp]
        rec[key + "_a"] = [b for a, b in sp]
    rec["kd"] = pd.to_numeric(fs.KD, errors="coerce")
    rec["sub_att"] = pd.to_numeric(fs["SUB.ATT"], errors="coerce")
    rec["rev"] = pd.to_numeric(fs["REV."], errors="coerce")
    rec["ctrl"] = fs.CTRL.map(mmss)
    st = pd.DataFrame(rec)
    num = [c for c in st.columns if c not in ("EVENT", "BOUT", "FIGHTER")]
    tot = st.groupby(["EVENT", "BOUT", "FIGHTER"], sort=False)[num].sum(min_count=1).reset_index()
    rnd = pd.to_numeric(fs.ROUND.str.extract(r"(\d+)")[0], errors="coerce").values
    st["rnd"] = rnd
    r1 = st[st.rnd == 1].groupby(["EVENT", "BOUT", "FIGHTER"], sort=False)[["sig_str_l", "head_l"]].sum().add_prefix("r1_").reset_index()
    late = st[st.rnd >= 3].groupby(["EVENT", "BOUT", "FIGHTER"], sort=False)[["sig_str_l", "head_l"]].sum().add_prefix("late_").reset_index()
    tot = tot.merge(r1, on=["EVENT", "BOUT", "FIGHTER"], how="left").merge(late, on=["EVENT", "BOUT", "FIGHTER"], how="left")
    num = num + ["r1_sig_str_l", "r1_head_l", "late_sig_str_l", "late_head_l"]

    key = fights.merge(fr[["fight_id", "event", "r_name", "b_name"]].assign(BOUT=fr.r_name + " vs. " + fr.b_name)[["fight_id", "BOUT"]], on="fight_id")
    rows = []
    for side, opp in (("r", "b"), ("b", "r")):
        d = key[["fight_id", "event", "BOUT", f"{side}_name", f"{side}_id", f"{opp}_id"]].rename(
            columns={f"{side}_name": "FIGHTER", f"{side}_id": "fighter_id", f"{opp}_id": "opp_id", "event": "EVENT"})
        d["corner"] = side
        rows.append(d)
    ff = pd.concat(rows, ignore_index=True)
    ff = ff.merge(tot, on=["EVENT", "BOUT", "FIGHTER"], how="left")
    # opponent stats (absorbed)
    opp = ff[["fight_id", "fighter_id"] + num].rename(columns={"fighter_id": "opp_id", **{c: "opp_" + c for c in num}})
    ff = ff.merge(opp, on=["fight_id", "opp_id"], how="left")
    ff["has_stats"] = ff["sig_str_l"].notna()
    ff = ff.drop(columns=["EVENT", "BOUT", "FIGHTER"])
    ff.to_parquet(OUT / "fighter_fights.parquet", index=False)

    fighters = tott[["fid", "FIGHTER", "height", "reach", "weight", "stance", "dob"]].rename(columns={"fid": "fighter_id", "FIGHTER": "name"})
    fighters.to_parquet(OUT / "fighters.parquet", index=False)

    print("fights", fights.shape, "fighter_fights", ff.shape, "stats coverage", ff.has_stats.mean().round(3))
    print("unresolved ids", (fights.r_id.str.startswith("name:") | fights.b_id.str.startswith("name:")).sum())
    print(fights.groupby(fights.date.dt.year).winner.apply(lambda w: (w[w >= 0] == 1).mean()).round(3).to_string())


if __name__ == "__main__":
    main()
