"""Leakage audit. Each test targets one way a backtest can be inflated.

  python src/audit.py scramble 2024-06-01   future-scramble test: randomise every result, stat, odds line and pro-record
                                            result on/after the cutoff, rebuild ALL features from scratch, and check that
                                            every feature of every bout on/before the cutoff is bit-identical.
  python src/audit.py static                source-level checks: canon/ufcstats date alignment, ranking board dates,
                                            missing-value leaks, duplicate bouts, odds source agreement, column names
  python src/audit.py null                  label-permutation null: same pipeline on shuffled labels must score ~50 %
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

from common import PROC, ROOT, fighter_cols, load, logit, metrics

warnings.filterwarnings("ignore")
SRC = ROOT / "src"
DB = ROOT / "data" / "raw" / "canon" / "canon.db"
LABEL_COLS = ["winner", "outcome", "method", "end_round", "end_time", "duration", "details"]


# ----------------------------------------------------------------------------------------------- future scramble
def scramble(cut):
    cut = pd.Timestamp(cut)
    fights = pd.read_parquet(PROC / "fights.parquet")
    dates = np.sort(fights[fights.outcome != "UPCOMING"].date.unique())
    cut = pd.Timestamp(dates[dates >= cut][0])  # snap to a real event date
    rng = np.random.default_rng(int(cut.value % 2**31))
    tmp = tempfile.mkdtemp(prefix=f"ufc_audit_{cut.date()}_")
    print(f"cutoff event date {cut.date()}  workdir {tmp}", flush=True)

    # ufcstats results + stats on/after the cutoff
    f = fights.copy()
    m = (f.date >= cut) & (f.outcome != "UPCOMING")
    flip = m & (rng.random(len(f)) < 0.5) & f.outcome.isin(["W/L", "L/W"])
    f.loc[flip, "outcome"] = f.loc[flip, "outcome"].map({"W/L": "L/W", "L/W": "W/L"})
    f.loc[flip, "winner"] = 1 - f.loc[flip, "winner"]
    f.loc[m, "method"] = rng.choice(["KO/TKO", "Submission", "Decision - Unanimous", "Decision - Split"], m.sum())
    f.loc[m, "duration"] = rng.uniform(10, 1500, m.sum()).round()
    f.loc[m, "end_round"] = rng.integers(1, 4, m.sum())
    f.to_parquet(os.path.join(tmp, "fights.parquet"), index=False)
    ff = pd.read_parquet(PROC / "fighter_fights.parquet")
    mf = ff.fight_id.isin(f.fight_id[m])
    num = [c for c in ff.columns if c not in ("fight_id", "fighter_id", "opp_id", "corner", "has_stats")]
    ff.loc[mf, num] = (ff.loc[mf, num].astype(float) * rng.uniform(0, 3, (mf.sum(), len(num)))).round()
    ff.to_parquet(os.path.join(tmp, "fighter_fights.parquet"), index=False)
    shutil.copy(PROC / "fighters.parquet", tmp)
    # odds of later bouts (they feed the market-history features of later bouts only)
    o = pd.read_parquet(PROC / "odds.parquet")
    mo = o.fight_id.isin(f.fight_id[f.date > cut])
    o.loc[mo, "mkt_close"] = rng.uniform(0.05, 0.95, mo.sum())
    o.to_parquet(os.path.join(tmp, "odds.parquet"), index=False)
    # pro records (all promotions) on/after the cutoff
    db = os.path.join(tmp, "canon.db")
    shutil.copy(DB, db)
    c = sqlite3.connect(db)
    ids = [r[0] for r in c.execute("select f.fight_id from fights f join events e using(event_id) where e.date >= ?", (str(cut.date()),))]
    swap = [(i,) for i in ids if rng.random() < 0.5]
    c.executemany("update fight_participants set result = case result when 'W' then 'L' when 'L' then 'W' else result end where fight_id = ?", swap)
    c.executemany("update fights set method = ? where fight_id = ?", [(str(rng.choice(["KO/TKO", "Submission", "Decision"])), i) for i in ids])
    c.commit()
    c.close()
    print(f"scrambled: {int(m.sum())} ufcstats bouts ({int(flip.sum())} winners flipped), {int(mo.sum())} odds lines, "
          f"{len(ids)} pro fights ({len(swap)} flipped)", flush=True)

    env = {**os.environ, "UFC_PROC": tmp, "UFC_CANON_DB": db}
    for script in ("canon_feats.py", "features.py"):
        subprocess.run([sys.executable, str(SRC / script)], env=env, check=True, stdout=subprocess.DEVNULL)

    a = pd.read_parquet(PROC / "features.parquet").set_index("fight_id")
    b = pd.read_parquet(os.path.join(tmp, "features.parquet")).set_index("fight_id").loc[a.index, a.columns]
    cols = [c for c in a.columns if c not in LABEL_COLS]

    def n_diff(rows):
        bad = {}
        for col in cols:
            x, y = a.loc[rows, col], b.loc[rows, col]
            if pd.api.types.is_numeric_dtype(x):
                dif = ~((x == y) | (x.isna() & y.isna()))
            else:
                dif = ~((x.astype(str) == y.astype(str)))
            if dif.any():
                bad[col] = int(dif.sum())
        return bad

    pre = a.index[(a.date <= cut) & (a.outcome != "UPCOMING")]
    post = a.index[(a.date > cut) & (a.outcome != "UPCOMING")]
    bad_pre = n_diff(pre)
    bad_post = n_diff(post)
    print(f"\nbouts on/before cutoff: {len(pre)}  ({(a.loc[pre].date == cut).sum()} on the cutoff night itself)")
    print(f"  feature columns that changed: {len(bad_pre)} of {len(cols)}" + (f"  -> LEAK: {bad_pre}" if bad_pre else "  -> no leakage"))
    print(f"bouts after cutoff: {len(post)}   feature columns that changed: {len(bad_post)} of {len(cols)} (sanity: should be many)")
    shutil.rmtree(tmp, ignore_errors=True)
    return not bad_pre and len(bad_post) > 50


# ----------------------------------------------------------------------------------------------- static checks
def static():
    d = load()
    ok = True
    fights = pd.read_parquet(PROC / "fights.parquet")

    print("1) canon (Sherdog) vs ufcstats dates for the same bout")
    c = sqlite3.connect(DB)
    fa = pd.read_sql("select fa.source_id as fight_id, e.date as cdate from fight_aliases fa join fights f on f.fight_id = fa.fight_id "
                     "join events e on e.event_id = f.event_id where fa.source = 'ufcstats'", c)
    fta = pd.read_sql("select fighter_id, source_id from fighter_aliases where source = 'ufcstats'", c)
    pf = pd.read_sql("select p.fight_id, p.fighter_id, e.date as cdate from fight_participants p join fights f using(fight_id) "
                     "join events e using(event_id)", c)
    c.close()
    fa["cdate"] = pd.to_datetime(fa.cdate, errors="coerce")
    x = fights.merge(fa, on="fight_id")
    early = (x.cdate < x.date).sum()
    print(f"   linked bouts {len(x)}: canon date earlier than ufcstats date: {early}, later: {(x.cdate > x.date).sum()}")
    # same pairing via fighter ids, within +-3 days (catches bouts without a fight alias)
    fmap = dict(zip(fta.source_id, fta.fighter_id))
    pf["cdate"] = pd.to_datetime(pf.cdate, errors="coerce")
    pair = pf.groupby("fight_id").agg(a=("fighter_id", "min"), b=("fighter_id", "max"), cdate=("cdate", "first"))
    u = fights.assign(ca=fights.r_id.map(fmap), cb=fights.b_id.map(fmap)).dropna(subset=["ca", "cb"])
    u["a"] = u[["ca", "cb"]].min(axis=1).astype(int)
    u["b"] = u[["ca", "cb"]].max(axis=1).astype(int)
    j = u.merge(pair, on=["a", "b"])
    j = j[(j.cdate - j.date).abs().dt.days <= 3]
    n_early = (j.cdate < j.date).sum()
    print(f"   same pairing within 3 days: {len(j)} matches, canon dated EARLIER (would leak the result): {n_early}")
    ok &= early == 0 and n_early == 0

    print("2) official rankings: board date must be before the event")
    w = pd.read_parquet(ROOT / "data" / "raw" / "nlbair" / "ufc_all_fights_wide.parquet")
    dcol = [c for c in w.columns if c in ("event_date", "date")][0]
    for s in ("f1", "f2"):
        bd = pd.to_datetime(w[f"{s}_rank_board_date"], errors="coerce")
        ed = pd.to_datetime(w[dcol], errors="coerce")
        v = bd.notna()
        print(f"   {s}: {v.sum()} boards, on/after event date: {(bd[v] >= ed[v]).sum()}")
        ok &= (bd[v] >= ed[v]).sum() == 0

    print("3) missing values that could reveal the outcome (2010+ fights, red side, by UFC experience)")
    e = d[d.date >= "2010-01-01"]
    for col in ("reach", "height", "age", "pro_n", "welo", "rank"):
        cc = f"r_{col}" if f"r_{col}" in e else f"rank_r"
        if cc not in e:
            continue
        miss = e[cc].isna()
        for lab, grp in (("debut", e.r_n_fights == 0), ("veteran", e.r_n_fights > 0)):
            mm = miss & grp
            if mm.sum() >= 20:
                wr_m, wr_k = e.y[mm].mean(), e.y[~miss & grp].mean()
                mk_m, mk_k = e.mkt_close[mm].mean(), e.mkt_close[~miss & grp].mean()
                flag = abs((wr_m - mk_m) - (wr_k - mk_k)) > 0.06 and mm.sum() > 50
                print(f"   {cc:10s} {lab:8s} missing {mm.sum():5d}: win {wr_m:.3f} (market {mk_m:.3f})  vs present win {wr_k:.3f} (market {mk_k:.3f})"
                      + ("   <-- CHECK" if flag else ""))
    f = pd.read_parquet(PROC / "fighters.parquet")
    print(f"   raw reach values that are non-integer: {(f.reach.dropna() % 1 != 0).mean():.3f}")

    print("4) duplicate bouts")
    k = fights[fights.outcome != "UPCOMING"].assign(lo=lambda t: t[["r_id", "b_id"]].min(axis=1), hi=lambda t: t[["r_id", "b_id"]].max(axis=1))
    k = k.sort_values("date")
    gap = k.groupby(["lo", "hi"]).date.diff().dt.days
    nc = k.outcome.eq("NC/NC")
    prev_nc = nc.groupby([k.lo, k.hi]).shift().fillna(False).astype(bool)  # an immediate rematch after a no-contest is real
    dup = (gap <= 30) & ~prev_nc & ~nc
    print(f"   same pairing twice within 30 days: {(gap <= 30).sum()} ({(gap <= 30).sum() - dup.sum()} are rematches after a no-contest),"
          f"  duplicate fight_id: {k.fight_id.duplicated().sum()}")
    ok &= dup.sum() == 0 and k.fight_id.duplicated().sum() == 0

    print("5) odds sources agree with each other (test 2024+)")
    o = pd.read_parquet(PROC / "odds.parquet")
    t = d[d.date >= "2024-01-01"][["fight_id", "y"]].merge(o, on="fight_id")
    srcs = {"nlbair": "n_close_r", "canon": "c_close_r", "ultimate": "mkt_p_red"}
    both = t.dropna(subset=list(srcs.values()))
    print(f"   fights with all three sources: {len(both)}")
    for a_, ca in srcs.items():
        acc = ((both[ca] > 0.5) == (both.y == 1)).mean()
        print(f"   {a_:9s} favourite won {acc:.4f}   corr with nlbair {np.corrcoef(both[ca], both.n_close_r)[0, 1]:.3f}")
    dis = both[((both.n_close_r > 0.5) != (both.c_close_r > 0.5))]
    print(f"   nlbair vs canon disagree on the favourite in {len(dis)} fights (pick'em-ish); nlbair right {((dis.n_close_r > .5) == (dis.y == 1)).mean():.2f}")

    print("6) feature names that look post-fight")
    fc = fighter_cols(d)
    sus = [c for c in fc if any(s in c for s in ("winner", "outcome", "method", "end_", "duration", "result", "details"))]
    print(f"   {len(fc)} per-fighter features; suspicious: {sus or 'none'}")
    ok &= not sus
    print("\nSTATIC CHECKS", "PASS" if ok else "FAIL (see above)")
    return ok


# ----------------------------------------------------------------------------------------------- null test
def null(seeds=(0, 1, 2)):
    from final_model import fit_stats_one, predict_stats_one, fit_stacker, predict_final
    d = load()
    fcols = fighter_cols(d)
    rs = []
    for s in seeds:
        rng = np.random.default_rng(s)
        dd = d.copy()
        # permute labels within year so base rates are kept but the link features -> outcome is broken
        dd["y"] = dd.groupby(dd.date.dt.year).y.transform(lambda v: rng.permutation(v.values))
        dd["lg_ens"] = np.nan
        for y in range(2016, 2027):
            tr = dd[(dd.date >= "2001-01-01") & (dd.date < f"{y}-01-01")]
            te = dd[dd.date.dt.year == y]
            fm = fit_stats_one(tr, fcols, "lgb_small", n_jobs=8)
            dd.loc[te.index, "lg_ens"] = logit(predict_stats_one(fm, te, fcols))
        dd["p_final"] = np.nan
        for y in range(2019, 2027):
            tr = dd[(dd.date >= "2016-01-01") & (dd.date < f"{y}-01-01")]
            te = dd[dd.date.dt.year == y]
            dd.loc[te.index, "p_final"] = predict_final(fit_stacker(tr), te)
        t = dd[dd.date >= "2024-01-01"]
        ms, mf = metrics(1 / (1 + np.exp(-t.lg_ens)), t.y), metrics(t.p_final, t.y)
        red = t.y.mean()
        rs.append((ms["acc"], mf["acc"], max(red, 1 - red)))
        print(f"seed {s}: shuffled-label test acc  stats {ms['acc']:.4f}  final(with market) {mf['acc']:.4f}  "
              f"(always-red on the shuffled labels {max(red, 1 - red):.4f}, n={ms['n']})", flush=True)
    print(f"mean: stats {np.mean([r[0] for r in rs]):.4f}  final {np.mean([r[1] for r in rs]):.4f}  "
          f"no-skill reference (always red) {np.mean([r[2] for r in rs]):.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scramble":
        sys.exit(0 if scramble(sys.argv[2]) else 1)
    elif cmd == "static":
        sys.exit(0 if static() else 1)
    elif cmd == "null":
        null()
