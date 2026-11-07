"""Betting backtest: is the model profitable against real bookmaker prices (margin included)?

Prices: BestFightOdds moneylines via nlbair/ufc-data (fight-canon-db as fallback), oriented to the red corner.
  open        one opening price per fighter
  close_worst the worst closing price across the books BFO tracks
  close_mid   the average of the worst and best closing price (a typical single book)
  close_best  the best closing price across books (line shopping; optimistic: some books have low limits)

Model probabilities are walk-forward (each year predicted only by models fit on earlier years):
  close model  the final model (stats ensemble + closing line) -> bets struck at the close
  open model   the same stacker refit with the OPENING line as its only market input -> bets struck at the open
  stats model  the stats ensemble alone (never sees any odds)

dev = 2016-2023 (strategy parameters are read off here), test = 2024-2026 (out of sample).
Needs data/processed/wf_stats_oof.parquet, written by src/evaluate.py.
Run: python src/betting.py   -> results/betting_report.json, results/betting_bets.csv
"""
import json
import sqlite3
import warnings

import numpy as np
import pandas as pd

from common import PROC, ROOT, load
from final_model import fit_stacker, predict_final

warnings.filterwarnings("ignore")
RES = ROOT / "results"
DEV, TEST = (2016, 2023), (2024, 2026)
rng = np.random.default_rng(0)
# bet-time sanity rule, fixed a priori (not tuned): an estimated edge above 30 % is almost always a bad line
# (e.g. an opener posted with the names swapped), which a book would void as a palpable error
MAX_EV = 0.30


def dec(ml):
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml > 0, 1 + ml / 100, np.where(ml < 0, 1 + 100 / -ml, np.nan))


# ------------------------------------------------------------------------------------------------ prices
def load_prices():
    fights = pd.read_parquet(PROC / "fights.parquet")[["fight_id", "r_id", "b_id", "r_name", "b_name"]]
    w = pd.read_parquet(ROOT / "data" / "raw" / "nlbair" / "ufc_all_fights_wide.parquet").merge(fights, on="fight_id")
    same = ((w.f1_id == w.r_id) | ((w.f1_id != w.b_id) & (w.f1_name == w.r_name))).values
    n = pd.DataFrame({"fight_id": w.fight_id})
    for k in ("open", "close_low", "close_high"):
        a, b = dec(w[f"f1_{k}"]), dec(w[f"f2_{k}"])
        n[f"r_{k}"], n[f"b_{k}"] = np.where(same, a, b), np.where(same, b, a)
    c = sqlite3.connect(ROOT / "data" / "raw" / "canon" / "canon.db")
    co = pd.read_sql("""select fa.source_id as fight_id, xa.source_id as uid, o.open, o.close_low, o.close_high
                        from odds o join fight_aliases fa on fa.fight_id=o.fight_id and fa.source='ufcstats'
                        join fighter_aliases xa on xa.fighter_id=o.fighter_id and xa.source='ufcstats' where o.book='all'""", c)
    c.close()
    co = co.drop_duplicates(["fight_id", "uid"]).merge(fights[["fight_id", "r_id", "b_id"]], on="fight_id")
    r = co[co.uid == co.r_id].set_index("fight_id")
    b = co[co.uid == co.b_id].set_index("fight_id")
    cc = pd.DataFrame(index=r.index.intersection(b.index))
    for k in ("open", "close_low", "close_high"):
        cc[f"r_{k}"], cc[f"b_{k}"] = dec(r.loc[cc.index, k]), dec(b.loc[cc.index, k])
    cc = cc[~cc.index.isin(n.fight_id)].reset_index()
    p = pd.concat([n.assign(src="nlbair"), cc.assign(src="canon")], ignore_index=True)
    for s in "rb":  # worst/best by payout, whatever order the source used
        lo, hi = p[f"{s}_close_low"], p[f"{s}_close_high"]
        p[f"{s}_close_worst"], p[f"{s}_close_best"] = np.minimum(lo, hi), np.maximum(lo, hi)
        p[f"{s}_close_mid"] = (lo + hi) / 2
    p["over_mid"] = 1 / p.r_close_mid + 1 / p.b_close_mid
    p["over_best"] = 1 / p.r_close_best + 1 / p.b_close_best
    p["over_open"] = 1 / p.r_open + 1 / p.b_open
    return p


# ------------------------------------------------------------------------------------------------ model probabilities
def open_frame(d):
    """What the model would see when the opening line is the latest market information."""
    o = d.copy()
    o["mkt_close"] = d.mkt_open
    o["mkt_p_red"] = d.mkt_open
    o["n_close_spread"] = np.nan
    return o


def walk_probs(d):
    d = d.copy()
    for col in ("p_close", "p_open"):
        d[col] = np.nan
    do = open_frame(d)
    for y in range(DEV[0], TEST[1] + 1):
        tr = (d.date >= "2010-01-01") & (d.date < f"{y}-01-01")
        te = d.date.dt.year == y
        d.loc[te, "p_close"] = predict_final(fit_stacker(d[tr]), d[te])
        st_o = fit_stacker(do[tr & do.mkt_close.notna()])
        po = predict_final(st_o, do[te])
        d.loc[te, "p_open"] = np.where(do.loc[te, "mkt_close"].notna(), po, np.nan)
    d["p_stats"] = 1 / (1 + np.exp(-d.lg_ens))
    # market-only recalibration: logit(p) = a * logit(close), a fitted walk-forward (no stats at all)
    from sklearn.linear_model import LogisticRegression
    from common import logit
    d["p_mktcal"] = np.nan
    for y in range(DEV[0], TEST[1] + 1):
        tr = d[(d.date >= "2010-01-01") & (d.date < f"{y}-01-01") & d.mkt_close.notna()]
        x = logit(tr.mkt_close.values)[:, None]
        m = LogisticRegression(C=1e6, fit_intercept=False).fit(np.vstack([x, -x]), np.r_[tr.y, 1 - tr.y])
        te = (d.date.dt.year == y) & d.mkt_close.notna()
        d.loc[te, "p_mktcal"] = m.predict_proba(logit(d.loc[te, "mkt_close"].values)[:, None])[:, 1]
    return d


# ------------------------------------------------------------------------------------------------ strategies
def flat_bets(g, p_col, price, rule, t=0.0, max_ev=None):
    """Returns DataFrame of bets: side, odds, stake, profit, fair (no-vig closing prob of the side)."""
    pr, pb = g[f"r_{price}"].values, g[f"b_{price}"].values
    p = g[p_col].values
    fav_r = pr < pb  # favourite by the price available when betting (never the closing line for opening bets)
    ev_r, ev_b = p * pr - 1, (1 - p) * pb - 1
    if rule == "favourite":
        side = np.where(fav_r, 1, 0); take = np.ones(len(g), bool)
    elif rule == "underdog":
        side = np.where(fav_r, 0, 1); take = np.ones(len(g), bool)
    elif rule == "pick":
        side = (p > 0.5).astype(int); take = np.ones(len(g), bool)
    elif rule == "pick_underdog":
        side = (p > 0.5).astype(int); take = (p > 0.5) != fav_r
    elif rule == "value":
        side = np.where(ev_r >= ev_b, 1, 0); take = np.maximum(ev_r, ev_b) > t
    elif rule == "value_underdog":
        side = np.where(fav_r, 0, 1); take = np.where(fav_r, ev_b, ev_r) > t
    elif rule == "value_favourite":
        side = np.where(fav_r, 1, 0); take = np.where(fav_r, ev_r, ev_b) > t
    else:
        raise ValueError(rule)
    take &= ~np.isnan(pr) & ~np.isnan(pb) & ~np.isnan(p)
    if rule.startswith("value"):
        cap = MAX_EV if max_ev is None else max_ev
        take &= np.where(side == 1, ev_r, ev_b) <= cap
    odds = np.where(side == 1, pr, pb)
    won = np.where(side == 1, g.y.values == 1, g.y.values == 0)
    stake = np.ones(len(g))
    if rule == "pick" and t == "conf":  # stake grows with the model's confidence, price ignored
        stake = 1 + 20 * (np.maximum(p, 1 - p) - 0.5)
    b = pd.DataFrame({"fight_id": g.fight_id.values, "date": g.date.values, "side": side, "odds": odds, "stake": stake,
                      "won": won, "p": np.where(side == 1, p, 1 - p),
                      "fair": np.where(side == 1, g.mkt_close.values, 1 - g.mkt_close.values)})[take]
    b["profit"] = np.where(b.won, b.stake * (b.odds - 1), -b.stake)
    return b


def kelly_bets(g, p_col, price, frac, cap=0.05, day_cap=0.25, t=0.0):
    """Fractional Kelly, bankroll updated after each event night; stakes are fractions of the current bankroll."""
    bets = flat_bets(g, p_col, price, "value", t)
    if bets.empty:
        return bets, 1.0, 0.0
    f = frac * (bets.p * bets.odds - 1) / (bets.odds - 1)
    bets["frac"] = f.clip(0, cap)
    bank, peak, mdd, rows = 1.0, 1.0, 0.0, []
    for _, day in bets.groupby("date", sort=True):
        fr = day.frac.values
        if fr.sum() > day_cap:
            fr = fr * day_cap / fr.sum()
        st = bank * fr
        pnl = np.where(day.won, st * (day.odds.values - 1), -st)
        rows.append(day.assign(stake=st, profit=pnl))
        bank += pnl.sum()
        peak = max(peak, bank)
        mdd = max(mdd, 1 - bank / peak)
    return pd.concat(rows), bank, mdd


def summary(b, boot=True):
    if b is None or len(b) == 0:
        return {"bets": 0}
    s = {"bets": int(len(b)), "staked": round(float(b.stake.sum()), 2), "profit": round(float(b.profit.sum()), 2),
         "roi": round(float(b.profit.sum() / b.stake.sum()), 4), "hit_rate": round(float(b.won.mean()), 3),
         "avg_odds": round(float(b.odds.mean()), 2),
         # what the bets were worth at fair (no-margin) closing prices: luck-free estimate of the edge
         "fair_roi": round(float(((b.fair * b.odds - 1) * b.stake).sum() / b.stake.sum()), 4)}
    if boot and len(b) >= 10:
        days = b.date.values
        u = np.unique(days)
        gp = b.groupby("date").profit.sum().reindex(u).values
        gs = b.groupby("date").stake.sum().reindex(u).values
        idx = rng.integers(0, len(u), (2000, len(u)))
        rois = gp[idx].sum(1) / gs[idx].sum(1)
        s["roi_ci95"] = [round(float(np.quantile(rois, 0.025)), 4), round(float(np.quantile(rois, 0.975)), 4)]
        s["p_profit_le_0"] = round(float((rois <= 0).mean()), 3)
    return s


def robustness(g, rep):
    """Pick each family's edge threshold on DEV (best ROI with >= 300 bets), then stress-test it."""
    fam = [("close_mid", "p_close"), ("close_best", "p_close"), ("open", "p_open"), ("close_mid", "p_mktcal"), ("close_best", "p_mktcal")]
    grid = (0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15)
    dev, test = period(g, DEV), period(g, TEST)
    out = []
    for price, pc in fam:
        cand = [(t, flat_bets(dev, pc, price, "value", t)) for t in grid]
        ok = [(t, b) for t, b in cand if len(b) >= 300]
        t, bd = max(ok, key=lambda tb: tb[1].profit.sum() / tb[1].stake.sum()) if ok else cand[0]  # fallback: loosest rule
        if len(bd) == 0:
            continue
        bt = flat_bets(test, pc, price, "value", t)
        r = {"price": price, "model": pc, "threshold_chosen_on_dev": t, "dev": summary(bd), "test": summary(bt)}
        # concentration: profit without the 10 biggest winning bets
        for lab, b in (("dev", bd), ("test", bt)):
            top = b.sort_values("profit", ascending=False).iloc[10:]
            r[f"{lab}_roi_without_top10_wins"] = round(float(top.profit.sum() / top.stake.sum()), 4)
            cap = b[b.odds <= 4.0]
            r[f"{lab}_roi_odds_le_4"] = round(float(cap.profit.sum() / cap.stake.sum()), 4)
            yr = b.groupby(pd.to_datetime(b.date).dt.year).apply(lambda x: x.profit.sum() / x.stake.sum())
            r[f"{lab}_years_positive"] = f"{int((yr > 0).sum())}/{len(yr)}"
            src = b.merge(g[["fight_id", "src"]], on="fight_id")
            r[f"{lab}_roi_by_source"] = {k: round(float(v.profit.sum() / v.stake.sum()), 4) for k, v in src.groupby("src") if len(v) >= 30}
        for lab, per in (("dev", dev), ("test", test)):
            for cap in (0.2, 0.5, 99.0):
                bb = flat_bets(per, pc, price, "value", t, max_ev=cap)
                r[f"{lab}_roi_ev_cap_{cap:g}"] = round(float(bb.profit.sum() / bb.stake.sum()), 4) if len(bb) else float("nan")
            bb = flat_bets(per, pc, price, "value", t)
            mv = (per.set_index("fight_id").loc[bb.fight_id, "mkt_close"].values - per.set_index("fight_id").loc[bb.fight_id, "mkt_open"].values)
            keep = ~(np.abs(mv) > 0.15)
            r[f"{lab}_roi_excl_open_close_moves_gt_15pts"] = round(float(bb.profit[keep].sum() / bb.stake[keep].sum()), 4)
        if price == "open":  # execution: filled part of the way from the opening price to the closing price
            for lab, per, b in (("dev", dev, bd), ("test", test, bt)):
                m = per.set_index("fight_id")
                for lam in (0.25, 0.5, 1.0):
                    qo = 1 / np.where(b.side == 1, m.loc[b.fight_id, "r_open"], m.loc[b.fight_id, "b_open"])
                    qc = 1 / np.where(b.side == 1, m.loc[b.fight_id, "r_close_mid"], m.loc[b.fight_id, "b_close_mid"])
                    o = 1 / (qo + lam * (qc - qo))
                    prof = np.where(b.won, o - 1, -1.0)
                    r[f"{lab}_roi_filled_{int(lam * 100)}pct_toward_close"] = round(float(prof.mean()), 4)
        out.append(r)
    rep["robustness"] = out
    print("\nROBUSTNESS (threshold chosen on dev, >= 300 dev bets)")
    for r in out:
        print(f"\n{r['price']} / {r['model']}  edge > {r['threshold_chosen_on_dev']:.2f}")
        for lab in ("dev", "test"):
            s_ = r[lab]
            ex = "  ".join(f"{k[len(lab) + 1:]}={v:+.2%}" for k, v in r.items() if k.startswith(lab + "_roi_") and isinstance(v, float))
            print(f"  {lab:4s} {s_['bets']:5d} bets ROI {s_['roi']:+.2%} CI{[f'{x:+.1%}' for x in s_.get('roi_ci95', [])]} "
                  f"P(ROI<=0)={s_.get('p_profit_le_0')} fair {s_['fair_roi']:+.2%} avg odds {s_['avg_odds']} | years+ {r[lab + '_years_positive']}")
            print(f"       {ex}  by source {r[lab + '_roi_by_source']}")


def realistic(g, rep):
    """$10k bankroll, quarter Kelly, $500 max bet, 25 % max exposure per night; opening bets filled halfway to the close."""
    out = []
    for price, pc, t, fill in (("close_mid", "p_close", 0.05, 0.0), ("close_best", "p_close", 0.10, 0.0),
                               ("open", "p_open", 0.10, 0.0), ("open", "p_open", 0.10, 0.5), ("open", "p_open", 0.05, 0.5)):
        for lab, yrs in (("dev", DEV), ("test", TEST)):
            per = period(g, yrs).set_index("fight_id", drop=False)
            b = flat_bets(per, pc, price, "value", t)
            if fill:
                qo = 1 / b.odds.values
                qc = 1 / np.where(b.side == 1, per.loc[b.fight_id, "r_close_mid"], per.loc[b.fight_id, "b_close_mid"])
                b["fill_odds"] = 1 / (qo + fill * (qc - qo))
            else:
                b["fill_odds"] = b.odds
            kf = (0.25 * (b.p * b.odds - 1) / (b.odds - 1)).clip(0, 0.05).values
            bank, peak, mdd, staked, n = 10000.0, 10000.0, 0.0, 0.0, 0
            for dte in np.unique(b.date.values):
                m = (b.date.values == dte)
                st = np.minimum(bank * kf[m], 500.0)
                if st.sum() > 0.25 * bank:
                    st *= 0.25 * bank / st.sum()
                pnl = np.where(b.won.values[m], st * (b.fill_odds.values[m] - 1), -st).sum()
                bank += pnl; staked += st.sum(); n += int(m.sum())
                peak = max(peak, bank); mdd = max(mdd, 1 - bank / peak)
            out.append({"price": price, "model": pc, "edge_gt": t, "fill_toward_close": fill, "period": lab, "bets": n,
                        "staked": round(staked), "profit": round(bank - 10000), "roi": round((bank - 10000) / staked, 4) if staked else None,
                        "max_drawdown": round(mdd, 3), "years": f"{yrs[0]}-{yrs[1]}"})
    rep["realistic_10k"] = out
    print("\nREALISTIC: $10,000 bankroll, quarter Kelly, max $500/bet")
    for r in out:
        print(f"  {r['period']:4s} {r['price']:10s} {r['model']:7s} edge>{r['edge_gt']:.2f} fill {r['fill_toward_close']:.0%}: "
              f"{r['bets']:4d} bets, staked ${r['staked']:>8,}, profit ${r['profit']:>+9,}  ROI {r['roi']:+.1%}  max drawdown {r['max_drawdown']:.0%}")


def period(d, years):
    return d[d.date.dt.year.between(*years)]


def main():
    d = load()
    d = d.merge(pd.read_parquet(PROC / "wf_stats_oof.parquet")[["fight_id", "lg_ens"]], on="fight_id", how="left")
    d = walk_probs(d)
    prices = load_prices()
    g = d[d.date.dt.year.between(DEV[0], TEST[1])].merge(prices, on="fight_id", how="inner")
    bad = ~g.over_mid.between(1.0, 1.15) | g.mkt_close.isna()
    print(f"fights with prices {len(g)}, dropped {int(bad.sum())} with missing/implausible lines (margin outside 0-15%)")
    g = g[~bad].reset_index(drop=True)
    rep = {"n_fights": {"dev": int(len(period(g, DEV))), "test": int(len(period(g, TEST)))},
           "bookmaker_margin": {k: {"mid_close": round(float(period(g, yrs).over_mid.mean() - 1), 4),
                                    "best_close": round(float(period(g, yrs).over_best.mean() - 1), 4),
                                    "open": round(float(period(g, yrs).over_open.mean() - 1), 4)} for k, yrs in (("dev", DEV), ("test", TEST))}}
    print("average bookmaker margin:", rep["bookmaker_margin"])
    for c in ("p_close", "p_open", "p_stats"):
        t = period(g, TEST)
        ok = t[c].notna()
        print(f"  {c}: test accuracy {((t[c][ok] > .5) == (t.y[ok] == 1)).mean():.4f} on {ok.sum()} fights")

    open_bad = ~g.over_open.between(1.0, 1.20)
    g.loc[open_bad, ["r_open", "b_open"]] = np.nan  # implausible opening pair (known at bet time): no opening bet
    print(f"opening lines unusable: {int(open_bad.sum())}")
    flat_specs = [("favourite", "p_close", None), ("underdog", "p_close", None), ("pick", "p_close", None),
                  ("pick", "p_close", "conf"), ("pick_underdog", "p_close", None)]
    for t in (0.0, 0.03, 0.05, 0.10, 0.15):
        flat_specs += [("value", "p_close", t), ("value_underdog", "p_close", t), ("value_favourite", "p_close", t)]
    open_specs = [("pick", "p_open", None), ("pick_underdog", "p_open", None)] + [("value", "p_open", t) for t in (0.0, 0.03, 0.05, 0.10, 0.15)]
    stats_specs = [("value", "p_stats", t) for t in (0.0, 0.05, 0.10, 0.20)]
    cal_specs = [("value", "p_mktcal", t) for t in (0.0, 0.03, 0.05, 0.10)]
    out, all_bets = [], []
    for price, specs in (("close_worst", flat_specs), ("close_mid", flat_specs), ("close_best", flat_specs),
                         ("open", open_specs), ("close_mid", stats_specs), ("open", stats_specs),
                         ("close_mid", cal_specs), ("close_best", cal_specs)):
        for rule, pc, t in specs:
            name = rule + ("" if t is None else ("_conf_stake" if t == "conf" else f">{t:.2f}"))
            row = {"price": price, "model": pc, "strategy": name}
            for lab, yrs in (("dev", DEV), ("test", TEST)):
                b = flat_bets(period(g, yrs), pc, price, rule, 0.0 if t in (None, "conf") else t) if t != "conf" else \
                    flat_bets(period(g, yrs), pc, price, "pick", "conf")
                row[lab] = summary(b)
                if lab == "test":
                    all_bets.append(b.assign(price=price, model=pc, strategy=name))
            out.append(row)
    kel = []
    for price, pc in (("close_mid", "p_close"), ("close_best", "p_close"), ("open", "p_open")):
        for frac in (0.1, 0.25, 0.5):
            for t in (0.0, 0.05):
                row = {"price": price, "model": pc, "strategy": f"kelly_{frac}_edge>{t:.2f}"}
                for lab, yrs in (("dev", DEV), ("test", TEST)):
                    b, bank, mdd = kelly_bets(period(g, yrs), pc, price, frac, t=t)
                    row[lab] = {**summary(b), "bankroll_x": round(float(bank), 3), "max_drawdown": round(float(mdd), 3)}
                kel.append(row)
    robustness(g, rep)
    realistic(g, rep)
    rep["flat"] = out
    rep["kelly"] = kel
    # per-year for the headline strategies
    per = []
    for price, pc, rule, t in (("close_mid", "p_close", "value", 0.05), ("close_best", "p_close", "value", 0.05),
                               ("open", "p_open", "value", 0.05), ("close_mid", "p_close", "favourite", 0.0),
                               ("close_mid", "p_close", "pick_underdog", 0.0)):
        for y in range(DEV[0], TEST[1] + 1):
            s = summary(flat_bets(g[g.date.dt.year == y], pc, price, rule, t), boot=False)
            per.append({"price": price, "model": pc, "strategy": f"{rule}>{t:.2f}" if rule == "value" else rule, "year": y, **s})
    rep["per_year"] = per
    json.dump(rep, open(RES / "betting_report.json", "w"), indent=1)
    pd.concat(all_bets).to_csv(RES / "betting_bets_test.csv", index=False)

    def line(r, k):
        s = r[k]
        if not s.get("bets"):
            return "      0 bets".ljust(58)
        ci = s.get("roi_ci95", [np.nan, np.nan])
        return f"{s['bets']:5d} bets ROI {s['roi']:+7.2%} [{ci[0]:+.1%},{ci[1]:+.1%}] fair {s['fair_roi']:+6.2%}"
    print(f"\n{'price':11s} {'model':8s} {'strategy':24s} | {'DEV 2016-23':58s} | TEST 2024-26")
    for r in out + kel:
        extra = f"  bank x{r['test']['bankroll_x']:.2f} dd {r['test']['max_drawdown']:.0%} (dev x{r['dev']['bankroll_x']:.2f})" if "bankroll_x" in r["test"] else ""
        print(f"{r['price']:11s} {r['model']:8s} {r['strategy']:24s} | {line(r, 'dev')} | {line(r, 'test')}{extra}")


if __name__ == "__main__":
    main()
