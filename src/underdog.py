"""Underdog specialist: models trained for PROFIT on underdog bets, instead of for predicting the winner.

Question: can a narrow model that only looks at underdogs, and is optimised for money rather than accuracy, find
profitable underdog bets that the general winner model (src/betting.py) misses?

For every fight the underdog is the side with the longer price at bet time (close_mid or open). One row per fight:
  R = won * odds - 1   profit of a 1-unit bet on the underdog at that price

Models (each refit every year on underdog rows from 2010 .. year-1, i.e. walk-forward):
  base       the general winner model (stacker) -> EV = p * odds - 1            (what src/betting.py does)
  mkt_bias   logistic regression of underdog wins on the no-vig price only      (favourite-longshot bias alone)
  lr_small   logistic regression on price + stats-model logit + disagreement + line move + margin
  lgb_resid  LightGBM boosted FROM the market price (learns where the market misprices underdogs), ~40 features
  lgb_profit LightGBM regression on R directly (expected profit, no probability in between)
  lgb_cost   LightGBM classifier, win weighted by (odds - 1), loss by 1  -> score > 0.5  <=>  EV > 0
  growth     linear stake policy  f = 5 % * sigmoid(w.x)  fitted to maximise mean log(1 + f * R)  (Kelly growth)

dev = 2016-2023 picks the model and the threshold (best ROI with >= MIN_BETS dev bets), test = 2024-2026.
Selection-bias benchmark ("null"): outcomes are re-drawn from the fair closing price (a world where the close is
exactly right, so no strategy at the close has any edge) and the whole fit-and-select procedure is rerun.

Needs data/processed/wf_stats_oof.parquet (src/evaluate.py).  Run: python src/underdog.py [n_null_sims]
"""
import json
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

import betting as B
from common import PROC, ROOT, load, logit

warnings.filterwarnings("ignore")
RES = ROOT / "results"
YEARS = list(range(B.DEV[0], B.TEST[1] + 1))
MIN_BETS = 150
EV_GRID = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20)
DIFF = ["age", "reach", "height", "days_since", "n_fights", "win_streak", "lose_streak", "last_res", "welo", "elo64", "glicko",
        "mk_surprise", "mk_surprise3", "mk_upset_wins", "mk_fav_losses", "mk_trend", "mk_last_imp", "mk_mean_imp", "mk_n", "debut",
        "pro_n", "pro_wr", "ko_loss_rate", "c_sdiff", "c_td15", "c_tddef", "finish_win_rate", "title_fights", "wc_change",
        "last_ko_loss", "fights_2y", "age_debut"]
SMALL = ["lg_q", "lg_odds", "s_u", "gap", "move", "move_na", "over"]
GROWTH = ["lg_q", "s_u", "gap", "move", "over"]
LGB = dict(learning_rate=0.02, num_leaves=7, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
           reg_lambda=10.0, n_estimators=300, verbose=-1, n_jobs=2)
sig = lambda z: 1 / (1 + np.exp(-z))


def frame():
    d = load().merge(pd.read_parquet(PROC / "wf_stats_oof.parquet")[["fight_id", "lg_ens"]], on="fight_id", how="left")
    d = B.walk_probs(d)
    g = d[d.date >= "2010-01-01"].merge(B.load_prices(), on="fight_id", how="inner")
    g = g[g.over_mid.between(1.0, 1.15) & g.mkt_close.notna() & g.lg_ens.notna()].reset_index(drop=True)
    g.loc[~g.over_open.between(1.0, 1.20), ["r_open", "b_open"]] = np.nan
    return g


def rows(g, fam):
    """One row per fight, oriented to the underdog at the bet-time price."""
    pr, pb = g[f"r_{fam}"].values, g[f"b_{fam}"].values
    ok = ~np.isnan(pr) & ~np.isnan(pb) & (pr != pb)
    g, pr, pb = g[ok].reset_index(drop=True), pr[ok], pb[ok]
    u = pr > pb  # red is the underdog
    sgn = np.where(u, 1.0, -1.0)
    o_u, o_f = np.where(u, pr, pb), np.where(u, pb, pr)
    x = pd.DataFrame({"fight_id": g.fight_id, "date": g.date, "year": g.date.dt.year, "odds": o_u})
    x["q"] = (1 / o_u) / (1 / o_u + 1 / o_f)
    x["lg_q"] = logit(x.q.values)
    x["lg_odds"] = np.log(o_u)
    x["over"] = 1 / o_u + 1 / o_f - 1
    x["s_u"] = sgn * g.lg_ens.values
    x["gap"] = x.s_u - x.lg_q
    if fam == "open":
        x["move"], x["move_na"] = 0.0, 1.0  # the line move is not known yet when betting the opener
    else:
        oo_u, oo_f = np.where(u, g.r_open, g.b_open), np.where(u, g.b_open, g.r_open)
        mv = x.lg_q - logit((1 / oo_u) / (1 / oo_u + 1 / oo_f))
        x["move_na"] = mv.isna().astype(float)
        x["move"] = mv.fillna(0.0)
    for c in DIFF:
        a, b = g[f"r_{c}"].astype(float).values, g[f"b_{c}"].astype(float).values
        x[f"D_{c}"] = sgn * (a - b)
    x["U_age"] = np.where(u, g.r_age, g.b_age)
    x["U_debut"] = np.where(u, g.r_debut, g.b_debut).astype(float)
    x["F_debut"] = np.where(u, g.b_debut, g.r_debut).astype(float)
    for c in ("women", "title", "sched_rounds", "wc_limit", "card_pos"):
        x[c] = g[c].astype(float).values
    won = np.where(u, g.y.values == 1, g.y.values == 0)
    x["won"] = won.astype(int)
    x["R"] = np.where(won, o_u - 1, -1.0)
    x["fair"] = np.where(u, g.mkt_close.values, 1 - g.mkt_close.values)  # no-vig closing prob of the underdog (CLV)
    pc = "p_open" if fam == "open" else "p_close"
    x["p_base"] = np.where(u, g[pc].values, 1 - g[pc].values)
    x["close_mid_u"] = np.where(u, g.r_close_mid, g.b_close_mid)
    x["src"] = g.src.values
    return x


FULL = None  # filled in main(): SMALL + D_* + bout columns


# ------------------------------------------------------------------------------------------------ models
def fit_predict(name, tr, te, won_col="won"):
    """Returns a score for te where higher = better bet, in EV units (score > t  <=>  bet)."""
    yw, R = tr[won_col].values, np.where(tr[won_col].values == 1, tr.odds.values - 1, -1.0)
    if name == "base":
        return te.p_base.values * te.odds.values - 1
    if name in ("mkt_bias", "lr_small"):
        cols = ["lg_q"] if name == "mkt_bias" else SMALL
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m = LogisticRegression(C=1.0 if name == "mkt_bias" else 0.1, max_iter=2000).fit((tr[cols] - mu) / sd, yw)
        return m.predict_proba((te[cols] - mu) / sd)[:, 1] * te.odds.values - 1
    if name == "lgb_resid":
        m = lgb.LGBMClassifier(objective="binary", **LGB).fit(tr[FULL], yw, init_score=tr.lg_q.values)
        return sig(m.predict(te[FULL], raw_score=True) + te.lg_q.values) * te.odds.values - 1
    if name == "lgb_profit":
        m = lgb.LGBMRegressor(objective="regression", **{**LGB, "min_child_samples": 150}).fit(tr[FULL], R)
        return m.predict(te[FULL])
    if name == "lgb_cost":
        w = np.where(yw == 1, tr.odds.values - 1, 1.0)
        m = lgb.LGBMClassifier(objective="binary", **LGB).fit(tr[FULL], yw, sample_weight=w)
        s = np.clip(m.predict_proba(te[FULL])[:, 1], 1e-6, 1 - 1e-6)
        return s / (1 - s) - 1  # monotone in the score, 0 at the EV = 0 boundary
    if name == "growth":
        mu, sd = tr[GROWTH].mean(), tr[GROWTH].std().replace(0, 1)
        X = np.c_[np.ones(len(tr)), ((tr[GROWTH] - mu) / sd).values]
        cap, lam = 0.05, 1e-3

        def obj(w):
            s = sig(X @ w)
            f = cap * s
            gain = 1 + f * R
            gr = -(X * (R * cap * s * (1 - s) / gain)[:, None]).mean(0) + 2 * lam * np.r_[0, w[1:]]
            return -np.log(gain).mean() + lam * (w[1:] ** 2).sum(), gr
        w = minimize(obj, np.r_[-3.0, np.zeros(len(GROWTH))], jac=True, method="L-BFGS-B").x
        Xt = np.c_[np.ones(len(te)), ((te[GROWTH] - mu) / sd).values]
        return cap * sig(Xt @ w)  # a stake (fraction of bankroll), not an EV
    raise ValueError(name)


MODELS = ["base", "mkt_bias", "lr_small", "lgb_resid", "lgb_profit", "lgb_cost", "growth"]
GROWTH_GRID = (0.0025, 0.005, 0.01, 0.02, 0.03)


def walk(x, name, won_col="won"):
    sc = pd.Series(np.nan, index=x.index)
    for y in YEARS:
        tr, te = x[x.year < y], x[x.year == y]
        if name == "base":
            tr = tr.iloc[:0]
        sc[te.index] = fit_predict(name, tr, te, won_col) if len(te) else []
    return sc


def bets(x, sc, name, t, won_col="won"):
    take = sc.notna() & (sc > t)
    if name != "growth":
        take &= sc <= B.MAX_EV  # same a-priori palpable-error rule as src/betting.py
    b = x[take]
    stake = sc[take].values if name == "growth" else np.ones(int(take.sum()))
    won = b[won_col].values == 1
    return pd.DataFrame({"fight_id": b.fight_id.values, "date": b.date.values, "odds": b.odds.values, "stake": stake, "won": won,
                         "fair": b.fair.values, "profit": np.where(won, stake * (b.odds.values - 1), -stake),
                         "close_mid_u": b.close_mid_u.values, "src": b.src.values})


def roi(b):
    return float(b.profit.sum() / b.stake.sum()) if len(b) and b.stake.sum() > 0 else np.nan


def select(x, sc, name, won_col="won"):
    dev = x.year.between(*B.DEV)
    grid = GROWTH_GRID if name == "growth" else EV_GRID
    cand = [(t, bets(x[dev], sc[dev], name, t, won_col)) for t in grid]
    ok = [(t, b) for t, b in cand if len(b) >= MIN_BETS]
    if not ok:
        return None, None
    return max(ok, key=lambda tb: roi(tb[1]))


# ------------------------------------------------------------------------------------------------ null
def null_one(x, seed, names):
    """World where the fair close is the truth: any dev ROI found here is selection luck."""
    r = np.random.default_rng(seed)
    z = x.copy()
    z["won_null"] = (r.random(len(z)) < z.fair.values).astype(int)
    out = {}
    for n in names:
        sc = walk(z, n, "won_null")
        t, bd = select(z, sc, n, "won_null")
        if t is None:
            continue
        test = z.year.between(*B.TEST)
        bt = bets(z[test], sc[test], n, t, "won_null")
        out[n] = (roi(bd), roi(bt))
    return out


def main():
    global FULL
    n_null = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    g = frame()
    rep = {}
    for fam in ("close_mid", "open"):
        x = rows(g, fam)
        FULL = SMALL + [c for c in x.columns if c.startswith(("D_", "U_", "F_"))] + ["women", "title", "sched_rounds", "wc_limit", "card_pos"]
        ev = x[x.year >= B.DEV[0]]
        dev, test = ev.year.between(*B.DEV), ev.year.between(*B.TEST)
        allu = bets(ev, pd.Series(0.1, index=ev.index), "x", 0.0)  # every underdog, flat
        print(f"\n=== {fam}: {len(x)} fights (train from 2010), every underdog: dev ROI {roi(allu[dev.values]):+.2%}, "
              f"test {roi(allu[test.values]):+.2%}, avg odds {allu.odds.mean():.2f}, win rate {allu.won.mean():.3f}")
        scores = dict(zip(MODELS, Parallel(n_jobs=len(MODELS))(delayed(walk)(x, n) for n in MODELS)))
        res = []
        for n in MODELS:
            sc = scores[n][ev.index]
            t, bd = select(ev, sc, n)
            if t is None:
                print(f"  {n:10s} no threshold with >= {MIN_BETS} dev bets")
                continue
            bt = bets(ev[test], sc[test], n, t)
            r = {"model": n, "threshold": t, "dev": B.summary(bd), "test": B.summary(bt)}
            for lab, b in (("dev", bd), ("test", bt)):
                gy = b.assign(yr=pd.to_datetime(b.date).dt.year).groupby("yr")[["profit", "stake"]].sum()
                yr = gy.profit / gy.stake
                r[f"{lab}_years_positive"] = f"{int((yr > 0).sum())}/{len(yr)}"
                top = b.sort_values("profit", ascending=False).iloc[10:]
                r[f"{lab}_roi_without_top10"] = round(roi(top), 4)
                r[f"{lab}_per_year"] = {int(k): round(float(v), 3) for k, v in yr.items()}
                if fam == "open":  # filled halfway from the opening price to the closing price
                    o = 1 / (1 / b.odds + 0.5 * (1 / b.close_mid_u - 1 / b.odds))
                    r[f"{lab}_roi_fill50"] = round(float((np.where(b.won, b.stake * (o - 1), -b.stake)).sum() / b.stake.sum()), 4)
            res.append(r)
            s, u = r["dev"], r["test"]
            if not u.get("bets"):
                print(f"  {n:10s} t={t:<6g} DEV {s['bets']:4d} bets ROI {s['roi']:+7.2%} | TEST 0 bets (stakes never above the dev-chosen minimum)")
                continue
            print(f"  {n:10s} t={t:<6g} DEV {s['bets']:4d} bets ROI {s['roi']:+7.2%} CI[{s['roi_ci95'][0]:+.0%},{s['roi_ci95'][1]:+.0%}] "
                  f"yrs+ {r['dev_years_positive']} | TEST {u['bets']:4d} bets ROI {u['roi']:+7.2%} CI[{u['roi_ci95'][0]:+.0%},"
                  f"{u['roi_ci95'][1]:+.0%}] P(loss) {u['p_profit_le_0']:.2f} CLV {u['fair_roi']:+.1%} odds {u['avg_odds']:.2f} "
                  f"w/o top10 {r['test_roi_without_top10']:+.1%}" + (f" fill50 {r['test_roi_fill50']:+.1%}" if fam == "open" else ""))
        best = max(res, key=lambda r: r["dev"]["roi"])
        print(f"  -> chosen on dev: {best['model']} (dev {best['dev']['roi']:+.2%}) -> TEST {best['test']['roi']:+.2%}")
        rep[fam] = {"every_underdog": {"dev": round(roi(allu[dev.values]), 4), "test": round(roi(allu[test.values]), 4)},
                    "models": res, "chosen_on_dev": best["model"]}
        if fam == "close_mid" and n_null:
            names = [n for n in MODELS if n != "base"]
            sims = Parallel(n_jobs=14)(delayed(null_one)(x.copy(), s, names) for s in range(n_null))
            best_dev = np.array([max(v[0] for v in s.values()) for s in sims])
            per = {n: np.array([s[n][0] for s in sims if n in s]) for n in names}
            real_best = max(r["dev"]["roi"] for r in res if r["model"] != "base")
            q = ev[dev.values]
            fair_ev = float((q.fair * q.odds - 1).mean())
            print(f"\n  NULL ({n_null} worlds where the fair close is right; true ROI of a random underdog bet {fair_ev:+.2%}):")
            for n in names:
                real = next((r["dev"]["roi"] for r in res if r["model"] == n), np.nan)
                print(f"    {n:10s} dev-selected ROI under null: median {np.median(per[n]):+.2%}, 95th pct {np.quantile(per[n], .95):+.2%}"
                      f"  | real {real:+.2%}  share of null worlds >= real {np.mean(per[n] >= real):.2f}")
            print(f"    best of all 6 models under null: median {np.median(best_dev):+.2%}, 95th pct {np.quantile(best_dev, .95):+.2%} | "
                  f"real best {real_best:+.2%}  share >= real {np.mean(best_dev >= real_best):.2f}")
            rep["null_close_mid"] = {"n": n_null, "true_roi_random_underdog_bet": round(fair_ev, 4),
                                     "best_of_models_dev_roi": {"median": round(float(np.median(best_dev)), 4),
                                                                "p95": round(float(np.quantile(best_dev, .95)), 4),
                                                                "share_ge_real": round(float(np.mean(best_dev >= real_best)), 3)},
                                     "per_model": {n: {"median": round(float(np.median(v)), 4), "p95": round(float(np.quantile(v, .95)), 4)}
                                                   for n, v in per.items()}}
    json.dump(rep, open(RES / "underdog_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
