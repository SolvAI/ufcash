"""Experiment 7 (DEV ONLY, 2016-2023): do fighter profiles or style matchups add information the market lacks?

  profile  Wikidata snapshot (data/raw/wikidata/sherdog_people.csv, matched by Sherdog id): citizenship region, born or
           citizen of the event's country, martial-arts background (wrestling / judo-sambo / striking / BJJ), education
  style    each fighter's pre-fight style vector (striking/grappling rates) -> 8 k-means styles (fit on pre-2016 rows)
           pair     shrunk mean residual vs the market of style A against style B in earlier fights
           personal shrunk mean residual of THIS fighter in earlier fights against opponents of the opponent's style

The Wikidata snapshot is written after the fact: whether a fighter has an entry (and how complete it is) reflects his
later career. Existence and completeness are therefore never used as features, and the profile test is run only on
fights where both fighters have an entry.

Each candidate is added to a no-intercept antisymmetric logistic regression on [logit(close), lg_ens], fit walk-forward
(train 2010 .. year-1). Reported: dev log-loss change with a bootstrap CI by event.
"""
import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from common import PROC, ROOT, load, logit

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
REGION = {"United States": "us", "Brazil": "br", "Japan": "jp", "Canada": "ca", "Mexico": "mx", "United Kingdom": "ukie", "Ireland": "ukie",
          "Australia": "aunz", "New Zealand": "aunz", "Russia": "cis", "Kazakhstan": "cis", "Kyrgyzstan": "cis", "Tajikistan": "cis",
          "Uzbekistan": "cis", "Azerbaijan": "cis", "Georgia": "cis", "Armenia": "cis", "Ukraine": "cis", "Belarus": "cis", "Moldova": "cis",
          "People's Republic of China": "eas", "China": "eas", "South Korea": "eas", "Mongolia": "eas"}
EVENT_CC = {"USA": "United States"}
STYLE = ["c_slpm", "c_sapm", "c_dist_share", "c_clinch_pm", "c_ground_share", "c_td15", "c_td_abs15", "c_ctrl", "c_ctrl_abs",
         "c_sub15", "c_kd15", "c_leg_share", "c_body_pm", "c_tdacc", "c_tddef", "c_sacc", "c_sdef"]
K, SHRINK = 8, 30.0


def profiles():
    c = sqlite3.connect(ROOT / "data" / "raw" / "canon" / "canon.db")
    al = pd.read_sql("select fighter_id, source, source_id from fighter_aliases where source in ('sherdog','ufcstats')", c)
    c.close()
    sh = al[al.source == "sherdog"].drop_duplicates("fighter_id").set_index("fighter_id").source_id
    uf = al[al.source == "ufcstats"][["fighter_id", "source_id"]].rename(columns={"source_id": "uid"})
    uf["sid"] = uf.fighter_id.map(sh)
    w = pd.read_csv(ROOT / "data" / "raw" / "wikidata" / "sherdog_people.csv", dtype={"sid": str}).drop_duplicates("sid")
    m = uf.merge(w, on="sid", how="inner").drop_duplicates("uid").set_index("uid")
    sp = m.sports.fillna("").str.lower()
    out = pd.DataFrame(index=m.index)
    out["region"] = m.citizenship.map(REGION).fillna("other")
    out["cit"], out["born"] = m.citizenship, m.birth_country
    out["wrest"] = sp.str.contains("wrestling") & ~sp.str.contains("professional wrestling") | sp.str.contains("greco|freestyle")
    out["grap"] = sp.str.contains("judo|sambo")
    out["strike"] = sp.str.contains("boxing|kickboxing|muay thai|karate|taekwondo|savate|sanda")
    out["bjj"] = sp.str.contains("jiu-jitsu|jiu jitsu")
    out["edu"] = m.n_edu > 0
    return out


def antisym_lr(tr, te, cols):
    """No intercept, both orientations: p(red) = sigmoid(w . x), x antisymmetric in the corners."""
    X = tr[cols].fillna(0).values
    m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000).fit(np.vstack([X, -X]), np.r_[tr.y, 1 - tr.y])
    return m.predict_proba(te[cols].fillna(0).values)[:, 1], dict(zip(cols, m.coef_[0].round(3)))


def style_feats(d):
    base = d[(d.date < "2016-01-01")]
    Z = pd.concat([base[[f"r_{c}" for c in STYLE]].set_axis(STYLE, axis=1)[base.r_n_stat_fights >= 3],
                   base[[f"b_{c}" for c in STYLE]].set_axis(STYLE, axis=1)[base.b_n_stat_fights >= 3]]).dropna()
    mu, sd = Z.mean(), Z.std()
    km = KMeans(K, n_init=10, random_state=0).fit((Z - mu) / sd)
    for s in "rb":
        X = d[[f"{s}_{c}" for c in STYLE]].set_axis(STYLE, axis=1)
        ok = X.notna().all(1) & (d[f"{s}_n_stat_fights"] >= 3)
        lab = np.full(len(d), -1)
        lab[ok.values] = km.predict(((X[ok] - mu) / sd).values)
        d[f"{s}_style"] = lab
    # residual vs market (red perspective); only fights with a closing line
    d = d.sort_values(["date", "order"] if "order" in d else "date").reset_index(drop=True)
    res = np.where(d.mkt_close.notna(), d.y - d.mkt_close, np.nan)
    # pair: as-of target encoding, using only earlier event dates
    pair_sum, pair_n = np.zeros((K, K)), np.zeros((K, K))
    fpair, fpers = np.zeros(len(d)), np.zeros(len(d))
    pers = {}  # (fighter, opp_style) -> [sum residual, n]
    for dt, idx in d.groupby("date", sort=True).indices.items():
        for i in idx:
            a, b = d.r_style.iat[i], d.b_style.iat[i]
            if a >= 0 and b >= 0:
                fpair[i] = pair_sum[a, b] / (pair_n[a, b] + SHRINK)
            ra, rb = d.r_id.iat[i], d.b_id.iat[i]
            sa, na = pers.get((ra, b), (0.0, 0))
            sb, nb = pers.get((rb, a), (0.0, 0))
            fpers[i] = sa / (na + 3.0) - sb / (nb + 3.0)
        for i in idx:  # update after the whole night (no same-night information)
            r = res[i]
            if np.isnan(r):
                continue
            a, b = d.r_style.iat[i], d.b_style.iat[i]
            if a >= 0 and b >= 0:
                pair_sum[a, b] += r; pair_n[a, b] += 1
                pair_sum[b, a] -= r; pair_n[b, a] += 1
            for f, opp_style, rr in ((d.r_id.iat[i], b, r), (d.b_id.iat[i], a, -r)):
                if opp_style >= 0:
                    s, n = pers.get((f, opp_style), (0.0, 0))
                    pers[(f, opp_style)] = (s + rr, n + 1)
    d["style_pair"], d["style_personal"] = fpair, fpers
    return d


def boot_delta(g, p0, p1):
    ll = lambda p: -(g.y * np.log(p) + (1 - g.y) * np.log(1 - p))
    diff = (ll(np.clip(p1, 1e-6, 1 - 1e-6)) - ll(np.clip(p0, 1e-6, 1 - 1e-6))).values
    ev = g.event.values
    u, inv = np.unique(ev, return_inverse=True)
    s, n = np.bincount(inv, diff), np.bincount(inv)
    idx = rng.integers(0, len(u), (2000, len(u)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return diff.mean(), np.quantile(b, [0.025, 0.975])


def main():
    d = load()
    d = d[d.date < "2024-01-01"].copy()  # dev only: the hold-out is not loaded
    d = d.merge(pd.read_parquet(PROC / "wf_stats_oof.parquet")[["fight_id", "lg_ens"]], on="fight_id", how="left")
    d = style_feats(d)
    pr = profiles()
    d["ev_country"] = d.location.str.split(",").str[-1].str.strip().replace(EVENT_CC)
    has = {}
    for s in "rb":
        p = pr.reindex(d[f"{s}_id"].values)
        has[s] = p.region.notna().values
        for reg in ("us", "br", "jp", "ca", "mx", "ukie", "aunz", "cis", "eas"):
            d[f"{s}_reg_{reg}"] = (p.region == reg).astype(float).values
        d[f"{s}_home"] = ((p.cit.values == d.ev_country.values) | (p.born.values == d.ev_country.values)).astype(float)
        for k in ("wrest", "grap", "strike", "bjj", "edu"):
            d[f"{s}_{k}"] = p[k].fillna(False).astype(float).values
    prof = []
    for k in ["reg_" + r for r in ("us", "br", "jp", "ca", "mx", "ukie", "aunz", "cis", "eas")] + ["home", "wrest", "grap", "strike", "bjj", "edu"]:
        d[f"D_{k}"] = d[f"r_{k}"] - d[f"b_{k}"]
        prof.append(f"D_{k}")
    d["both_wiki"] = has["r"] & has["b"]
    d["lg_mkt"] = logit(d.mkt_close.values)
    d["corner"] = 1.0
    d = d[d.mkt_close.notna() & d.lg_ens.notna() & (d.date >= "2010-01-01")].reset_index(drop=True)
    base = ["lg_mkt", "lg_ens", "corner"]
    safe = [c for c in prof if c.startswith("D_reg") or c == "D_home"]
    sets = {"profile_safe (region, home)": (base + safe, True), "profile_all (+ background, education)": (base + prof, True),
            "style_pair": (base + ["style_pair"], False), "style_personal": (base + ["style_personal"], False),
            "style_both": (base + ["style_pair", "style_personal"], False)}
    print(f"dev fights with a closing line: {int((d.date >= '2016-01-01').sum())}, both fighters in Wikidata: "
          f"{int(((d.date >= '2016-01-01') & d.both_wiki).sum())}")
    for name, (cols, wiki_only) in sets.items():
        g = d[d.both_wiki] if wiki_only else d
        p0, p1 = pd.Series(np.nan, index=g.index), pd.Series(np.nan, index=g.index)
        coef = None
        for y in range(2016, 2024):
            tr, te = g[g.date.dt.year < y], g[g.date.dt.year == y]
            p0[te.index], _ = antisym_lr(tr, te, base)
            p1[te.index], coef = antisym_lr(tr, te, cols)
        ev = g[g.date >= "2016-01-01"]
        m, ci = boot_delta(ev, p0[ev.index], p1[ev.index])
        a0 = ((p0[ev.index] > .5) == (ev.y == 1)).mean()
        a1 = ((p1[ev.index] > .5) == (ev.y == 1)).mean()
        extra = {k: v for k, v in coef.items() if k not in base}
        print(f"  {name:40s} n={len(ev)}  log-loss change {m:+.5f} CI[{ci[0]:+.5f},{ci[1]:+.5f}]  acc {a0:.4f} -> {a1:.4f}  coefs(2023 fit) {extra}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------------------------ follow-up
CIS = {"Russia", "Kazakhstan", "Kyrgyzstan", "Tajikistan", "Uzbekistan", "Azerbaijan", "Georgia", "Armenia", "Ukraine", "Belarus", "Moldova"}


def origin_check():
    """The one profile signal with a sizeable coefficient on dev was 'CIS citizen'. The Wikidata version may be biased
    (entries for unsuccessful foreign fighters are rarer), so rebuild it without Wikidata: a fighter counts as CIS if
    >= 50 % of his (>= 3) earlier non-UFC pro fights were held in a CIS country (canon.db event locations, as of the
    fight date). Then: does betting the CIS side in CIS-vs-non-CIS fights beat the closing price?  Hypothesis fixed on
    dev, so the hold-out is looked at once."""
    import betting as B
    c = sqlite3.connect(ROOT / "data" / "raw" / "canon" / "canon.db")
    q = pd.read_sql("""select fp.fighter_id, e.date, e.location, coalesce(e.promotion, '') as promotion
                       from fight_participants fp join fights f using(fight_id) join events e using(event_id)
                       where e.date is not null""", c)
    al = pd.read_sql("select fighter_id, source_id as uid from fighter_aliases where source='ufcstats'", c)
    c.close()
    q = q[~q.promotion.str.startswith("UFC")].copy()
    q["date"] = pd.to_datetime(q.date, errors="coerce")
    q = q.dropna(subset=["date"])
    q["cis"] = q.location.fillna("").str.split(",").str[-1].str.strip().isin(CIS).astype(float)
    q["n"] = 1.0
    cum = q.groupby(["fighter_id", "date"])[["n", "cis"]].sum().groupby(level=0).cumsum().reset_index().sort_values("date")
    d = load().merge(B.load_prices(), on="fight_id", how="inner")
    d = d[d.mkt_close.notna() & (d.date >= "2010-01-01") & d.over_mid.between(1.0, 1.15)].reset_index(drop=True)
    fmap = al.drop_duplicates("uid").set_index("uid").fighter_id
    for s in "rb":
        z = pd.DataFrame({"i": d.index, "date": d.date, "fighter_id": d[f"{s}_id"].map(fmap)}).dropna().sort_values("date")
        z["fighter_id"] = z.fighter_id.astype(int)
        cum["fighter_id"] = cum.fighter_id.astype(int)
        m = pd.merge_asof(z, cum, on="date", by="fighter_id", allow_exact_matches=False).set_index("i")
        d[f"{s}_cis"] = ((m.n >= 3) & (m.cis / m.n >= 0.5)).reindex(d.index).fillna(False).values
    one = d[d.r_cis != d.b_cis].copy()
    red = one.r_cis.values
    one["won"] = np.where(red, one.y == 1, one.y == 0)
    one["fair"] = np.where(red, one.mkt_close, 1 - one.mkt_close)
    for price in ("close_mid", "open"):
        one[f"o_{price}"] = np.where(red, one[f"r_{price}"], one[f"b_{price}"])
    print(f"\nCIS-trained (by pre-UFC fight locations) vs non-CIS: {len(one)} fights 2010-2026")
    for lab, yrs in (("2010-15", (2010, 2015)), ("dev 2016-23", (2016, 2023)), ("test 2024-26", (2024, 2026))):
        g = one[one.date.dt.year.between(*yrs)]
        line = f"  {lab:13s} n={len(g):4d}  CIS side won {g.won.mean():.3f} vs fair close {g.fair.mean():.3f}"
        for price in ("close_mid", "open"):
            o = g[f"o_{price}"]
            ok = o.notna()
            prof = np.where(g.won[ok], o[ok] - 1, -1.0)
            se = prof.std() / np.sqrt(ok.sum())
            line += f" | bet CIS side @{price}: ROI {prof.mean():+.1%} (+/- {1.96 * se:.1%}, n={int(ok.sum())})"
        print(line)


if __name__ == "__main__":
    import sys
    if "origin" in sys.argv:
        origin_check()
