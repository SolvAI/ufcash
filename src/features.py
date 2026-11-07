"""Leakage-free pre-fight features.

Fights are processed date by date: features for every bout on date D are computed from the
state built on fights strictly before D, and only then is the state updated with D's results.

Output: data/processed/features.parquet — one row per bout, columns r_*, b_* (per corner) + bout info.
"""
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = Path(os.environ.get("UFC_PROC", ROOT / "data" / "processed"))  # overridable for the leakage audit

STATS = ["sig_str_l", "sig_str_a", "total_str_l", "total_str_a", "td_l", "td_a", "head_l", "head_a", "body_l",
         "body_a", "leg_l", "leg_a", "distance_l", "distance_a", "clinch_l", "clinch_a", "ground_l", "ground_a",
         "kd", "sub_att", "rev", "ctrl", "r1_sig_str_l", "r1_head_l", "late_sig_str_l", "late_head_l"]


def method_cat(m):
    m = str(m)
    if m.startswith("KO/TKO") or "Doctor" in m:
        return "ko"
    if m.startswith("Submission"):
        return "sub"
    if "Unanimous" in m:
        return "udec"
    if "Split" in m or "Majority" in m:
        return "sdec"
    return "other"


# ---------------------------------------------------------------- rating systems
class Elo:
    def __init__(self, k=32.0, init=1500.0, k_new=None, n_new=0):
        self.k, self.init, self.k_new, self.n_new = k, init, k_new or k, n_new
        self.r = defaultdict(lambda: init)
        self.n = defaultdict(int)

    def expect(self, a, b):
        return 1.0 / (1.0 + 10 ** ((self.r[b] - self.r[a]) / 400.0))

    def kk(self, a):
        return self.k_new if self.n[a] < self.n_new else self.k

    def update(self, a, b, score):
        """score in [0,1] from a's perspective."""
        e = self.expect(a, b)
        da, db = self.kk(a) * (score - e), self.kk(b) * ((1 - score) - (1 - e))
        return da, db

    def apply(self, deltas):
        for f, d in deltas:
            self.r[f] += d
            self.n[f] += 1


class Glicko:
    """Glicko-1 with rating-period = one event date; RD inflates with inactivity (days)."""
    q = math.log(10) / 400

    def __init__(self, init=1500.0, rd0=300.0, c=60.0, rd_min=40.0):
        self.r = defaultdict(lambda: init)
        self.rd = defaultdict(lambda: rd0)
        self.last = {}
        self.rd0, self.c, self.rd_min = rd0, c, rd_min

    def cur_rd(self, f, date):
        if f not in self.last:
            return self.rd0
        t = (date - self.last[f]).days / 365.25
        return min(math.sqrt(self.rd[f] ** 2 + (self.c ** 2) * t), self.rd0)

    @staticmethod
    def g(rd):
        return 1 / math.sqrt(1 + 3 * (Glicko.q ** 2) * (rd ** 2) / math.pi ** 2)

    def expect(self, a, b, date):
        rda, rdb = self.cur_rd(a, date), self.cur_rd(b, date)
        g = self.g(math.sqrt(rda ** 2 + rdb ** 2))
        return 1 / (1 + 10 ** (-g * (self.r[a] - self.r[b]) / 400))

    def update(self, a, b, score, date):
        rda, rdb = self.cur_rd(a, date), self.cur_rd(b, date)
        out = []
        for me, op, s, rdm, rdo in ((a, b, score, rda, rdb), (b, a, 1 - score, rdb, rda)):
            g = self.g(rdo)
            e = 1 / (1 + 10 ** (-g * (self.r[me] - self.r[op]) / 400))
            d2 = 1 / (self.q ** 2 * g ** 2 * e * (1 - e))
            denom = 1 / rdm ** 2 + 1 / d2
            nr = self.r[me] + self.q / denom * g * (s - e)
            nrd = max(math.sqrt(1 / denom), self.rd_min)
            out.append((me, nr, nrd))
        return out

    def apply(self, ups, date):
        for f, nr, nrd in ups:
            self.r[f], self.rd[f], self.last[f] = nr, nrd, date


# ---------------------------------------------------------------- per-fighter history features
PRIOR_MIN = 15.0  # pseudo-minutes for shrinking per-minute rates


def rate(num, den, prior_rate, k=PRIOR_MIN):
    return (num + prior_rate * k) / (den + k)


def ratio(num, den, prior, k):
    return (num + prior * k) / (den + k)


def fighter_feats(h, date, pri, wc_limit):
    """h: list of past fight dicts (chronological). pri: population priors."""
    f = {}
    n = len(h)
    f["n_fights"] = n
    if n == 0:
        f.update({"debut": 1})
    else:
        f["debut"] = 0
    res = [x["res"] for x in h]  # 1 win, 0 loss, 0.5 draw, None nc
    valid = [r for r in res if r is not None]
    w = sum(1 for r in valid if r == 1)
    l = sum(1 for r in valid if r == 0)
    f["wins"], f["losses"] = w, l
    f["win_rate"] = (w + 1) / (w + l + 2)
    for m in ("ko", "sub", "udec", "sdec"):
        f[f"win_{m}"] = sum(1 for x in h if x["res"] == 1 and x["mc"] == m)
        f[f"loss_{m}"] = sum(1 for x in h if x["res"] == 0 and x["mc"] == m)
    f["finish_win_rate"] = (f["win_ko"] + f["win_sub"] + 0.5) / (n + 1.5)
    f["finished_rate"] = (f["loss_ko"] + f["loss_sub"] + 0.3) / (n + 1.5)
    f["ko_loss_rate"] = (f["loss_ko"] + 0.2) / (n + 1.5)
    f["dec_rate"] = (sum(1 for x in h if x["mc"] in ("udec", "sdec")) + 0.5) / (n + 1)
    # streaks
    ws = ls = 0
    for r in reversed(valid):
        if r == 1 and ls == 0:
            ws += 1
        elif r == 0 and ws == 0:
            ls += 1
        else:
            break
    f["win_streak"], f["lose_streak"] = ws, ls
    for k in (3, 5):
        last = valid[-k:]
        f[f"last{k}_wr"] = (sum(1 for r in last if r == 1) + 0.5) / (len(last) + 1)
    f["last_res"] = valid[-1] if valid else 0.5
    f["last_finished"] = int(bool(h) and h[-1]["res"] == 0 and h[-1]["mc"] in ("ko", "sub"))
    f["last_ko_loss"] = int(bool(h) and h[-1]["res"] == 0 and h[-1]["mc"] == "ko")
    ko_losses_2y = sum(1 for x in h if x["res"] == 0 and x["mc"] == "ko" and (date - x["date"]).days < 730)
    f["ko_losses_2y"] = ko_losses_2y
    # time
    mins = sum(x["dur"] for x in h) / 60.0
    f["total_min"] = mins
    f["avg_min"] = mins / n if n else np.nan
    f["title_fights"] = sum(1 for x in h if x["title"])
    f["title_wins"] = sum(1 for x in h if x["title"] and x["res"] == 1)
    f["five_rd"] = sum(1 for x in h if x["sched"] == 5)
    f["days_since"] = (date - h[-1]["date"]).days if n else np.nan
    f["days_since_debut"] = (date - h[0]["date"]).days if n else 0
    f["fights_per_year"] = n / max(f["days_since_debut"] / 365.25, 0.5) if n else 0
    f["fights_2y"] = sum(1 for x in h if (date - x["date"]).days < 730)
    prev_wc = [x["wc"] for x in h if not np.isnan(x["wc"])]
    f["wc_change"] = (wc_limit - prev_wc[-1]) if (prev_wc and not np.isnan(wc_limit)) else 0.0
    f["wc_change_mean"] = (wc_limit - np.mean(prev_wc)) if (prev_wc and not np.isnan(wc_limit)) else 0.0
    # opponents
    oe = [x["opp_elo"] for x in h]
    f["opp_elo_mean"] = np.mean(oe) if oe else np.nan
    f["opp_elo_last3"] = np.mean(oe[-3:]) if oe else np.nan
    wins_oe = [x["opp_elo"] for x in h if x["res"] == 1]
    f["beaten_elo_max"] = max(wins_oe) if wins_oe else np.nan
    f["beaten_elo_mean"] = np.mean(wins_oe) if wins_oe else np.nan
    f["opp_winrate_mean"] = np.mean([x["opp_wr"] for x in h]) if h else np.nan

    # striking / grappling stats (career, shrunk) and recency weighted
    hs = [x for x in h if x["st"] is not None and x["dur"] > 0]
    f["n_stat_fights"] = len(hs)

    def agg(weights):
        tot = defaultdict(float)
        for wgt, x in zip(weights, hs):
            for k, v in x["st"].items():
                tot[k] += wgt * v
            tot["min"] += wgt * x["dur"] / 60.0
        return tot

    for tag, weights in (("c", [1.0] * len(hs)), ("e", [0.6 ** (len(hs) - 1 - i) for i in range(len(hs))])):
        t = agg(weights)
        m = t["min"]
        kk = PRIOR_MIN if tag == "c" else PRIOR_MIN * 0.6
        g = lambda key: t.get(key, 0.0)
        f[f"{tag}_slpm"] = rate(g("sig_str_l"), m, pri["slpm"], kk)
        f[f"{tag}_sapm"] = rate(g("opp_sig_str_l"), m, pri["slpm"], kk)
        f[f"{tag}_sdiff"] = f[f"{tag}_slpm"] - f[f"{tag}_sapm"]
        f[f"{tag}_satt_pm"] = rate(g("sig_str_a"), m, pri["sapm_att"], kk)
        f[f"{tag}_sacc"] = ratio(g("sig_str_l"), g("sig_str_a"), pri["sacc"], 40)
        f[f"{tag}_sdef"] = 1 - ratio(g("opp_sig_str_l"), g("opp_sig_str_a"), pri["sacc"], 40)
        f[f"{tag}_tot_pm"] = rate(g("total_str_l"), m, pri["tot_pm"], kk)
        f[f"{tag}_tot_abs_pm"] = rate(g("opp_total_str_l"), m, pri["tot_pm"], kk)
        f[f"{tag}_head_pm"] = rate(g("head_l"), m, pri["head_pm"], kk)
        f[f"{tag}_head_abs_pm"] = rate(g("opp_head_l"), m, pri["head_pm"], kk)
        f[f"{tag}_body_pm"] = rate(g("body_l"), m, pri["body_pm"], kk)
        f[f"{tag}_leg_pm"] = rate(g("leg_l"), m, pri["leg_pm"], kk)
        f[f"{tag}_dist_pm"] = rate(g("distance_l"), m, pri["dist_pm"], kk)
        f[f"{tag}_clinch_pm"] = rate(g("clinch_l"), m, pri["clinch_pm"], kk)
        f[f"{tag}_ground_pm"] = rate(g("ground_l"), m, pri["ground_pm"], kk)
        f[f"{tag}_ground_abs_pm"] = rate(g("opp_ground_l"), m, pri["ground_pm"], kk)
        f[f"{tag}_td15"] = 15 * rate(g("td_l"), m, pri["td_pm"], kk)
        f[f"{tag}_tdatt15"] = 15 * rate(g("td_a"), m, pri["tda_pm"], kk)
        f[f"{tag}_tdacc"] = ratio(g("td_l"), g("td_a"), pri["tdacc"], 5)
        f[f"{tag}_tddef"] = 1 - ratio(g("opp_td_l"), g("opp_td_a"), pri["tdacc"], 5)
        f[f"{tag}_td_abs15"] = 15 * rate(g("opp_td_l"), m, pri["td_pm"], kk)
        f[f"{tag}_sub15"] = 15 * rate(g("sub_att"), m, pri["sub_pm"], kk)
        f[f"{tag}_sub_abs15"] = 15 * rate(g("opp_sub_att"), m, pri["sub_pm"], kk)
        f[f"{tag}_kd15"] = 15 * rate(g("kd"), m, pri["kd_pm"], kk)
        f[f"{tag}_kd_abs15"] = 15 * rate(g("opp_kd"), m, pri["kd_pm"], kk)
        f[f"{tag}_ctrl"] = ratio(g("ctrl"), 60 * m, pri["ctrl"], 60 * kk)
        f[f"{tag}_ctrl_abs"] = ratio(g("opp_ctrl"), 60 * m, pri["ctrl"], 60 * kk)
        f[f"{tag}_rev15"] = 15 * rate(g("rev"), m, pri["rev_pm"], kk)
        sl = g("sig_str_l") + 1e-9
        f[f"{tag}_head_share"] = ratio(g("head_l"), sl, pri["head_share"], 20)
        f[f"{tag}_leg_share"] = ratio(g("leg_l"), sl, pri["leg_share"], 20)
        f[f"{tag}_dist_share"] = ratio(g("distance_l"), sl, pri["dist_share"], 20)
        f[f"{tag}_ground_share"] = ratio(g("ground_l"), sl, pri["ground_share"], 20)
        f[f"{tag}_str_share"] = ratio(g("sig_str_l"), g("sig_str_l") + g("opp_sig_str_l"), 0.5, 30)
        f[f"{tag}_ctrl_share"] = ratio(g("ctrl"), g("ctrl") + g("opp_ctrl"), 0.5, 120)
    # opponent-adjusted performance (minute weighted, shrunk to 0), career and recency-weighted
    ha = [x for x in h if x.get("adj") is not None]
    for tag, wts in (("c", [1.0] * len(ha)), ("e", [0.6 ** (len(ha) - 1 - i) for i in range(len(ha))])):
        W = sum(w_ * x["adj"]["min"] for w_, x in zip(wts, ha))
        kk = 20.0 if tag == "c" else 12.0
        for key in ("sl", "sa", "td", "tda", "ctrl", "ctrla", "kd", "kda"):
            f[f"{tag}_adj_{key}"] = sum(w_ * x["adj"]["min"] * x["adj"][key] for w_, x in zip(wts, ha)) / (W + kk)
        f[f"{tag}_adj_net"] = f[f"{tag}_adj_sl"] - f[f"{tag}_adj_sa"]
    # round splits: early vs late output (cardio / fade), accumulated damage
    r1m = sum(min(x["dur"], 300) for x in hs) / 60.0
    latem = sum(max(x["dur"] - 600, 0) for x in hs if x["sched"] == 3) / 60.0 + sum(max(x["dur"] - 600, 0) for x in hs if x["sched"] == 5) / 60.0
    g1 = lambda k: sum(x["st"][k] for x in hs)
    f["r1_diff_pm"] = (g1("r1_sig_str_l") - g1("opp_r1_sig_str_l")) / (r1m + 10)
    f["late_diff_pm"] = (g1("late_sig_str_l") - g1("opp_late_sig_str_l")) / (latem + 10)
    f["r1_slpm"] = rate(g1("r1_sig_str_l"), r1m, pri["slpm"], 10)
    f["late_slpm"] = rate(g1("late_sig_str_l"), latem, pri["slpm"], 10)
    f["fade"] = f["late_slpm"] - f["r1_slpm"]
    f["late_min"] = latem
    f["cum_head_abs"] = g1("opp_head_l")
    f["cum_sig_abs"] = g1("opp_sig_str_l")
    f["cum_kd_abs"] = g1("opp_kd")
    f["cum_head_abs_2y"] = sum(x["st"]["opp_head_l"] for x in hs if (date - x["date"]).days < 730)
    # market history: how the betting market rated them before, and how they did relative to it
    hm = [x for x in h if x.get("imp") is not None and not np.isnan(x["imp"]) and x["res"] is not None]
    f["mk_n"] = len(hm)
    f["mk_mean_imp"] = (sum(x["imp"] for x in hm) + 0.5 * 2) / (len(hm) + 2)
    f["mk_last_imp"] = hm[-1]["imp"] if hm else 0.5
    f["mk_last3_imp"] = (sum(x["imp"] for x in hm[-3:]) + 0.5) / (len(hm[-3:]) + 1)
    f["mk_surprise"] = sum(x["res"] - x["imp"] for x in hm) / (len(hm) + 3)
    f["mk_surprise3"] = sum(x["res"] - x["imp"] for x in hm[-3:]) / (len(hm[-3:]) + 1.5)
    f["mk_upset_wins"] = sum(1 for x in hm if x["res"] == 1 and x["imp"] < 0.4)
    f["mk_fav_losses"] = sum(1 for x in hm if x["res"] == 0 and x["imp"] > 0.6)
    f["mk_trend"] = (hm[-1]["imp"] - hm[-2]["imp"]) if len(hm) >= 2 else 0.0
    # promotional standing: where the UFC puts them on cards
    cps = [x["card_pos"] for x in h if x.get("card_pos") is not None and not pd.isna(x["card_pos"])]
    f["card_pos_mean"] = (sum(cps) + 6 * 2) / (len(cps) + 2)
    f["card_pos_last"] = cps[-1] if cps else 8.0
    f["main_events"] = sum(1 for c in cps if c == 0)
    f["top3_events"] = sum(1 for c in cps if c <= 2)
    # last fight performance
    if hs:
        x = hs[-1]["st"]
        f["last_str_share"] = (x["sig_str_l"] + 5) / (x["sig_str_l"] + x["opp_sig_str_l"] + 10)
    else:
        f["last_str_share"] = np.nan
    return f


def compute_priors(ff):
    s = ff[ff.has_stats & (ff.duration > 0)]
    m = s.duration.sum() / 60.0
    p = {
        "slpm": s.sig_str_l.sum() / m, "sapm_att": s.sig_str_a.sum() / m, "sacc": s.sig_str_l.sum() / s.sig_str_a.sum(),
        "tot_pm": s.total_str_l.sum() / m, "head_pm": s.head_l.sum() / m, "body_pm": s.body_l.sum() / m,
        "leg_pm": s.leg_l.sum() / m, "dist_pm": s.distance_l.sum() / m, "clinch_pm": s.clinch_l.sum() / m,
        "ground_pm": s.ground_l.sum() / m, "td_pm": s.td_l.sum() / m, "tda_pm": s.td_a.sum() / m,
        "tdacc": s.td_l.sum() / s.td_a.sum(), "sub_pm": s.sub_att.sum() / m, "kd_pm": s.kd.sum() / m,
        "ctrl": s.ctrl.sum() / s.duration.sum(), "rev_pm": s.rev.sum() / m,
        "head_share": s.head_l.sum() / s.sig_str_l.sum(), "leg_share": s.leg_l.sum() / s.sig_str_l.sum(),
        "dist_share": s.distance_l.sum() / s.sig_str_l.sum(), "ground_share": s.ground_l.sum() / s.sig_str_l.sum(),
    }
    return p


def impute_physical(fighters, fights):
    """ufcstats leaves reach/DOB/stance blank mostly for fighters with short UFC careers, so missingness leaks the
    future (missing reach -> 20% win rate). Hide it: stochastic regression imputation, deterministic per fighter."""
    f = fighters.copy()
    rng = np.random.default_rng(12345)
    noise = pd.Series(rng.standard_normal(len(f)), index=f.index)
    noise2 = pd.Series(rng.standard_normal(len(f)), index=f.index)
    noise3 = pd.Series(rng.standard_normal(len(f)), index=f.index)
    # real heights/reaches are whole inches, so imputed values are rounded too (a fractional value would mark them)
    w = f.weight.where(f.weight.between(100, 300))
    okh = f.height.notna() & w.notna()
    bh, ah = np.polyfit(w[okh], f.height[okh], 1)
    sdh = np.std(f.height[okh] - (ah + bh * w[okh]))
    mh = f.height.isna()
    f.loc[mh, "height"] = (ah + bh * w[mh].fillna(w.median()) + sdh * noise3[mh]).round()
    ok = f.reach.notna() & f.height.notna()
    b, a = np.polyfit(f.height[ok], f.reach[ok], 1)
    sd = np.std(f.reach[ok] - (a + b * f.height[ok]))
    miss = f.reach.isna()
    f.loc[miss, "reach"] = (a + b * f.height[miss] + sd * noise[miss]).round()
    # DOB: draw an age-at-UFC-debut ~ N(mean, sd) of known fighters
    first = pd.concat([fights[["date", "r_id"]].rename(columns={"r_id": "fighter_id"}),
                       fights[["date", "b_id"]].rename(columns={"b_id": "fighter_id"})]).groupby("fighter_id").date.min()
    f["first"] = f.fighter_id.map(first)
    known = f.dob.notna() & f["first"].notna()
    age_debut = (f["first"][known] - f.dob[known]).dt.days / 365.25
    mu, sdv = age_debut.mean(), age_debut.std()
    md = f.dob.isna() & f["first"].notna()
    f.loc[md, "dob"] = f["first"][md] - pd.to_timedelta((mu + sdv * noise2[md]) * 365.25, unit="D")
    f["stance"] = f.stance.replace({"Unknown": "Orthodox", "Open Stance": "Switch", "Sideways": "Orthodox"})
    return f.set_index("fighter_id")


def main():
    fights = pd.read_parquet(PROC / "fights.parquet").sort_values("order").reset_index(drop=True)
    ff = pd.read_parquet(PROC / "fighter_fights.parquet")
    fighters = impute_physical(pd.read_parquet(PROC / "fighters.parquet").drop_duplicates("fighter_id"), fights)
    ff = ff.merge(fights[["fight_id", "duration"]], on="fight_id")
    # priors from an early window only (avoid using future info, ~ constants anyway)
    early = fights[fights.date < "2012-01-01"].fight_id
    pri = compute_priors(ff[ff.fight_id.isin(early)])
    stat_cols = STATS + ["opp_" + c for c in STATS]
    stat_map = {}
    for row in ff.itertuples(index=False):
        if row.has_stats:
            stat_map[(row.fight_id, row.corner)] = {c: (0.0 if pd.isna(getattr(row, c)) else float(getattr(row, c))) for c in stat_cols}

    hist = defaultdict(list)
    tot = defaultdict(lambda: defaultdict(float))  # running career totals (for opponent baselines)

    def base_rates(fid):
        t = tot[fid]
        m = t["min"]
        k = PRIOR_MIN
        return {"slpm": rate(t["sig_str_l"], m, pri["slpm"], k), "sapm": rate(t["opp_sig_str_l"], m, pri["slpm"], k),
                "td_pm": rate(t["td_l"], m, pri["td_pm"], k), "td_abs_pm": rate(t["opp_td_l"], m, pri["td_pm"], k),
                "ctrl": ratio(t["ctrl"], 60 * m, pri["ctrl"], 60 * k), "ctrl_abs": ratio(t["opp_ctrl"], 60 * m, pri["ctrl"], 60 * k),
                "kd_pm": rate(t["kd"], m, pri["kd_pm"], k), "kd_abs_pm": rate(t["opp_kd"], m, pri["kd_pm"], k)}

    elos = {
        "elo32": Elo(k=32), "elo64": Elo(k=64), "elo_new": Elo(k=40, k_new=90, n_new=3),
    }
    elo_mov = Elo(k=48)        # margin-of-victory scored
    elo_str = Elo(k=40)        # score = significant-strike share
    elo_ctrl = Elo(k=40)       # score = control/grappling share
    elo_kd = Elo(k=40)         # score = damage (kd + ko) share
    gl = Glicko()

    def mov_score(res, mc):
        if res is None:
            return None
        if res == 0.5:
            return 0.5
        s = {"ko": 1.0, "sub": 1.0, "udec": 0.85, "sdec": 0.65, "other": 0.75}[mc]
        return s if res == 1 else 1 - s

    rows = []
    fight_meta = fights.assign(mc=fights.method.map(method_cat))
    _o = pd.read_parquet(PROC / "odds.parquet")
    mkt_map = dict(zip(_o.fight_id, _o.mkt_close))
    for date, day in fight_meta.groupby("date", sort=True):
        # ---- features (pre-fight)
        for fr in day.itertuples(index=False):
            rec = {"fight_id": fr.fight_id}
            for side, fid, oid in (("r", fr.r_id, fr.b_id), ("b", fr.b_id, fr.r_id)):
                h = hist[fid]
                feats = fighter_feats(h, date, pri, fr.wc_limit)
                for name, e in elos.items():
                    feats[name] = e.r[fid]
                feats["elo_mov"] = elo_mov.r[fid]
                feats["elo_str"] = elo_str.r[fid]
                feats["elo_ctrl"] = elo_ctrl.r[fid]
                feats["elo_kd"] = elo_kd.r[fid]
                feats["glicko"] = gl.r[fid]
                feats["glicko_rd"] = gl.cur_rd(fid, date)
                feats["glicko_lo"] = gl.r[fid] - 2 * feats["glicko_rd"]
                # physical
                if fid in fighters.index:
                    p = fighters.loc[fid]
                    feats["height"], feats["reach"] = p.height, p.reach
                    feats["ape"] = p.reach - p.height if pd.notna(p.reach) and pd.notna(p.height) else np.nan
                    feats["age"] = (date - p.dob).days / 365.25 if pd.notna(p.dob) else np.nan
                    feats["stance"] = p.stance
                else:
                    feats.update({"height": np.nan, "reach": np.nan, "ape": np.nan, "age": np.nan, "stance": "Orthodox"})
                feats["age_debut"] = feats["age"] - feats["days_since_debut"] / 365.25 if pd.notna(feats["age"]) else np.nan
                for k, v in feats.items():
                    rec[f"{side}_{k}"] = v
            # expected scores from rating systems (red perspective)
            for name, e in list(elos.items()) + [("elo_mov", elo_mov), ("elo_str", elo_str), ("elo_ctrl", elo_ctrl), ("elo_kd", elo_kd)]:
                rec[f"p_{name}"] = e.expect(fr.r_id, fr.b_id)
            rec["p_glicko"] = gl.expect(fr.r_id, fr.b_id, date)
            rows.append(rec)

        # ---- update state with this date's results
        deltas = defaultdict(list)
        gl_ups = []
        new_hist = []
        for fr in day.itertuples(index=False):
            if fr.outcome == "UPCOMING":
                continue
            if fr.outcome == "W/L":
                rr = 1.0
            elif fr.outcome == "L/W":
                rr = 0.0
            elif fr.outcome == "D/D":
                rr = 0.5
            else:
                rr = None
            st_r = stat_map.get((fr.fight_id, "r"))
            st_b = stat_map.get((fr.fight_id, "b"))
            dur = fr.duration if pd.notna(fr.duration) else 0.0
            if rr is not None:
                for name, e in elos.items():
                    da, db = e.update(fr.r_id, fr.b_id, rr)
                    deltas[name] += [(fr.r_id, da), (fr.b_id, db)]
                ms = mov_score(rr, fr.mc)
                da, db = elo_mov.update(fr.r_id, fr.b_id, ms)
                deltas["elo_mov"] += [(fr.r_id, da), (fr.b_id, db)]
                gl_ups += gl.update(fr.r_id, fr.b_id, rr, date)
            if st_r is not None and dur > 0:
                s_str = (st_r["sig_str_l"] + 3) / (st_r["sig_str_l"] + st_b["sig_str_l"] + 6)
                da, db = elo_str.update(fr.r_id, fr.b_id, s_str)
                deltas["elo_str"] += [(fr.r_id, da), (fr.b_id, db)]
                g_r = st_r["ctrl"] / 60 + 2 * st_r["td_l"] + 1.5 * st_r["sub_att"]
                g_b = st_b["ctrl"] / 60 + 2 * st_b["td_l"] + 1.5 * st_b["sub_att"]
                s_ctrl = (g_r + 1) / (g_r + g_b + 2)
                da, db = elo_ctrl.update(fr.r_id, fr.b_id, s_ctrl)
                deltas["elo_ctrl"] += [(fr.r_id, da), (fr.b_id, db)]
                ko_r = 2.0 * (rr == 1.0 and fr.mc == "ko")
                ko_b = 2.0 * (rr == 0.0 and fr.mc == "ko")
                d_r = st_r["kd"] + ko_r + st_r["head_l"] / 40
                d_b = st_b["kd"] + ko_b + st_b["head_l"] / 40
                s_kd = (d_r + 0.5) / (d_r + d_b + 1)
                da, db = elo_kd.update(fr.r_id, fr.b_id, s_kd)
                deltas["elo_kd"] += [(fr.r_id, da), (fr.b_id, db)]
            for fid, oid, res, st in ((fr.r_id, fr.b_id, rr, st_r), (fr.b_id, fr.r_id, None if rr is None else 1 - rr, st_b)):
                opp_h = hist[oid]
                adj = None
                if st is not None and dur > 0:
                    ob = base_rates(oid)
                    mn = dur / 60.0
                    adj = {"sl": st["sig_str_l"] / mn - ob["sapm"], "sa": st["opp_sig_str_l"] / mn - ob["slpm"],
                           "td": 15 * (st["td_l"] / mn - ob["td_abs_pm"]), "tda": 15 * (st["opp_td_l"] / mn - ob["td_pm"]),
                           "ctrl": st["ctrl"] / dur - ob["ctrl_abs"], "ctrla": st["opp_ctrl"] / dur - ob["ctrl"],
                           "kd": 15 * (st["kd"] / mn - ob["kd_abs_pm"]), "kda": 15 * (st["opp_kd"] / mn - ob["kd_pm"]), "min": mn}
                ov = [x["res"] for x in opp_h if x["res"] is not None]
                new_hist.append((fid, {
                    "date": date, "res": res, "mc": fr.mc, "dur": dur, "sched": fr.sched_rounds, "title": bool(fr.title),
                    "wc": fr.wc_limit if pd.notna(fr.wc_limit) else np.nan, "opp": oid, "opp_elo": elos["elo32"].r[oid],
                    "opp_wr": (sum(1 for r in ov if r == 1) + 1) / (len(ov) + 2),
                    "st": st if (st is not None and dur > 0) else None, "adj": adj, "card_pos": fr.card_pos,
                    "imp": (mkt_map.get(fr.fight_id, np.nan) if fid == fr.r_id else 1 - mkt_map.get(fr.fight_id, np.nan)),
                }))
        for name, e in list(elos.items()) + [("elo_mov", elo_mov), ("elo_str", elo_str), ("elo_ctrl", elo_ctrl), ("elo_kd", elo_kd)]:
            e.apply(deltas[name])
        gl.apply(gl_ups, date)
        for fid, x in new_hist:
            hist[fid].append(x)
            if x["st"] is not None:
                t = tot[fid]
                for k2, v in x["st"].items():
                    t[k2] += v
                t["min"] += x["dur"] / 60.0

    feat = pd.DataFrame(rows)
    out = fights.merge(feat, on="fight_id")
    odds = pd.read_parquet(PROC / "odds.parquet")
    out = out.merge(odds.drop(columns=["odds_source"]), on="fight_id", how="left")
    cf = PROC / "canon_feats.parquet"
    if cf.exists():
        out = out.merge(pd.read_parquet(cf), on="fight_id", how="left")
    out.to_parquet(PROC / "features.parquet", index=False)
    print(out.shape)


if __name__ == "__main__":
    main()
