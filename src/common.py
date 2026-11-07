"""Shared data loading / symmetric design-matrix construction."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

TRAIN_END = pd.Timestamp("2022-01-01")
VALID_END = pd.Timestamp("2024-01-01")
CORNER_RELIABLE_FROM = pd.Timestamp("2010-01-01")  # before this ufcstats lists the winner first

EXCLUDE_FIGHTER = {"stance", "imp", "dec_imp", "sub_imp", "ko_imp", "rank", "p4p"}
STANCES = ["Orthodox", "Southpaw", "Switch"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load():
    d = pd.read_parquet(PROC / "features.parquet")
    d = d[d.winner >= 0].copy()
    d["y"] = (d.winner == 1).astype(int)
    return d.reset_index(drop=True)


def fighter_cols(d):
    r = {c[2:] for c in d.columns if c.startswith("r_")}
    b = {c[2:] for c in d.columns if c.startswith("b_")}
    cols = sorted((r & b) - EXCLUDE_FIGHTER - {"id", "name"})
    return [c for c in cols if pd.api.types.is_numeric_dtype(d["r_" + c])]


PAIR_P = ["p_elo32", "p_elo64", "p_elo_new", "p_elo_mov", "p_elo_str", "p_elo_ctrl", "p_elo_kd", "p_glicko", "p_welo", "p_welo_mov"]


def side_frame(d, a, b, fcols, flip):
    """Build features from the perspective of corner a vs corner b."""
    X = {}
    for c in fcols:
        xa, xb = d[f"{a}_{c}"].astype(float), d[f"{b}_{c}"].astype(float)
        X[f"A_{c}"] = xa.values
        X[f"B_{c}"] = xb.values
        X[f"D_{c}"] = (xa - xb).values
    for s in STANCES:
        X[f"A_st_{s}"] = (d[f"{a}_stance"] == s).astype(float).values
        X[f"B_st_{s}"] = (d[f"{b}_stance"] == s).astype(float).values
    X["southpaw_vs_orth"] = ((d[f"{a}_stance"] == "Southpaw") & (d[f"{b}_stance"] == "Orthodox")).astype(float).values - \
                            ((d[f"{b}_stance"] == "Southpaw") & (d[f"{a}_stance"] == "Orthodox")).astype(float).values
    # explicit style-matchup terms: what A tends to do x what B tends to allow (and vice versa)
    g = lambda s, c: d[f"{s}_{c}"].astype(float).values
    for name, off, de in (("sig", "c_slpm", "c_sapm"), ("sig_e", "e_slpm", "e_sapm"), ("head", "c_head_pm", "c_head_abs_pm"),
                          ("td", "c_td15", "c_td_abs15"), ("ctrl", "c_ctrl", "c_ctrl_abs"), ("kd", "c_kd15", "c_kd_abs15"),
                          ("sub", "c_sub15", "c_sub_abs15"), ("ground", "c_ground_pm", "c_ground_abs_pm")):
        ma = g(a, off) * g(b, de)
        mb = g(b, off) * g(a, de)
        X[f"M_{name}"] = np.log((ma + 1e-3) / (mb + 1e-3))
    X["M_tdacc_vs_def"] = g(a, "c_tdacc") * (1 - g(b, "c_tddef")) - g(b, "c_tdacc") * (1 - g(a, "c_tddef"))
    X["M_sacc_vs_def"] = g(a, "c_sacc") * (1 - g(b, "c_sdef")) - g(b, "c_sacc") * (1 - g(a, "c_sdef"))
    X["M_adj"] = (g(a, "c_adj_sl") - g(b, "c_adj_sa")) - (g(b, "c_adj_sl") - g(a, "c_adj_sa"))
    X["M_ko_threat"] = g(a, "c_kd15") * g(b, "ko_loss_rate") - g(b, "c_kd15") * g(a, "ko_loss_rate")
    X["M_sub_threat"] = g(a, "c_sub15") * g(b, "pro_subl_rate") - g(b, "c_sub15") * g(a, "pro_subl_rate")
    for p in PAIR_P:
        v = d[p].values
        X[p] = (1 - v) if flip else v
        X["lg_" + p] = logit(X[p])
    # market
    def pr(col):
        v = d[col].values.astype(float)
        return (1 - v) if flip else v
    X["mkt_close"] = pr("mkt_close")
    X["lg_mkt_close"] = logit(X["mkt_close"])
    X["mkt_open"] = pr("mkt_open")
    X["lg_mkt_open"] = logit(X["mkt_open"])
    X["line_move"] = X["lg_mkt_close"] - X["lg_mkt_open"]
    X["close_spread"] = d["n_close_spread"].values
    X["overround"] = d["n_close_overround"].fillna(d["overround"]).values
    for k in ("dec", "sub", "ko"):
        X[f"A_{k}_imp"] = d[f"{a}_{k}_imp"].values
        X[f"B_{k}_imp"] = d[f"{b}_{k}_imp"].values
    X["A_rank"] = d[f"rank_{a}"].values
    X["B_rank"] = d[f"rank_{b}"].values
    X["D_rank"] = X["A_rank"] - X["B_rank"]
    X["A_p4p"] = d[f"p4p_{a}"].values
    X["B_p4p"] = d[f"p4p_{b}"].values
    # bout
    X["corner"] = np.where(d.date >= CORNER_RELIABLE_FROM, -1.0 if flip else 1.0, 0.0)
    X["sched_rounds"] = d.sched_rounds.values
    X["card_pos"] = d.card_pos.values
    X["title"] = d.title.astype(float).values
    X["women"] = d.women.astype(float).values
    X["wc_limit"] = d.wc_limit.values
    X["year"] = d.date.dt.year.values + d.date.dt.dayofyear.values / 366
    return pd.DataFrame(X)


def design(d, fcols=None, both=True):
    """Returns X (stacked orig + swapped if both), y, fight index."""
    fcols = fcols or fighter_cols(d)
    Xo = side_frame(d, "r", "b", fcols, flip=False)
    if not both:
        return Xo, d.y.values, np.arange(len(d))
    Xs = side_frame(d, "b", "r", fcols, flip=True)
    X = pd.concat([Xo, Xs], ignore_index=True)
    y = np.concatenate([d.y.values, 1 - d.y.values])
    idx = np.concatenate([np.arange(len(d)), np.arange(len(d))])
    return X, y, idx


def sym_predict(predict_fn, d, fcols=None):
    """P(red wins) averaged over both orientations."""
    fcols = fcols or fighter_cols(d)
    Xo = side_frame(d, "r", "b", fcols, flip=False)
    Xs = side_frame(d, "b", "r", fcols, flip=True)
    return 0.5 * (predict_fn(Xo) + 1 - predict_fn(Xs))


def metrics(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return {"acc": float(((p > 0.5) == (y == 1)).mean()), "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "brier": float(((p - y) ** 2).mean()), "n": int(len(y))}
