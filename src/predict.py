"""Turning pair probabilities into final match lists (precision-first)."""
import itertools

import numpy as np

from evaluate import macro_f05


def decide(pairs, probs, T=0.6, alpha=0.8, one_to_one=True):
    """Select matches per Source 1 entity.

    1. keep pairs with p >= T
    2. keep only pairs with p >= alpha * (best p for that S1)
    3. one-to-one: each S2/S3 record goes to at most one S1 (its highest-p S1)
    S1 entities with nothing left get an empty list (singleton prediction).
    """
    df = pairs.copy()
    df["p"] = probs
    df = df[df.p >= T]
    if len(df):
        df = df[df.p >= alpha * df.groupby("s1").p.transform("max")]
    if one_to_one and len(df):
        df = df.sort_values("p", ascending=False).drop_duplicates("cand")
    df = df.sort_values(["s1", "p"], ascending=[True, False])
    return df.groupby("s1")["cand"].apply(list).to_dict()


def tune(pairs, oof, truth_map, Ts=None, alphas=None, one_to_one_opts=(True, False)):
    """Grid-search the decision rule on out-of-fold probabilities."""
    Ts = Ts if Ts is not None else np.round(np.arange(0.30, 0.95, 0.05), 2)
    alphas = alphas if alphas is not None else [0.5, 0.7, 0.8, 0.9, 1.0]
    best = None
    for T, a, o in itertools.product(Ts, alphas, one_to_one_opts):
        s = macro_f05(decide(pairs, oof, T, a, o), truth_map)
        if best is None or s > best["score"]:
            best = {"T": float(T), "alpha": float(a), "one_to_one": o, "score": s}
    return best
