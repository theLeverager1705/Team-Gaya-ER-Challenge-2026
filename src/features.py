"""Pair features for the matcher. All features are language-agnostic similarities,
so the model can generalize to countries it never saw in training (e.g. France).
Country itself is deliberately NOT a feature.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def _jaccard(a, b):
    """Token-set Jaccard similarity of two space-separated strings."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return np.nan
    return len(sa & sb) / len(sa | sb)


def _string_sims(a, b, prefix):
    """A block of rapidfuzz similarities between two lists of strings (0-1 scale)."""
    out = {
        f"{prefix}_ratio": [fuzz.ratio(x, y) / 100 for x, y in zip(a, b)],
        f"{prefix}_partial": [fuzz.partial_ratio(x, y) / 100 for x, y in zip(a, b)],
        f"{prefix}_tsort": [fuzz.token_sort_ratio(x, y) / 100 for x, y in zip(a, b)],
        f"{prefix}_tset": [fuzz.token_set_ratio(x, y) / 100 for x, y in zip(a, b)],
        f"{prefix}_jw": [JaroWinkler.similarity(x, y) for x, y in zip(a, b)],
        f"{prefix}_jacc": [_jaccard(x, y) for x, y in zip(a, b)],
    }
    return pd.DataFrame(out)


def build_features(cand, s1, pool):
    """Join candidate pairs with both records and compute the feature matrix.

    Returns (pairs_df, feature_column_names). pairs_df keeps s1/cand ids.
    """
    cols = ["entity_id", "name_full", "name_core", "legal", "addr", "postal",
            "numbers", "acronym", "first_tok", "source"]
    L = s1[cols].add_prefix("l_")
    R = pool[cols].add_prefix("r_")
    df = (cand.merge(L, left_on="s1", right_on="l_entity_id")
              .merge(R, left_on="cand", right_on="r_entity_id")
              .reset_index(drop=True))

    feats = [
        _string_sims(df.l_name_core.tolist(), df.r_name_core.tolist(), "nc"),
        _string_sims(df.l_name_full.tolist(), df.r_name_full.tolist(), "nf"),
        _string_sims(df.l_addr.tolist(), df.r_addr.tolist(), "ad"),
    ]
    F = pd.concat([df[["blk_name_cos", "blk_full_cos", "n_retrievers"]]] + feats, axis=1)

    # exact / structural signals
    F["core_exact"] = (df.l_name_core == df.r_name_core).astype(int)
    F["legal_match"] = np.where((df.l_legal == "") | (df.r_legal == ""), 0.5,
                                (df.l_legal == df.r_legal).astype(float))
    F["acr_match"] = ((df.l_acronym != "") & ((df.l_acronym == df.r_name_core.str.replace(" ", ""))
                      | (df.r_acronym == df.l_name_core.str.replace(" ", "")))).astype(int)
    F["first_tok_eq"] = (df.l_first_tok == df.r_first_tok).astype(int)
    F["len_ratio"] = (np.minimum(df.l_name_core.str.len(), df.r_name_core.str.len())
                      / np.maximum(df.l_name_core.str.len(), df.r_name_core.str.len()).clip(lower=1))
    F["postal_state"] = np.select(
        [(df.l_postal == "") | (df.r_postal == ""), df.l_postal == df.r_postal],
        [0.5, 1.0], default=0.0)
    F["num_jacc"] = [_jaccard(a, b) for a, b in zip(df.l_numbers, df.r_numbers)]
    F["addr_missing"] = ((df.l_addr == "").astype(int) + (df.r_addr == "").astype(int))
    F["is_s3"] = (df.r_source == "S3").astype(int)

    # context features: how this pair compares with its competitors
    base = 0.5 * F["nc_tset"] + 0.3 * F["blk_name_cos"] + 0.2 * F["ad_tset"]
    F["ctx_base"] = base
    g1 = base.groupby(df.s1)
    F["ctx_rank_in_s1"] = g1.rank(ascending=False, method="min")
    F["ctx_gap_to_best"] = g1.transform("max") - base
    F["ctx_n_strong"] = (base > 0.8).groupby(df.s1).transform("sum")
    g2 = base.groupby(df.cand)
    F["ctx_rank_in_cand"] = g2.rank(ascending=False, method="min")
    F["ctx_gap_cand_best"] = g2.transform("max") - base

    pairs = df[["s1", "cand"]].copy()
    return pairs, F.astype(float), list(F.columns)
