"""
Pairwise similarity features for candidate pairs.

All string similarities are computed over whole arrays of pairs with
rapidfuzz.process.cpdist (C++, multithreaded), not per-pair Python calls;
there are tens of millions of pairs on the full test set.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEATURE_COLUMNS = [
    "blk_score", "blk_name", "blk_addr", "blk_rank", "n_cands",
    "blk_score_gap", "blk_score_rel",
    "name_ratio", "name_token_set", "name_token_sort", "name_partial", "name_jw",
    "name_token_set_gap", "name_either_empty",
    "addr_ratio", "addr_token_set", "addr_token_sort", "addr_token_set_gap",
    "num_token_set", "first_num_eq", "first_num_missing",
    "q_name_ntok", "c_name_ntok", "q_name_non_ascii", "c_name_non_ascii", "c_is_s3",
]


GROUP_COLUMNS = [
    "p1", "p1_rank", "p1_gap", "p1_max_other", "p1_sum_other", "n_p1_05",
    "same_src_p1_sum_other", "ref_p1", "ref_same_src",
    "ref_name_token_set", "ref_addr_token_set", "ref_num_token_set",
]
STAGE2_COLUMNS = FEATURE_COLUMNS + GROUP_COLUMNS


def _sim(a, b, scorer):
    """Element-wise similarity of two equal-length string arrays, all cores."""
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def compute_features(pairs: pd.DataFrame, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    """Build the FEATURE_COLUMNS matrix for candidate pairs (blocking scores,
    name/address/number similarities, script and source flags, and gaps to the
    entity's best candidate). pairs must contain every candidate of each
    Source 1 entity it covers, because the gap features are per entity."""
    i = pairs["s1_row"].to_numpy()
    j = pairs["c_row"].to_numpy()

    def take(df, col, idx):
        return df[col].to_numpy(dtype=object)[idx]

    f = pd.DataFrame({
        "blk_score": pairs["blk_score"].to_numpy(np.float32),
        "blk_name": pairs["blk_name"].to_numpy(np.float32),
        "blk_addr": pairs["blk_addr"].to_numpy(np.float32),
        "blk_rank": pairs["blk_rank"].to_numpy(np.float32),
    })

    qn, cn = take(s1, "name_core", i), take(s23, "name_core", j)
    f["name_ratio"] = _sim(qn, cn, fuzz.ratio)
    f["name_token_set"] = _sim(qn, cn, fuzz.token_set_ratio)
    f["name_token_sort"] = _sim(qn, cn, fuzz.token_sort_ratio)
    f["name_partial"] = _sim(qn, cn, fuzz.partial_ratio)
    f["name_jw"] = _sim(qn, cn, JaroWinkler.normalized_similarity)
    f["name_either_empty"] = ((qn == "") | (cn == "")).astype(np.float32)
    del qn, cn

    qa, ca = take(s1, "addr_norm", i), take(s23, "addr_norm", j)
    f["addr_ratio"] = _sim(qa, ca, fuzz.ratio)
    f["addr_token_set"] = _sim(qa, ca, fuzz.token_set_ratio)
    f["addr_token_sort"] = _sim(qa, ca, fuzz.token_sort_ratio)
    del qa, ca

    f["num_token_set"] = _sim(take(s1, "addr_nums", i), take(s23, "addr_nums", j), fuzz.token_set_ratio)
    q_first = s1["addr_first_num"].to_numpy()[i]
    c_first = s23["addr_first_num"].to_numpy()[j]
    f["first_num_missing"] = ((q_first < 0) | (c_first < 0)).astype(np.float32)
    f["first_num_eq"] = ((q_first == c_first) & (q_first >= 0)).astype(np.float32)

    f["q_name_ntok"] = s1["name_ntok"].to_numpy()[i].astype(np.float32)
    f["c_name_ntok"] = s23["name_ntok"].to_numpy()[j].astype(np.float32)
    f["q_name_non_ascii"] = s1["name_non_ascii"].to_numpy()[i]
    f["c_name_non_ascii"] = s23["name_non_ascii"].to_numpy()[j]
    f["c_is_s3"] = s23["is_s3"].to_numpy()[j].astype(np.float32)

    g = f.groupby(i, sort=False)
    f["n_cands"] = g["blk_score"].transform("size").astype(np.float32)
    best = g["blk_score"].transform("max")
    f["blk_score_gap"] = best - f["blk_score"]
    f["blk_score_rel"] = f["blk_score"] / best
    f["name_token_set_gap"] = g["name_token_set"].transform("max") - f["name_token_set"]
    f["addr_token_set_gap"] = g["addr_token_set"].transform("max") - f["addr_token_set"]

    return f[FEATURE_COLUMNS]


def add_group_features(f: pd.DataFrame, pairs: pd.DataFrame, p1: np.ndarray, s23: pd.DataFrame) -> pd.DataFrame:
    """Stage-2 features from the stage-1 probabilities of an entity's OTHER candidates.

    True matches of one business resemble each other, so each candidate is compared
    with its entity's best other candidate (the "reference"): a record with an
    unrelated trade name but the reference's address is likely the same business.
    Also adds where the candidate sits in its entity's probability distribution.
    Modifies f in place and returns it with STAGE2_COLUMNS.
    """
    i = pairs["s1_row"].to_numpy()
    j = pairs["c_row"].to_numpy()
    p1 = np.asarray(p1, dtype=np.float64)
    n = len(p1)

    order = np.lexsort((-p1, i))
    first = np.ones(n, dtype=bool)
    first[1:] = i[order][1:] != i[order][:-1]
    starts = np.flatnonzero(first)
    gid_sorted = np.cumsum(first) - 1
    gid = np.empty(n, dtype=np.int64)
    gid[order] = gid_sorted
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n) - starts[gid_sorted]
    size = np.diff(np.r_[starts, n])[gid]

    top = order[starts]                                    # best candidate per group
    second = order[np.minimum(starts + 1, n - 1)]          # second best (valid if size > 1)
    ref = np.where(rank == 0, second[gid], top[gid])        # best OTHER candidate
    has_ref = size > 1

    is_s3 = s23["is_s3"].to_numpy()[j]
    sum_p = np.bincount(gid, weights=p1)
    sum_p_s3 = np.bincount(gid, weights=p1 * is_s3)
    same_src_sum = np.where(is_s3, sum_p_s3[gid], (sum_p - sum_p_s3)[gid])

    f["p1"] = p1.astype(np.float32)
    f["p1_rank"] = rank.astype(np.float32)
    f["p1_gap"] = (p1[top[gid]] - p1).astype(np.float32)
    f["p1_max_other"] = np.where(has_ref, p1[ref], 0.0).astype(np.float32)
    f["p1_sum_other"] = (sum_p[gid] - p1).astype(np.float32)
    f["n_p1_05"] = np.bincount(gid, weights=p1 >= 0.5)[gid].astype(np.float32)
    f["same_src_p1_sum_other"] = (same_src_sum - p1).astype(np.float32)
    f["ref_p1"] = np.where(has_ref, p1[ref], np.nan).astype(np.float32)
    f["ref_same_src"] = np.where(has_ref, is_s3 == is_s3[ref], np.nan).astype(np.float32)

    j_ref = j[ref]
    for col, src in (("ref_name_token_set", "name_core"), ("ref_addr_token_set", "addr_norm"),
                     ("ref_num_token_set", "addr_nums")):
        vals = s23[src].to_numpy(dtype=object)
        sim = _sim(vals[j], vals[j_ref], fuzz.token_set_ratio)
        f[col] = np.where(has_ref, sim, np.nan).astype(np.float32)

    return f[STAGE2_COLUMNS]
