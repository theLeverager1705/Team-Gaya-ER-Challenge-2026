"""Candidate generation (blocking).

Union of cheap retrievers, each run per country label (open set):
  1. char n-gram TF-IDF on the core name      -> top-K by cosine
  2. char n-gram TF-IDF on name + address     -> top-K by cosine
  3. exact postal code + same first name token
  4. acronym key (e.g. 'ibm' vs 'international business machines')

The output of `generate_candidates` is exactly the set the matcher scores, so it is
also what gets written to candidate_pairs.tsv.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _topk_sparse(A, B, k, chunk=512):
    """For each row of A, indices + cosine scores of the top-k rows of B.

    A and B are L2-normalized sparse TF-IDF matrices, so A @ B.T is cosine.
    Processed in chunks to keep memory bounded.
    """
    k = min(k, B.shape[0])
    idx_out, sc_out = [], []
    BT = B.T.tocsc()
    for start in range(0, A.shape[0], chunk):
        sims = (A[start:start + chunk] @ BT).toarray()
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        scores = np.take_along_axis(sims, part, axis=1)
        idx_out.append(part)
        sc_out.append(scores)
    return np.vstack(idx_out), np.vstack(sc_out)


def fit_vectorizers(texts_name, texts_full):
    """Fit char TF-IDF vectorizers on all names / name+address texts of a split."""
    v_name = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                             sublinear_tf=True, dtype=np.float32)
    v_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2,
                             sublinear_tf=True, dtype=np.float32, max_features=500_000)
    v_name.fit(texts_name)
    v_full.fit(texts_full)
    return v_name, v_full


def _key_pairs(s1, pool, cols):
    """Exact-key join on the given columns, ignoring empty keys."""
    a = s1[["entity_id"] + cols].rename(columns={"entity_id": "s1"})
    b = pool[["entity_id"] + cols].rename(columns={"entity_id": "cand"})
    for c in cols:
        a = a[a[c] != ""]
        b = b[b[c] != ""]
    return a.merge(b, on=cols)[["s1", "cand"]]


def generate_candidates(s1, pool, k_name=30, k_full=20, max_per_s1=60,
                        block_by_country=True):
    """Return a DataFrame (s1, cand, blk_name_cos, blk_full_cos) of candidate pairs.

    s1 / pool must already carry the normalized columns from normalize.py.
    If block_by_country is False, all records are compared regardless of country.
    """
    s1 = s1.copy()
    pool = pool.copy()
    s1["full_txt"] = s1["name_core"] + " | " + s1["addr"]
    pool["full_txt"] = pool["name_core"] + " | " + pool["addr"]
    v_name, v_full = fit_vectorizers(
        pd.concat([s1["name_core"], pool["name_core"]]),
        pd.concat([s1["full_txt"], pool["full_txt"]]),
    )

    groups = s1.groupby("ckey") if block_by_country else [("all", s1)]
    frames = []
    for ck, g1 in groups:
        g2 = pool[pool["ckey"] == ck] if block_by_country else pool
        if len(g2) == 0:
            continue
        for vec, col, k, tag in [(v_name, "name_core", k_name, "blk_name_cos"),
                                 (v_full, "full_txt", k_full, "blk_full_cos")]:
            A = vec.transform(g1[col])
            B = vec.transform(g2[col])
            idx, sc = _topk_sparse(A, B, k)
            frames.append(pd.DataFrame({
                "s1": np.repeat(g1["entity_id"].values, idx.shape[1]),
                "cand": g2["entity_id"].values[idx.ravel()],
                "score": sc.ravel(), "retriever": tag,
            }))

    # exact keys (country is part of the key when blocking by country)
    ctry = ["ckey"] if block_by_country else []
    for cols in (["postal", "first_tok"], ["acronym"]):
        kp = _key_pairs(s1, pool, cols + ctry)
        kp["score"], kp["retriever"] = 1.0, "key_" + "_".join(cols)
        frames.append(kp)

    cand = pd.concat(frames, ignore_index=True)
    cand = cand[cand["score"] > 0]
    wide = cand.pivot_table(index=["s1", "cand"], columns="retriever",
                            values="score", aggfunc="max").reset_index()
    for c in ["blk_name_cos", "blk_full_cos"]:
        if c not in wide:
            wide[c] = 0.0
    wide = wide.fillna(0.0)
    wide["n_retrievers"] = (wide.drop(columns=["s1", "cand"]) > 0).sum(axis=1)
    # keep the best max_per_s1 candidates per S1 by combined retrieval score
    wide["blk_best"] = wide[["blk_name_cos", "blk_full_cos"]].max(axis=1) + 0.1 * wide["n_retrievers"]
    wide = (wide.sort_values(["s1", "blk_best"], ascending=[True, False])
                .groupby("s1").head(max_per_s1).reset_index(drop=True))
    keep = ["s1", "cand", "blk_name_cos", "blk_full_cos", "n_retrievers"]
    return wide[keep]


def candidates_to_map(cand):
    """{s1: [cand ids]} for writing candidate_pairs.tsv."""
    return cand.groupby("s1")["cand"].apply(list).to_dict()
