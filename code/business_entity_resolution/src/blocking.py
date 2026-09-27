"""
Candidate generation: IDF-weighted rare-token retrieval, partitioned by country.

Each record is turned into hashed keys: core-name tokens, core-name bigrams,
address tokens and address bigrams. Keys that occur in more than `df_cap`
Source 2/3 records are dropped as non-discriminative. A Source 1 record's
score against a Source 2/3 record is the summed IDF of the keys they share,
computed for a whole chunk of Source 1 rows at once as a sparse matrix product,
and only the top-K scoring records per Source 1 entity are kept.

Address keys matter as much as name keys: many Source 2 names are written in
a native script (Bengali, Devanagari) or are an unrelated trade name, and the
address is then the only shared signal.

Countries are an open set: partitions are whatever labels appear in Source 1.
"""
import gc
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

from text import addr_keys, name_keys

N_FEATURES = 2 ** 24


def _hash(texts, analyzer) -> sp.csr_matrix:
    """Binary record x hashed-key matrix (deterministic murmur hashing, no vocabulary in memory)."""
    vec = HashingVectorizer(analyzer=analyzer, n_features=N_FEATURES, alternate_sign=False,
                            norm=None, binary=True, dtype=np.float32)
    return vec.transform(texts).tocsr()


def _topk_per_row(C: sp.csr_matrix, k: int):
    """Vectorized top-k per row of a sparse score matrix: (rows, cols, scores, ranks)."""
    C.sum_duplicates()
    counts = np.diff(C.indptr)
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), counts)
    order = np.lexsort((-C.data, rows))
    rank = np.arange(len(order)) - np.repeat(C.indptr[:-1], counts)
    mask = rank < k
    sel = order[mask]
    return rows[sel], C.indices[sel], C.data[sel], rank[mask]


class CandidateGenerator:
    """Retrieves the top_k Source 2/3 records per Source 1 entity by shared-rare-key IDF score."""

    def __init__(self, top_k: int = 30, df_cap: int = 1000, chunk_size: int = 5000,
                 verbose: bool = True):
        self.top_k = top_k
        self.df_cap = df_cap
        self.chunk_size = chunk_size
        self.verbose = verbose

    def _index(self, texts, analyzer):
        """Index Source 2/3 texts: returns (key x record matrix, per-key IDF); keys above df_cap are dropped."""
        B = _hash(texts, analyzer)
        df = np.bincount(B.indices, minlength=N_FEATURES)
        keep = (df > 0) & (df <= self.df_cap)
        idf = np.zeros(N_FEATURES, dtype=np.float32)
        idf[keep] = np.log1p(B.shape[0] / df[keep])
        B.data = keep[B.indices].astype(np.float32)
        B.eliminate_zeros()
        return B.T.tocsr(), idf

    @staticmethod
    def _query(texts, analyzer, idf):
        """Source 1 record x key matrix weighted by the corpus IDF (unknown/common keys get 0)."""
        A = _hash(texts, analyzer)
        A.data = idf[A.indices]
        A.eliminate_zeros()
        return A

    def generate(self, s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
        """Returns one row per candidate pair: s1_row / c_row are positional
        indices into s1 / s23, plus the blocking scores (reused as features)."""
        parts = []
        s1_country = np.asarray(s1["country"].astype(str), dtype=object)
        s23_country = np.asarray(s23["country"].astype(str), dtype=object)

        for country in np.unique(s1_country):
            s1_idx = np.flatnonzero(s1_country == country)
            s23_idx = np.flatnonzero(s23_country == country)
            if len(s23_idx) == 0:
                continue
            t0 = time.time()
            BnT, idf_n = self._index(s23["name_core"].to_numpy()[s23_idx], name_keys)
            BaT, idf_a = self._index(s23["addr_norm"].to_numpy()[s23_idx], addr_keys)
            An = self._query(s1["name_core"].to_numpy()[s1_idx], name_keys, idf_n)
            Aa = self._query(s1["addr_norm"].to_numpy()[s1_idx], addr_keys, idf_a)
            if self.verbose:
                print(f"  [{country}] {len(s1_idx):,} source1 vs {len(s23_idx):,} source2/3 "
                      f"records, indexed in {time.time() - t0:.0f}s")

            t0 = time.time()
            n_pairs = 0
            for start in range(0, len(s1_idx), self.chunk_size):
                stop = start + self.chunk_size
                Cn = (An[start:stop] @ BnT).tocsr()
                C = (Cn + (Aa[start:stop] @ BaT)).tocsr()
                rows, cols, score, rank = _topk_per_row(C, self.top_k)
                name_score = np.asarray(Cn[rows, cols]).ravel().astype(np.float32)
                parts.append(pd.DataFrame({
                    "s1_row": s1_idx[start + rows].astype(np.int32),
                    "c_row": s23_idx[cols].astype(np.int32),
                    "blk_score": score.astype(np.float32),
                    "blk_name": name_score,
                    "blk_addr": (score - name_score).astype(np.float32),
                    "blk_rank": rank.astype(np.int16),
                }))
                n_pairs += len(rows)
                del Cn, C
            if self.verbose:
                print(f"  [{country}] {n_pairs:,} candidate pairs in {time.time() - t0:.0f}s", flush=True)
            # free this country's index before the next one is built (peak memory)
            del BnT, BaT, An, Aa
            gc.collect()

        if not parts:
            return pd.DataFrame(columns=["s1_row", "c_row", "blk_score", "blk_name",
                                         "blk_addr", "blk_rank"])
        return pd.concat(parts, ignore_index=True)
