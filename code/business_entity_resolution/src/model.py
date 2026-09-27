"""
Pair classifier (LightGBM, MIT license) and the decision rule that turns pair
probabilities into per-entity match lists.

Decision rule, tuned on held-out entities to maximize macro F_0.5:
  - keep every candidate with probability >= t_all
  - additionally keep an entity's single best candidate if its probability >= t_top
Only 5.6% of training entities are singletons, and an empty prediction scores
0 for every other entity, so a lower bar for the top candidate pays off.
"""
import lightgbm as lgb
import numpy as np

from evaluate import per_entity_f05

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.1,
    "num_leaves": 127,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "deterministic": True,
    "verbose": -1,
}


def train_model(X_tr, y_tr, X_va, y_va, max_rounds: int = 1500) -> lgb.Booster:
    """Fit the pair classifier with early stopping on the validation pairs."""
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])


def apply_decision(rule: dict, entity, prob) -> np.ndarray:
    """Match mask for candidate pairs under a saved decision rule (see config.json)."""
    if rule["type"] == "expected_f":
        return decide_expected_f(entity, prob, rule["c_missing"], rule["s_empty"])
    return decide(prob, top_candidate_mask(entity, prob), rule["t_all"], rule["t_top"])


PREFILTER_GRID = (0.0, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1)


def prefilter_table(entity, p1, p2, label, n_true, rule):
    """For each stage-1 cutoff: (cutoff, candidates per entity, recall ceiling, macro F0.5)
    when pairs below the cutoff leave the candidate set and the final model never scores them."""
    out = []
    for tau in PREFILTER_GRID:
        keep = p1 >= tau
        pred = np.zeros(len(p1), dtype=bool)
        if keep.any():
            pred[keep] = apply_decision(rule, entity[keep], p2[keep])
        out.append({"p1_cutoff": tau, "candidates_per_entity": round(float(keep.sum() / len(n_true)), 3),
                    "recall_ceiling": round(float(label[keep].sum() / max(n_true.sum(), 1)), 4),
                    "macro_f05": round(macro_f05(entity, pred, label, n_true), 4)})
    return out


def choose_prefilter(tune_table, tolerance: float = 0.0001) -> float:
    """Largest cutoff whose tune-split macro F0.5 is within `tolerance` of using no cutoff,
    i.e. the smallest candidate set that costs no measurable accuracy."""
    base = tune_table[0]["macro_f05"]
    return max(r["p1_cutoff"] for r in tune_table if r["macro_f05"] >= base - tolerance)


def top_candidate_mask(entity: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """True for each entity's single highest-probability candidate pair."""
    order = np.lexsort((-prob, entity))
    sorted_entity = entity[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_entity[1:] != sorted_entity[:-1]
    mask = np.zeros(len(prob), dtype=bool)
    mask[order[first]] = True
    return mask


def decide(prob, is_top, t_all: float, t_top: float) -> np.ndarray:
    """Final match mask for candidate pairs under the two-threshold rule."""
    return (prob >= t_all) | (is_top & (prob >= t_top))


def macro_f05(entity, pred, label, n_true) -> float:
    """entity: dense codes 0..E-1 per pair; n_true: true-match counts for all E
    entities, including ones whose true matches never reached the candidate set."""
    E = len(n_true)
    n_pred = np.bincount(entity, weights=pred, minlength=E)
    tp = np.bincount(entity, weights=pred & label, minlength=E)
    return float(per_entity_f05(n_true, n_pred, tp).mean())


def tune_thresholds(entity, prob, label, n_true):
    """Grid-search (t_all, t_top) maximizing macro F_0.5; returns ((t_all, t_top), score)."""
    is_top = top_candidate_mask(entity, prob)
    best_score, best = -1.0, (0.5, 0.5)
    for t_all in np.round(np.arange(0.20, 0.96, 0.025), 3):
        for t_top in np.round(np.arange(0.02, t_all + 1e-9, 0.02), 3):
            s = macro_f05(entity, decide(prob, is_top, t_all, t_top), label, n_true)
            if s > best_score:
                best_score, best = s, (float(t_all), float(t_top))
    return best, best_score


def decide_expected_f(entity, prob, c_missing: float, s_empty: float) -> np.ndarray:
    """Per-entity rule: predict the top-j candidates (by probability) for the j
    that maximizes a plug-in estimate of that entity's expected F_0.5:
        EF(j) = 1.25 * sum_{i<=j} p_i / (j + 0.25 * (sum_i p_i + c_missing)),
    where c_missing is the expected number of true matches blocking never
    retrieved. Predict nothing when s_empty * prod_i(1 - p_i), an estimate of
    P(entity is a singleton), beats the best EF(j)."""
    entity = np.asarray(entity)
    order = np.lexsort((-prob, entity))
    e = entity[order]
    p = np.clip(np.asarray(prob, dtype=np.float64)[order], 0.0, 1.0 - 1e-7)
    n = len(p)
    first = np.ones(n, dtype=bool)
    first[1:] = e[1:] != e[:-1]
    starts = np.flatnonzero(first)
    gid = np.cumsum(first) - 1

    cs = np.cumsum(p)
    before = np.r_[0.0, cs[starts[1:] - 1]]
    S = cs - before[gid]
    j = (np.arange(n) - starts[gid] + 1).astype(np.float64)
    T = np.add.reduceat(p, starts)[gid] + c_missing
    ef = 1.25 * S / (j + 0.25 * T)

    best = np.maximum.reduceat(ef, starts)
    j_star = np.minimum.reduceat(np.where(ef >= best[gid] - 1e-12, j, np.inf), starts)
    ef_empty = s_empty * np.exp(np.add.reduceat(np.log1p(-p), starts))
    keep_sorted = (j <= j_star[gid]) & (best > ef_empty)[gid]

    keep = np.empty(n, dtype=bool)
    keep[order] = keep_sorted
    return keep


def tune_expected_f(entity, prob, label, n_true):
    """Grid-search (c_missing, s_empty) for decide_expected_f; returns ((c, s), score)."""
    best_score, best = -1.0, (0.0, 1.0)
    for c in (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0):
        for s in np.round(np.arange(0.1, 1.01, 0.1), 2):
            score = macro_f05(entity, decide_expected_f(entity, prob, c, s), label, n_true)
            if score > best_score:
                best_score, best = score, (float(c), float(s))
    return best, best_score
