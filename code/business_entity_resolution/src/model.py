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
