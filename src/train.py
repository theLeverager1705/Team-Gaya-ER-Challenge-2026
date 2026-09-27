"""LightGBM pair matcher with GroupKFold cross-validation by Source 1 id."""
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=42,
)


def label_pairs(pairs, truth_map):
    """1 if (s1, cand) is a ground-truth match, else 0."""
    true = {(k, x) for k, v in truth_map.items() for x in v}
    return np.array([(a, b) in true for a, b in zip(pairs.s1, pairs.cand)], dtype=int)


def train_cv(X, y, groups, n_folds=5, num_rounds=2000):
    """Train one model per fold; return (models, out_of_fold_probabilities)."""
    oof = np.zeros(len(y))
    models = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        dva = lgb.Dataset(X.iloc[va], y[va])
        m = lgb.train(PARAMS, dtr, num_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        models.append(m)
        print(f"  fold {fold}: best_iter={m.best_iteration}")
    return models, oof


def predict_ensemble(models, X):
    """Average the fold models' probabilities on new pairs."""
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
