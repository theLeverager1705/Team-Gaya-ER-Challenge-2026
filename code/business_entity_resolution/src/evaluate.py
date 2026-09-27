"""
Official metric: F_0.5 computed per Source 1 entity, macro-averaged over all
Source 1 entities. An entity with no true matches scores 1.0 for an empty
prediction and 0.0 for any prediction.
"""
import numpy as np
import pandas as pd


def per_entity_f05(n_true, n_pred, tp) -> np.ndarray:
    """F_0.5 for each entity from its true-match count, predicted count and true positives."""
    n_true = np.asarray(n_true, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64)
    precision = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    recall = np.divide(tp, n_true, out=np.zeros_like(tp), where=n_true > 0)
    denom = 0.25 * precision + recall
    f = np.divide(1.25 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)


def _parse_id_lists(df: pd.DataFrame, col: str) -> dict:
    """{source1_entity_id: set of listed ids} from a two-column results/ground-truth table."""
    return {k: {t for t in v.split(",") if t} for k, v in zip(df.iloc[:, 0], df[col])}


def _read_tsv(path) -> pd.DataFrame:
    """Read a challenge TSV as plain strings (no quote processing, empty cells kept as '')."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def score_files(matching_path, ground_truth_path) -> dict:
    """Score a matching_results.tsv against a ground-truth file, over the
    entities listed in the matching file (so a held-out subset can be scored)."""
    truth = _parse_id_lists(_read_tsv(ground_truth_path), "matched_entity_ids")
    pred = _parse_id_lists(_read_tsv(matching_path), "matched_entity_ids")
    missing = [k for k in pred if k not in truth]
    if missing:
        raise ValueError(f"{len(missing)} predicted entities not in ground truth, e.g. {missing[:3]}")
    keys = list(pred)
    n_true = [len(truth[k]) for k in keys]
    n_pred = [len(pred[k]) for k in keys]
    tp = [len(truth[k] & pred[k]) for k in keys]
    f = per_entity_f05(n_true, n_pred, tp)
    tp_sum, pred_sum, true_sum = sum(tp), sum(n_pred), sum(n_true)
    return {
        "entities": len(keys),
        "macro_f05": float(f.mean()),
        "pair_precision": tp_sum / pred_sum if pred_sum else 0.0,
        "pair_recall": tp_sum / true_sum if true_sum else 0.0,
    }
