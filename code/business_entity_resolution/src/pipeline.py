"""
End-to-end pipeline.

  train:   blocking on a sample of training Source 1 entities against the FULL
           training Source 2/3 corpus -> features -> LightGBM -> tune decision
           thresholds for macro F_0.5 on held-out entities -> save model.
  predict: blocking for every Source 1 entity -> features -> model ->
           matching_results.tsv + candidate_pairs.tsv.
"""
import csv
import json
import time
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import CandidateGenerator
from features import FEATURE_COLUMNS, compute_features
from model import decide, macro_f05, top_candidate_mask, train_model, tune_thresholds
from text import load_source, prepare_records

PKG_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PKG_DIR.parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "dataset"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_MODEL_DIR = PKG_DIR / "models"

BLOCKING = {"top_k": 30, "df_cap": 1000}
FEATURE_CHUNK_PAIRS = 3_000_000


def _read_raw(path) -> pd.DataFrame:
    """Read a challenge TSV as plain strings. quoting=3 (QUOTE_NONE) because
    some names contain literal double quotes and the files are never quoted."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def _log(msg):
    """Print immediately (progress of long stages is visible when output is redirected)."""
    print(msg, flush=True)


def is_holdout(entity_ids) -> np.ndarray:
    """~10% of training Source 1 entities, chosen by a stable hash of the id,
    are never used for model training so they can score the full pipeline."""
    return np.fromiter((zlib.crc32(e.encode()) % 10 == 0 for e in entity_ids), bool, len(entity_ids))


def load_corpus(split_dir: Path, split: str) -> pd.DataFrame:
    """Load and normalize Source 2 and Source 3 into one table with an is_s3 flag."""
    parts = []
    for src, is_s3 in (("source2", False), ("source3", True)):
        t0 = time.time()
        df = load_source(split_dir / f"{split}_{src}.tsv")
        df["is_s3"] = is_s3
        parts.append(df)
        _log(f"  loaded {split}_{src}: {len(df):,} records in {time.time() - t0:.0f}s")
    s23 = pd.concat(parts, ignore_index=True)
    s23["country"] = s23["country"].astype(str).astype("category")
    return s23


def _entity_chunks(s1_rows: np.ndarray, target: int):
    """Split pair indices into slices of ~target pairs without splitting an
    entity's candidates (features are relative to the entity's candidate set).
    Pairs of one entity are contiguous, as produced by CandidateGenerator."""
    n = len(s1_rows)
    start = 0
    while start < n:
        stop = min(start + target, n)
        while stop < n and s1_rows[stop] == s1_rows[stop - 1]:
            stop += 1
        yield start, stop
        start = stop


def _truth_map(gt: pd.DataFrame) -> dict:
    """{source1_entity_id: [true matched ids]} from the ground-truth table."""
    return {a: [t for t in b.split(",") if t] for a, b in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


def _pair_labels(pairs, s1, s23, truth: dict) -> np.ndarray:
    """1 where a candidate pair is a ground-truth match, else 0."""
    keys = {(a, m) for a, ms in truth.items() for m in ms}
    s1_ids = s1["entity_id"].to_numpy()[pairs["s1_row"].to_numpy()]
    c_ids = s23["entity_id"].to_numpy()[pairs["c_row"].to_numpy()]
    return np.fromiter(((a, c) in keys for a, c in zip(s1_ids, c_ids)), bool, len(pairs))


def train(data_dir=DEFAULT_DATA_DIR, model_dir=DEFAULT_MODEL_DIR, n_entities: int = 300_000,
          seed: int = 42) -> dict:
    """Train the pair classifier, tune thresholds, and save model.txt + config.json (with metrics)."""
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    split_dir = data_dir / "train"
    model_dir.mkdir(parents=True, exist_ok=True)

    _log("Loading training Source 2/3 corpus...")
    s23 = load_corpus(split_dir, "train")

    raw_s1 = _read_raw(split_dir / "train_source1.tsv")
    raw_s1 = raw_s1[~is_holdout(raw_s1["entity_id"].to_numpy())]
    raw_s1 = raw_s1.sample(n=min(n_entities, len(raw_s1)), random_state=seed).reset_index(drop=True)
    s1 = prepare_records(raw_s1)
    s1["country"] = s1["country"].astype("category")
    del raw_s1
    _log(f"Training on {len(s1):,} sampled Source 1 entities")

    gt = _read_raw(split_dir / "train_ground_truth.tsv")
    gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))]
    truth = _truth_map(gt)
    n_true = np.array([len(truth.get(e, [])) for e in s1["entity_id"]], dtype=np.int64)

    _log("Blocking...")
    t0 = time.time()
    pairs = CandidateGenerator(**BLOCKING).generate(s1, s23)
    label = _pair_labels(pairs, s1, s23, truth)
    blocking_stats = {
        "entities": int(len(s1)),
        "candidate_pairs": int(len(pairs)),
        "pairs_per_entity": float(len(pairs) / len(s1)),
        "pair_recall": float(label.sum() / n_true.sum()),
        "seconds": round(time.time() - t0, 1),
    }
    _log(f"Blocking: {blocking_stats}")

    _log("Computing features...")
    t0 = time.time()
    X = compute_features(pairs, s1, s23)
    _log(f"Features for {len(X):,} pairs in {time.time() - t0:.0f}s")

    # Entity-level split: 80% fit / 10% early stopping + threshold tuning /
    # 10% report (never used for any fitting decision)
    rng = np.random.default_rng(seed)
    role = rng.choice(np.array([0, 1, 2], dtype=np.int8), size=len(s1), p=[0.8, 0.1, 0.1])
    pair_role = role[pairs["s1_row"].to_numpy()]

    fit, tune = pair_role == 0, pair_role == 1
    _log("Training LightGBM...")
    t0 = time.time()
    booster = train_model(X[fit], label[fit], X[tune], label[tune])
    _log(f"Trained {booster.best_iteration} rounds in {time.time() - t0:.0f}s")

    prob = booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)

    def entity_view(r):
        ent = np.flatnonzero(role == r)
        code = np.full(len(s1), -1, dtype=np.int64)
        code[ent] = np.arange(len(ent))
        m = pair_role == r
        return code[pairs["s1_row"].to_numpy()[m]], prob[m], label[m], n_true[ent]

    e1, p1, l1, t1 = entity_view(1)
    (t_all, t_top), tune_score = tune_thresholds(e1, p1, l1, t1)
    e2, p2, l2, t2 = entity_view(2)
    report_score = macro_f05(e2, decide(p2, top_candidate_mask(e2, p2), t_all, t_top), l2, t2)
    naive_score = macro_f05(e2, p2 >= 0.5, l2, t2)
    _log(f"Thresholds t_all={t_all} t_top={t_top} | macro F0.5 tune={tune_score:.4f} "
         f"report={report_score:.4f} (p>=0.5 only: {naive_score:.4f})")

    importance = dict(zip(FEATURE_COLUMNS, booster.feature_importance("gain").round(1).tolist()))
    config = {
        "blocking": BLOCKING,
        "features": FEATURE_COLUMNS,
        "thresholds": {"t_all": t_all, "t_top": t_top},
        "best_iteration": booster.best_iteration,
        "metrics": {
            "blocking": blocking_stats,
            "macro_f05_tune_split": round(tune_score, 4),
            "macro_f05_report_split": round(report_score, 4),
            "macro_f05_report_split_p05_only": round(naive_score, 4),
            "singleton_rate": float((n_true == 0).mean()),
        },
        "feature_importance_gain": importance,
        "n_train_entities": int(len(s1)),
        "seed": seed,
    }
    booster.save_model(str(model_dir / "model.txt"), num_iteration=booster.best_iteration)
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    _log(f"Saved model to {model_dir}")
    return config


def _write_id_lists(path, s1_ids, rows, cand_ids, column):
    """Write one row per Source 1 entity (empty list when it has no ids), tab-separated, unquoted."""
    lists = pd.Series(cand_ids).groupby(rows, sort=False).agg(",".join)
    out = np.full(len(s1_ids), "", dtype=object)
    out[lists.index.to_numpy()] = lists.to_numpy()
    pd.DataFrame({"source1_entity_id": s1_ids, column: out}).to_csv(
        path, sep="\t", index=False, quoting=csv.QUOTE_NONE, encoding="utf-8")


def predict(data_dir=DEFAULT_DATA_DIR, model_dir=DEFAULT_MODEL_DIR, output_dir=DEFAULT_OUTPUT_DIR,
            split: str = "test", holdout_only: bool = False) -> dict:
    """holdout_only (train split): score only the hash-held-out training
    entities, never seen by the model, to check the whole pipeline end to end."""
    data_dir, model_dir, output_dir = Path(data_dir), Path(model_dir), Path(output_dir)
    split_dir = data_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((model_dir / "config.json").read_text())
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    t_all, t_top = config["thresholds"]["t_all"], config["thresholds"]["t_top"]

    _log(f"Loading {split} data...")
    s23 = load_corpus(split_dir, split)
    raw_s1 = _read_raw(split_dir / f"{split}_source1.tsv")
    if holdout_only:
        raw_s1 = raw_s1[is_holdout(raw_s1["entity_id"].to_numpy())].reset_index(drop=True)
    s1 = prepare_records(raw_s1)
    s1["country"] = s1["country"].astype("category")
    del raw_s1
    _log(f"  {len(s1):,} Source 1 entities")

    _log("Blocking...")
    t0 = time.time()
    pairs = CandidateGenerator(**config["blocking"]).generate(s1, s23)
    _log(f"  {len(pairs):,} candidate pairs ({len(pairs) / len(s1):.1f}/entity) in {time.time() - t0:.0f}s")

    _log("Scoring candidates...")
    t0 = time.time()
    rows = pairs["s1_row"].to_numpy()
    prob = np.empty(len(pairs), dtype=np.float32)
    for a, b in _entity_chunks(rows, FEATURE_CHUNK_PAIRS):
        X = compute_features(pairs.iloc[a:b], s1, s23)
        prob[a:b] = booster.predict(X[config["features"]])
        _log(f"  scored {b:,}/{len(pairs):,} pairs")
    _log(f"  done in {time.time() - t0:.0f}s")

    matched = decide(prob, top_candidate_mask(rows, prob), t_all, t_top)

    s1_ids = s1["entity_id"].to_numpy()
    c_ids = s23["entity_id"].to_numpy()[pairs["c_row"].to_numpy()]
    cand_path = output_dir / "candidate_pairs.tsv"
    match_path = output_dir / "matching_results.tsv"
    _write_id_lists(cand_path, s1_ids, rows, c_ids, "candidate_entity_ids")
    _write_id_lists(match_path, s1_ids, rows[matched], c_ids[matched], "matched_entity_ids")

    n_with = len(np.unique(rows[matched]))
    stats = {
        "entities": int(len(s1)),
        "candidate_pairs": int(len(pairs)),
        "matched_pairs": int(matched.sum()),
        "entities_with_matches": int(n_with),
        "entities_predicted_singleton": int(len(s1) - n_with),
    }
    _log(f"Wrote {match_path} and {cand_path}: {stats}")
    return stats
