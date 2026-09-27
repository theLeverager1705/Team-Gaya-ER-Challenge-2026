"""End-to-end pipeline: data -> normalize -> blocking -> features -> LightGBM -> outputs.

Usage (from the repo root):
    python src/run_pipeline.py --data-dir ../student_resource/dataset --out-dir output
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from blocking import candidates_to_map, generate_candidates
from evaluate import blocking_recall, breakdown
from features import build_features
from io_utils import load_split, load_truth, write_id_lists
from normalize import add_normalized_columns
from predict import decide, tune
from train import label_pairs, predict_ensemble, train_cv


def prepare(data_dir, split, args):
    """Load one split, normalize it, generate candidates and features."""
    s1, pool = load_split(data_dir, split)
    s1, pool = add_normalized_columns(s1), add_normalized_columns(pool)
    cand = generate_candidates(s1, pool, k_name=args.k_name, k_full=args.k_full,
                               max_per_s1=args.max_per_s1,
                               block_by_country=not args.no_country_block)
    pairs, X, cols = build_features(cand, s1, pool)
    return s1, pool, cand, pairs, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--k-name", type=int, default=30)
    ap.add_argument("--k-full", type=int, default=20)
    ap.add_argument("--max-per-s1", type=int, default=60)
    ap.add_argument("--no-country-block", action="store_true")
    ap.add_argument("--train-only", action="store_true", help="CV report, skip test")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------- train: blocking recall + CV ----------------
    truth = load_truth(Path(args.data_dir) / "train" / "train_ground_truth.tsv")
    s1, pool, cand, pairs, X = prepare(args.data_dir, "train", args)
    br = blocking_recall(candidates_to_map(cand), truth)
    print(f"[train] blocking: {br}  ({time.time()-t0:.0f}s)")

    y = label_pairs(pairs, truth)
    print(f"[train] pairs={len(pairs)} positives={y.sum()}")
    models, oof = train_cv(X, y, pairs.s1.values)
    best = tune(pairs, oof, truth)
    report = {"blocking": br, "decision": best,
              "cv": breakdown(decide(pairs, oof, best["T"], best["alpha"], best["one_to_one"]), truth)}
    print("[train] CV:", json.dumps(report, indent=2))
    (out / "cv_report.json").write_text(json.dumps(report, indent=2))
    if args.train_only:
        return

    # ---------------- test: predict + write both files ----------------
    t1, tpool, tcand, tpairs, tX = prepare(args.data_dir, "test", args)
    probs = predict_ensemble(models, tX[X.columns])
    matches = decide(tpairs, probs, best["T"], best["alpha"], best["one_to_one"])
    s1_ids = t1["entity_id"].tolist()
    # candidate_pairs = exactly the pairs the model scored
    write_id_lists(candidates_to_map(tpairs), s1_ids, out / "candidate_pairs.tsv", "candidate_entity_ids")
    res = write_id_lists(matches, s1_ids, out / "matching_results.tsv", "matched_entity_ids")
    n_match = (res.matched_entity_ids != "").sum()
    print(f"[test] wrote {len(res)} rows, {n_match} with matches  ({time.time()-t0:.0f}s total)")
    print("[test] country counts:", t1["country"].value_counts().to_dict())


if __name__ == "__main__":
    main()
