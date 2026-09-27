#!/usr/bin/env python3
"""
Business Entity Resolution - command line entry point.

  python src/main.py train                     # fit model on training data -> models/
  python src/main.py predict                   # test set -> ../../output/*.tsv
  python src/main.py predict --split train --holdout-only --output-dir <dir>
                                               # end-to-end check on unseen training entities
  python src/main.py score --matching <file> --ground-truth <file>
"""
import argparse
import json
import os
import sys

# Windows consoles default to cp1252; names include Devanagari/Bengali text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import score_files  # noqa: E402
from pipeline import DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, predict, train  # noqa: E402


def main():
    """Parse the subcommand and run the matching pipeline stage."""
    parser = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the pair classifier")
    p_train.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p_train.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p_train.add_argument("--n-entities", type=int, default=300_000,
                         help="training Source 1 entities to sample (default: %(default)s)")
    p_train.add_argument("--seed", type=int, default=42)

    p_pred = sub.add_parser("predict", help="write matching_results.tsv and candidate_pairs.tsv")
    p_pred.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p_pred.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p_pred.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_pred.add_argument("--split", default="test", choices=["test", "train"])
    p_pred.add_argument("--holdout-only", action="store_true",
                        help="with --split train: only the hash-held-out entities never used in training")

    p_score = sub.add_parser("score", help="macro F0.5 of a matching file against ground truth")
    p_score.add_argument("--matching", required=True)
    p_score.add_argument("--ground-truth", required=True)

    args = parser.parse_args()
    if args.command == "train":
        result = train(args.data_dir, args.model_dir, args.n_entities, args.seed)
        print(json.dumps(result["metrics"], indent=2))
    elif args.command == "predict":
        predict(args.data_dir, args.model_dir, args.output_dir, args.split, args.holdout_only)
    else:
        print(json.dumps(score_files(args.matching, args.ground_truth), indent=2))


if __name__ == "__main__":
    main()
