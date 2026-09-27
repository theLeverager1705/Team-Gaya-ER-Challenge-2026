# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 record, find all matching Source 2 / Source 3 records.
Pipeline: **normalize → blocking (candidates) → pair features → LightGBM → precision-first decision rule**.

## Layout

```
src/
  io_utils.py      read TSVs, write the two submission files
  normalize.py     text cleaning, legal suffixes, abbreviations (open-set country labels)
  blocking.py      candidate generation: char TF-IDF (name, name+address), postal key, acronym key
  features.py      pair similarity + context features (no country feature)
  train.py         LightGBM, 5-fold GroupKFold by Source 1 id
  predict.py       decision rule (threshold, relative-to-best, one-to-one) + OOF tuning
  evaluate.py      exact macro F0.5 scorer, blocking recall
  run_pipeline.py  end to end
configs/  notebooks/  docs/
```

## Setup (SageMaker JupyterLab space, Python 3.11)

```bash
git clone https://github.com/<team>/business_entity_resolution.git
cd business_entity_resolution
pip install -r requirements.txt
```

Put the organizers' `student_resource/` folder next to this repo (it is git-ignored):

```
student_resource/dataset/train/*.tsv
student_resource/dataset/test/*.tsv
student_resource/utils/validate_submission.py
business_entity_resolution/   <- this repo
```

## Reproduce end to end

```bash
# 1. CV report only (blocking recall + OOF macro F0.5 + tuned thresholds)
python src/run_pipeline.py --data-dir ../student_resource/dataset --out-dir output --train-only

# 2. Full run: trains on train, writes output/matching_results.tsv and output/candidate_pairs.tsv
python src/run_pipeline.py --data-dir ../student_resource/dataset --out-dir output

# 3. Validate before uploading (must print PASS)
cd ../student_resource
python3 utils/validate_submission.py \
  --matching ../business_entity_resolution/output/matching_results.tsv \
  --candidate ../business_entity_resolution/output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Useful flags: `--k-name 30 --k-full 20 --max-per-s1 60` (blocking size), `--no-country-block`.

Instance used: ml.m5.4xlarge (CPU). Runtime: fill in after the first real run.

## Rules we follow

- No external lookups (no geocoding, registries, APIs, web data). Only the provided data.
- Every model is MIT / Apache 2.0 and ≤ 8B parameters. No GPL packages.
- `candidate_pairs.tsv` is exactly the set the model scored; every match is a subset of it.

## Submission log

| # | Git tag | Change | CV F0.5 | Public LB |
|---|---------|--------|---------|-----------|
| 1 | sub-01  | LightGBM baseline | | |
