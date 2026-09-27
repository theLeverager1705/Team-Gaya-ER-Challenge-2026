# Business Entity Resolution (Amazon ML Challenge 2026)

For every Source 1 business, find the Source 2 / Source 3 records that describe the same
real-world business. Pipeline: **blocking → pair features → LightGBM classifier → macro-F0.5 tuned decision rule**.

## Layout

```
<submission root>/
├── dataset/{train,test}/*.tsv        # challenge data (not shipped)
├── output/                           # matching_results.tsv, candidate_pairs.tsv
├── Documentation_template.md         # methodology write-up
└── code/business_entity_resolution/
    ├── src/
    │   ├── main.py        # CLI: train / predict / score
    │   ├── pipeline.py    # orchestration of train and predict
    │   ├── text.py        # normalization, tokenization, blocking keys
    │   ├── blocking.py    # candidate generation (IDF-weighted rare-key retrieval)
    │   ├── features.py    # vectorized pair features (rapidfuzz)
    │   ├── model.py       # LightGBM + threshold tuning
    │   └── evaluate.py    # official macro F0.5
    ├── models/            # trained model.txt + config.json (thresholds, metrics)
    ├── utils/validate_submission.py   # official validator (unchanged)
    └── requirements.txt
```

## Setup

Python 3.10+ (developed on 3.14, Windows 11, 31 GB RAM, 14 cores).

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

Place the data so that `dataset/` sits next to `code/` (paths are resolved relative to this
folder, so commands work from any working directory).

## Reproduce

```bash
# 1. Train (samples 300k training Source 1 entities; blocks them against the FULL
#    training Source 2/3 corpus). Writes models/model.txt and models/config.json.
python src/main.py train

# 2. Predict the test set -> ../../output/matching_results.tsv and candidate_pairs.tsv
python src/main.py predict

# 3. Validate the format (official validator)
python utils/validate_submission.py -m ../../output/matching_results.tsv \
    -c ../../output/candidate_pairs.tsv -t ../../dataset/test
```

Optional end-to-end check on ~10% of training entities that are never used for training
(chosen by a stable hash of the entity id):

```bash
python src/main.py predict --split train --holdout-only --output-dir ../../output/holdout
python src/main.py score --matching ../../output/holdout/matching_results.tsv \
    --ground-truth ../../dataset/train/train_ground_truth.tsv
```

Everything is deterministic (fixed seeds, deterministic LightGBM, murmur hashing).

## Method (short)

1. **Normalization**: accents stripped (`Cáble` → `cable`), punctuation removed, generic
   abbreviations expanded (`Rd` → `road`, `Pvt` → `private`), legal forms and function
   words dropped from a "core" name.
2. **Blocking**: within each country (an open set, so France is handled like any other label), every
   record is hashed into rare keys: core-name tokens and bigrams, address tokens and
   bigrams. Keys shared by more than 1,000 Source 2/3 records are dropped. Candidates are
   scored by the summed IDF of shared keys (one sparse matrix product per chunk of
   Source 1 rows), and the top 30 per entity are kept. Address keys let us match records
   whose name is in a native script or is a different trade name.
3. **Features**: blocking scores (total/name/address, rank, gap to the best candidate),
   rapidfuzz name and address similarities (ratio, token set/sort, partial, Jaro-Winkler),
   address-number agreement, native-script share, source flag.
4. **Model**: LightGBM binary classifier (MIT licence) trained on labelled candidate pairs.
5. **Decision rule**: keep candidates with p ≥ t_all, plus each entity's best candidate if
   p ≥ t_top. Both thresholds are grid-searched to maximize macro F0.5 on held-out entities.

Constraints: only the provided data is used (no external lookups); the model is LightGBM
(MIT), far below 8B parameters; `country` is treated as an open set.
