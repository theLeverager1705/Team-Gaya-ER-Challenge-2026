# Business Entity Resolution — Amazon ML Challenge 2026 (Team Gaya)

For every Source 1 business, find all Source 2 / Source 3 records describing the same
real-world business. Pipeline:

```
normalize -> blocking (IDF-weighted rare keys, top-30 per entity, per country)
          -> 26 pair features -> stage-1 LightGBM
          -> 12 group-context features -> stage-2 LightGBM
          -> decision rule tuned for macro F0.5 -> matching_results.tsv
```

## Results (training entities never used for fitting or rule selection)

| Version | What | Macro F0.5 (report split) |
|---|---|---|
| v1 | stage-1 LightGBM + two-threshold rule | 0.9165 (200k-entity sample) |
| v2 | + stage-2 group-context model, rule chosen by tuning split | **0.9264** (300k-entity sample; stage 1 alone 0.9160) |

v2 report split (30,057 entities): pair precision 0.980, pair recall 0.854; US 0.9469, India 0.8950.
Blocking keeps 91.4% of true pairs among the top 30 candidates, which is the recall ceiling.
Every measured score (all stage/rule combinations) is stored in `models/config.json`.

## Setup

Python 3.10+ (developed on Python 3.14, Windows 11, 14 cores, 31 GB RAM).

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

The challenge data must be at `<repo root>/dataset/{train,test}/*.tsv` (paths are resolved
relative to this folder, so commands work from any working directory).

## Reproduce

```bash
# 1. Train (300k sampled training Source 1 entities blocked against the FULL training
#    Source 2/3 corpus). Writes models/model_stage1.txt, models/model_stage2.txt and
#    models/config.json (chosen decision rule + every measured score).
python src/main.py train

# 2. Predict the test set -> <repo root>/output/matching_results.tsv, candidate_pairs.tsv
python src/main.py predict

# 3. Official format check
python utils/validate_submission.py -m ../../output/matching_results.tsv \
    -c ../../output/candidate_pairs.tsv -t ../../dataset/test
```

End-to-end check on the ~10% of training entities excluded from training by a stable hash
of the entity id:

```bash
python src/main.py predict --split train --holdout-only --output-dir ../../output/holdout
python src/main.py score --matching ../../output/holdout/matching_results.tsv \
    --ground-truth ../../dataset/train/train_ground_truth.tsv
```

Runtime on the machine above (other applications holding ~18 GB of RAM): training 31 min;
test prediction TEST_RUNTIME.
All randomness is seeded; LightGBM runs with `deterministic=True`; key hashing is murmur3.

## Source files (`src/`)

| File | Purpose |
|---|---|
| `main.py` | CLI: `train`, `predict`, `score` |
| `pipeline.py` | Loading, entity splits, two-stage training, rule selection, prediction, output writing |
| `text.py` | Normalization (accent stripping, abbreviations, legal forms), numbers, blocking keys |
| `blocking.py` | `CandidateGenerator`: hashed rare-key index, IDF scoring by sparse matrix product, top-K |
| `features.py` | 26 stage-1 pair features; 12 stage-2 group-context features |
| `model.py` | LightGBM training; threshold rule; per-entity expected-F0.5 rule; rule tuning |
| `evaluate.py` | Official macro F0.5 (per entity; singletons score 1 only for an empty prediction) |

## Method

1. **Normalization.** NFKD accent stripping (`Cáble` → `cable`), lowercase, punctuation removed,
   generic abbreviations expanded (`rd` → `road`, `pvt` → `private`, `ltd` → `limited`). A
   *core name* drops legal forms and function words (`private`, `limited`, `llc`, `sarl`, `the`, ...).
2. **Blocking.** Keys = core-name tokens and bigrams, address tokens (including numbers) and
   bigrams, hashed into 2^24 buckets. Keys found in more than 1,000 Source 2/3 records are dropped.
   Score = summed IDF of shared keys, computed per country partition (open set) as a sparse
   matrix product over chunks of 5,000 Source 1 rows; the top 30 per entity are kept.
   Address keys recover records whose name is in a native script (Bengali, Devanagari) or is an
   unrelated trade name.
3. **Stage-1 features (26).** Blocking score (total, name part, address part), rank, number of
   candidates, gap/ratio to the entity's best score; rapidfuzz on core names (ratio, token-set,
   token-sort, partial, Jaro-Winkler); on addresses (ratio, token-set, token-sort); token-set on
   address numbers; first-number equality/missing; token counts; non-ASCII share of each name;
   Source 3 flag; gaps to the entity's best name/address token-set score.
4. **Stage-1 model.** LightGBM (binary, 127 leaves, learning rate 0.1, early stopping).
   Out-of-fold probabilities (2 folds) for the training entities.
5. **Stage-2 features (12) and model.** For each candidate: its stage-1 probability, rank and gap
   to the best; max/sum of the other candidates' probabilities (overall and same source); count of
   candidates with p ≥ 0.5; and similarity (name, address, numbers) to the entity's best *other*
   candidate. True matches of one business resemble each other, so this recovers e.g. trade-name
   records at the same address. Stage 2 = LightGBM on the 26 + 12 features.
6. **Decision rule** (chosen on the tuning split, then reported on the untouched split):
   - *threshold*: keep p ≥ t_all, plus the entity's best candidate if p ≥ t_top;
   - *expected F0.5*: for each entity, predict the top-j candidates for the j that maximizes
     1.25·Σ_{i≤j} p_i / (j + 0.25·(Σ_i p_i + c)), or nothing when s·Π(1 − p_i) is larger.

Entity-level splits of the training sample: 80% fit, 10% tune (early stopping + rule selection),
10% report (never used for any choice).

## Constraints

- Only the provided data is used: no external lookups, geocoding, registries or augmentation.
  The abbreviation and legal-form lists are generic hand-written rules.
- Model: LightGBM (MIT). Dependencies: pandas, numpy, scipy, scikit-learn (BSD; used only for
  feature hashing), rapidfuzz (MIT). No GPL. Model size is a few MB, far below 8B parameters.
- `country` is never one-hot encoded or filtered; it only partitions blocking, so France works
  like any other label.
