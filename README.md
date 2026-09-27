# Business Entity Resolution — Amazon ML Challenge 2026 (Team Gaya)

For every Source 1 business, find all Source 2 / Source 3 records that describe the same
real-world business. Scored by macro F0.5 (per Source 1 entity, singletons included).

## Repository layout

```
code/business_entity_resolution/   <- THE SUBMITTED PIPELINE (see its README for full details)
  src/                              main.py (CLI), pipeline.py, text.py, blocking.py,
                                    features.py, model.py, evaluate.py
  models/                           trained LightGBM models + config.json (rule, metrics)
  utils/validate_submission.py      official validator (unchanged)
  README.md, requirements.txt
Documentation_template.md           methodology write-up (final package document)
submissions/SUBMISSIONS.md          version history of every leaderboard upload
src/, requirements.txt, notebooks/  early starter prototype (char TF-IDF blocking, SageMaker);
                                    kept for history, NOT used for any submission
```

`dataset/`, `output/` and `artifacts/` are git-ignored (data and multi-GB outputs).

## Quick start

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python src/main.py train      # ~25 min; writes models/
python src/main.py predict    # writes ../../output/matching_results.tsv + candidate_pairs.tsv
python utils/validate_submission.py -m ../../output/matching_results.tsv \
    -c ../../output/candidate_pairs.tsv -t ../../dataset/test
```

Place the challenge data at `dataset/train/*.tsv` and `dataset/test/*.tsv` in the repo root.

## Final submission package

```
<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt, models/, utils/}
└── Documentation_template.md
```

## Rules we follow

- No external lookups (no geocoding, registries, APIs, web data). Only the provided data.
- Model: LightGBM (MIT). All dependencies are MIT/BSD; no GPL. Far below 8B parameters.
- `country` is an open set: blocking partitions on whatever labels appear (France included).
- `candidate_pairs.tsv` is exactly the set the model scored; matches are a subset of it.
