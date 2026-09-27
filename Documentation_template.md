# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Team Gaya  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
Candidates come from IDF-weighted rare-key retrieval over **both name and address** within each
country. A two-stage LightGBM model then scores each candidate pair: stage 1 uses 26 pair
features, and stage 2 adds 12 *group-context* features describing the entity's other
candidates. A decision rule chosen for macro F0.5 turns the scores into match lists. On
30,057 training entities never used for fitting or rule selection, the final model scores
**macro F0.5 = 0.9264** (pair precision 0.980, pair recall 0.854), against 0.9160 without stage 2.
Candidate generation ends with a learned pre-filter, so the final model scores only **3.8 candidates
per Source 1 entity** on the test set (6.65M pairs out of the 1.7·10¹³ possible), at no measurable loss
(report-split macro F0.5 0.9263). The full test set runs on one laptop.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale.** 2.21M train / 1.73M test Source 1 entities, each split with ~10M Source 2+3 records,
  so all-pairs comparison is impossible and blocking must use indexes.
- **Labels.** 5.6% of training entities are singletons; the others have 1–11 matches (mean 3.7).
  All 17,362 true pairs in a 5,000-entity sample share the `country` label; no country is empty.
- **Noise seen in the data.** Source 2 names often written in native script (Bengali, Devanagari);
  unrelated trade/DBA names at the same address (`Syndrex` matching `Straight Edge Pub`, both at
  515 Monroe Ave); injected diacritics (`Cáble`, `Prívate`); character swaps (`Co1onial`);
  word-order changes; legal-suffix churn (`Pvt Ltd` / `Private Limited` / `Partners Partners`);
  reordered, abbreviated addresses; perturbed street numbers (`26` vs `26/6`, `515` vs `17`).
  **Consequence: the address is as important as the name, and name-only matching fails.**
- Up to 349 lines per file contain literal `"`, so files are read with quote handling disabled
  (every line has exactly 4 tab-separated fields).

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage gradient-boosted pair classifier + metric-tuned decision rule  
**Core Innovation:** (1) retrieval over hashed name *and* address keys, computed as chunked sparse
matrix products, which is fast enough for the full corpus and recovers matches whose names share
nothing; (2) a second stage that judges each candidate against the entity's other candidates,
because the true matches of one business resemble each other.

---

## 3. Candidate Generation (Blocking)
- **Normalization:** NFKD accent stripping, lowercase, punctuation removed, generic abbreviations
  expanded (street types, units, directions, legal forms). A *core name* drops legal forms and
  function words.
- **Blocking keys used:** core-name tokens, core-name bigrams, address tokens (numbers included),
  and address bigrams, hashed to 2²⁴ buckets. Keys occurring in more than 1,000 Source 2/3 records
  are dropped as non-discriminative.
- **Step 1, retrieval:** score = Σ IDF of shared keys, computed inside each country partition.
  Partitions are whatever labels occur (an open set), so France needs no special handling. The top 30
  records per entity are retrieved (51.9M pairs for the test set).
- **Step 2, learned pre-filter:** the stage-1 LightGBM (26 pair features, see §4) scores the 30
  retrieved pairs, and only pairs with p1 ≥ 0.1 stay candidates. The cutoff is the largest value in
  {0, 0.001, …, 0.1} whose tune-split macro F0.5 is within 0.0001 of using no cutoff. The final model
  (stage 2) runs inference **only on these candidates**, and they are exactly `candidate_pairs.tsv`.
  To be transparent: the stage-1 model does score all 30 retrieved pairs; the stage-2 group features
  summarize those stage-1 scores.
- **Candidate pairs generated (test):** 6,652,763, i.e. **3.84 per entity**, versus 1.7·10¹³ possible
  pairs, a reduction of more than 99.99999%. 56,812 entities have no candidate and are predicted singletons.
- **How true matches were kept:** recall was measured against ground truth at full corpus scale.
  Retrieval keeps 79.2% of true pairs in the top 5, 90.4% @20, **91.5% @30**, 93.0% @60 (20k
  entities); K = 30 is the knee. The pre-filter trade-off on held-out entities (report split):

  | p1 cutoff | candidates / entity | recall ceiling | macro F0.5 |
  |---|---|---|---|
  | none | 29.95 | 91.37% | 0.9264 |
  | 0.01 | 4.31 | 91.22% | 0.9264 |
  | **0.1 (chosen on tune split)** | **3.63** | **90.42%** | **0.9263** |

---

## 4. Matching Model

**Features used (stage 1, 26):**
- Name features: rapidfuzz ratio, token-set, token-sort, partial ratio and Jaro-Winkler on core names;
  empty-name flag; token counts; share of non-ASCII characters (native-script detection).
- Address features: ratio, token-set, token-sort; token-set on address numbers; first-number
  equality and missing flag.
- Other: blocking score (total, name part, address part), rank among the entity's candidates,
  number of candidates, gap and ratio to the entity's best score, gaps to the best name/address
  token-set score, Source 3 flag.

**Features used (stage 2, +12 group-context):** stage-1 probability, its rank and gap to the entity's
best; max and sum of the other candidates' probabilities (overall and same source); number of
candidates with p ≥ 0.5; name, address and number similarity to the entity's best *other* candidate,
plus that candidate's probability and source. Stage 2 is trained on **out-of-fold** stage-1
probabilities (2 folds), so it learns from probabilities of test-time quality.

**Model type:** LightGBM binary classifiers (MIT licence; 127 leaves, learning rate 0.1, early
stopping on the tune split). Stage 1 has 1,067 trees and stage 2 has 571 (model files 15 MB and 8 MB, far
below 8B parameters). Training uses 300k sampled training entities (9.0M candidate pairs), split by
entity into 80% fit / 10% tune / 10% report.

**Feature importance (gain).** Stage 1: rank among candidates 47%, address-number token-set 15%,
address token-set 6%, relative blocking score 5%, name Jaro-Winkler 4%. Stage 2: stage-1
probability 67%, its rank within the entity 19%, number of candidates with p ≥ 0.5 4%,
address-number token-set 3%.

**Threshold selection method:** two rules are tuned on the tune split for macro F0.5 (per entity,
singletons included), and the best (stage, rule) pair is chosen there:
- *threshold rule*: keep p ≥ t_all, plus the entity's best candidate if p ≥ t_top;
- *expected-F0.5 rule*: for each entity, predict its top-j candidates for the j maximizing
  1.25·Σ_{i≤j} p_i / (j + 0.25·(Σ_i p_i + c)), or nothing when s·Π(1 − p_i), an estimate that the
  entity is a singleton, is larger.

Chosen (highest tune score): stage 2 + threshold rule, t_all = 0.725, t_top = 0.56, applied
to the pre-filtered candidates.

---

## 5. Results & Error Analysis

Macro F0.5 on the tune split (~30k entities, used for selection) and the untouched report split
(30,057 entities, never used for any fitting or selection):

| Model | Rule | Tune split | Report split |
|---|---|---|---|
| Stage 1 only | threshold (0.675 / 0.64) | 0.9164 | 0.9160 |
| Stage 1 only | expected F0.5 (c = 0.8, s = 1.0) | 0.9166 | 0.9171 |
| **Stage 1 + 2** | **threshold (0.725 / 0.56)** | **0.9269** | **0.9264** |
| Stage 1 + 2 | expected F0.5 (c = 0.0, s = 0.9) | 0.9267 | 0.9265 |

v1 (stage 1 only, 200k training entities) scored 0.9165 on its report split. The group-context
stage adds about one point. The two decision rules tie once probabilities are this good.

**Public leaderboard:** v2 scored **0.902**. That is below the held-out 0.926, which is expected: the test
set is 15% France, a country absent from training, and 47% India, the harder training country (0.895
held-out, versus 0.947 for the US).

- **Where the loss comes from (report split):** pair precision is 0.980 and pair recall 0.854. Of
  103,940 true pairs, 8,965 (8.6%) never reach the top-30 candidates and 6,266 (6.0%) are
  retrieved but rejected by the model; there are 1,837 false positives. 1,521 of 1,673 singletons
  are correctly left empty. By share of lost F0.5: 69% is partial recall on entities with
  matches, 24% is entities with matches that get an empty prediction, and 7% is singletons given a
  false match. India is harder than US (0.8950 vs 0.9469), driven by native-script names and
  noisier addresses. France is unseen in training; the features are language-agnostic.
- **Common false positives (wrong merges):** near-identical *distractor* records, meaning the same
  name at a slightly different number (`1572` vs `1582 Hardee Street`, door `16-5-1` vs `16-5-3`,
  `2/322F` vs `G-2/326F`); a different business at the same address (`Mclean` at 933 Klare
  Lane); and one-character name changes (`LFZ Foods` vs `LZ Foods`) at an identical address.
- **Common false negatives (missed matches):** candidates with an empty address; true matches
  whose name is an unrelated trade name (`Belohalo`, `>> Quonylacira`) or is written in Kannada or
  Devanagari script; heavy typos (`CHENNAI PRTEDCTON`); and street numbers that differ between
  sources (`1557` vs `1330 Bunce Road`).

---

## 6. Conclusion
Treating the address as a first-class blocking signal, learning the match decision from 2.2M
labelled entities, and adding a second stage that judges each candidate against the entity's
other candidates gives macro F0.5 ≈ 0.926 on unseen training entities, with a fast and
fully reproducible pipeline. The largest remaining gains are blocking recall (8.6% of true pairs
are never retrieved) and native-script names, for example through transliteration-aware keys.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `main.py` (CLI), `pipeline.py` (training/prediction
orchestration), `text.py` (normalization and keys), `blocking.py` (candidate generation),
`features.py` (stage-1 and stage-2 features), `model.py` (LightGBM, decision rules, tuning),
`evaluate.py` (official macro F0.5). Trained models and the chosen rule, with every measured
score, are in `models/config.json`.

```bash
pip install -r requirements.txt
python src/main.py train      # -> models/
python src/main.py predict    # -> output/matching_results.tsv, output/candidate_pairs.tsv
```

### B. Additional Results
Only the provided data is used: no external lookups, geocoding or augmentation. The
abbreviation lists are generic hand-written rules. All steps are deterministic (fixed seeds,
deterministic LightGBM, murmur hashing). Submission history: `submissions/SUBMISSIONS.md`.
