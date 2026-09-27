# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We retrieve candidates with IDF-weighted rare-key blocking over **both name and address** within each
country, score every candidate pair with a LightGBM classifier on string-similarity and
blocking-context features, and turn pair probabilities into match lists with thresholds tuned directly
for macro F0.5. On training entities never used for fitting or tuning, the pipeline scores
**macro F0.5 = 0.9165**. It runs on the full data (1.7M × 10M records) on one laptop in about 20 minutes.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** 2.2M train / 1.7M test Source 1 entities against ~10M Source 2+3 records per split, so all-pairs
  comparison (~10¹³ pairs) is impossible and blocking must use indexes.
- **Label structure:** 5.6% of training entities are singletons; the rest have 1–11 matches (mean 3.7).
  Every true pair shares the same `country` label (17,362 of 17,362 checked); no country is missing.
- **Noise seen in the data:** names in native scripts (Bengali/Devanagari) in Source 2; unrelated trade/DBA
  names at the same address; injected diacritics (`Cáble`, `Prívate`) and character swaps (`Co1onial`);
  word-order changes; legal-suffix churn (`Pvt Ltd` / `Private Limited` / `Partners Partners`); address
  components reordered and abbreviated (`Rd`/`Road`, `NY`/`New York`); street numbers perturbed (`26` vs `26/6`).
  **Conclusion: the address is at least as important as the name, and matching on names alone fails.**
- Some fields contain literal `"` characters, so files are read with quoting disabled.

### 2.2 Solution Strategy
**Approach Type:** Blocking + gradient-boosted pair classifier + metric-tuned decision rule  
**Core Innovation:** IDF-weighted retrieval over hashed name *and* address keys (tokens and bigrams),
computed as chunked sparse matrix products. It is fast enough to use the full corpus and recovers matches
whose names share nothing.

---

## 3. Candidate Generation (Blocking)
- **Normalization:** Unicode NFKD with accents stripped, lowercase, punctuation removed, generic abbreviation expansion
  (street types, units, directions, legal forms). A *core name* drops legal forms and function words.
- **Blocking keys used:** core-name tokens, core-name bigrams, address tokens (including numbers), and address
  bigrams, all hashed to 2²⁴ buckets. Keys that occur in more than 1,000 Source 2/3 records are dropped as
  non-discriminative.
- **Scoring:** score(s1, c) = Σ IDF of shared keys, computed within each country partition (an open set, so France
  is handled like any other label) as a sparse matrix product per chunk of 20k Source 1 rows. The top 30
  candidates per entity are kept.
- **Candidate pairs generated:** ~30 per entity (5.99M for the 200k-entity training sample; about 52M for the test set).
- **How true matches were kept:** recall was measured against ground truth at full corpus scale. Top-K recall
  was 79.2% @5, 90.4% @20, **91.5% @30** and 93.0% @60, so K = 30 was chosen as the cost/recall knee. Address keys
  are what recover native-script and trade-name matches.

---

## 4. Matching Model

**Features used (26):**
- Name features: rapidfuzz ratio, token-set, token-sort, partial ratio, Jaro-Winkler on core names; empty-name flag;
  token counts; share of non-ASCII characters (detects native-script names).
- Address features: ratio, token-set, token-sort; token-set over the address *numbers*; first-number equality/missing.
- Other: blocking score (total, name part, address part), rank within the entity's candidates, number of candidates,
  gap and ratio to the entity's best blocking score, gap to the best name/address token-set score, Source 3 flag.

**Model type:** LightGBM binary classifier (MIT licence; 763 trees, 127 leaves; ≪ 8B parameters). Trained on 200k
sampled training entities (80% fit / 10% early stopping and threshold tuning / 10% untouched report split, split by entity).
Most important features by gain: candidate rank (47%), address-number agreement (15%), address token-set (6%),
relative blocking score (5%), name Jaro-Winkler (4%).

**Threshold selection method:** grid search maximizing **macro F0.5** (the official per-entity metric, singletons
included) over a two-threshold rule: keep candidates with p ≥ t_all, plus each entity's best candidate if p ≥ t_top.
Selected t_all = 0.65, t_top = 0.64.

---

## 5. Results & Error Analysis

| Split (training entities, never used for fitting) | Macro F0.5 |
|---|---|
| Threshold-tuning split (10%) | 0.9141 |
| Report split (10%, untouched) | **0.9165** |
| Report split, plain p ≥ 0.5 rule | 0.9133 |

- **Recall ceiling:** 8.5% of true pairs never reach the top-30 candidates, so this is the main remaining loss.
- **Common false positives (wrong merges):** [filled in from the hold-out analysis below]
- **Common false negatives (missed matches):** [filled in from the hold-out analysis below]

---

## 6. Conclusion
Treating the address as a first-class blocking signal and learning the match decision from 2.2M labelled
entities gives a fast, fully reproducible pipeline at macro F0.5 ≈ 0.92 on unseen training entities. The largest
remaining gain is blocking recall (candidates beyond the top 30 and native-script names).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `main.py` (CLI), `pipeline.py` (train/predict orchestration),
`text.py` (normalization and keys), `blocking.py` (candidate generation), `features.py` (pair features),
`model.py` (LightGBM and threshold tuning), `evaluate.py` (official macro F0.5). The trained model and thresholds are in `models/`.

```bash
pip install -r requirements.txt
python src/main.py train      # -> models/model.txt, models/config.json
python src/main.py predict    # -> output/matching_results.tsv, output/candidate_pairs.tsv
```

### B. Additional Results
Only the provided data is used: no external lookups, geocoding or augmentation. The abbreviation lists are generic
hand-written rules. All steps are deterministic (fixed seeds, deterministic LightGBM, murmur hashing).
