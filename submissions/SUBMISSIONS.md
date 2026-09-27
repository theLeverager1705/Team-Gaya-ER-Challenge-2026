# Submission history

Each leaderboard upload is recorded here. A copy of each uploaded file is kept locally as
`submissions/vN_matching_results.tsv` (git-ignored because of size); the code that produced it
is the git commit in the table. "Report F0.5" is macro F0.5 on training entities never used for
fitting or rule selection (US + India only; the test set also has France).

| # | Code commit | Model | Candidates / entity | Report F0.5 | Public LB F0.5 |
|---|---|---|---|---|---|
| v1 | `b5b2286` | Stage-1 LightGBM (200k train entities), threshold 0.65 / 0.64 | 30 | 0.9165 | (not recorded) |
| v2 | `269f891` | + stage-2 group-context LightGBM (300k train entities), threshold 0.725 / 0.56 | 30 | 0.9264 | 0.902 |
| v3 | (this commit) | v2 + learned pre-filter p1 ≥ 0.1 | 3.84 (test) | 0.9263 | (fill in) |

Every file passed the official validator. v2 and v3 were also checked with `--check-ids`.
The v1 file was written with CRLF line endings (fixed from v2 onward, commit `269f891`).
