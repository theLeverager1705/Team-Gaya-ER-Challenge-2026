# Submission history

Each leaderboard upload is recorded here. A copy of each uploaded file is kept locally as
`submissions/vN_matching_results.tsv` (git-ignored because of size); the code that produced it
is the git commit in the table. "Report F0.5" is macro F0.5 on training entities never used for
fitting or rule selection.

| # | Code commit | Model | Report F0.5 | Uploaded (IST) | Public LB F0.5 |
|---|---|---|---|---|---|
| v1 | `b5b2286` | Stage-1 LightGBM (763 trees, 200k train entities), threshold rule t_all 0.65 / t_top 0.64 | 0.9165 | (fill in) | (fill in) |
| v2 | `820b988` | + stage-2 group-context LightGBM (300k train entities), threshold rule t_all 0.725 / t_top 0.56 | 0.9264 | (fill in) | (fill in) |

Both files passed the official validator (`utils/validate_submission.py`) before upload.
