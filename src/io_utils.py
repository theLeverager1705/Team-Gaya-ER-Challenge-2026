"""Reading the challenge TSV files and writing the two submission files."""
from pathlib import Path

import pandas as pd


def load_source(path):
    """Read one source file as strings. Empty cells stay '' (never NaN)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_split(data_dir, split):
    """Return (s1, pool) for 'train' or 'test'.

    pool = Source 2 and Source 3 stacked; the source is kept in a 'source' column
    derived from the entity_id prefix (S2-/S3-).
    """
    d = Path(data_dir) / split
    s1 = load_source(d / f"{split}_source1.tsv")
    s2 = load_source(d / f"{split}_source2.tsv")
    s3 = load_source(d / f"{split}_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True)
    pool["source"] = pool["entity_id"].str[:2]
    s1["source"] = "S1"
    return s1, pool


def load_truth(path):
    """Ground truth as {source1_entity_id: [matched ids]}; singletons map to []."""
    gt = load_source(path)
    return {
        r.source1_entity_id: [x for x in r.matched_entity_ids.split(",") if x]
        for r in gt.itertuples(index=False)
    }


def write_id_lists(mapping, s1_ids, path, value_col):
    """Write one row per Source 1 id, ids comma-joined, de-duplicated, order kept.

    Every id in s1_ids gets a row even if it has no matches (empty cell).
    """
    rows = []
    for sid in s1_ids:
        seen, ids = set(), []
        for x in mapping.get(sid, []):
            if x not in seen:
                seen.add(x)
                ids.append(x)
        rows.append((sid, ",".join(ids)))
    out = pd.DataFrame(rows, columns=["source1_entity_id", value_col])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False)
    return out
