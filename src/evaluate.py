"""Local scoring: exact macro F0.5 as defined by the organizers, plus blocking recall."""


def f05(pred, truth):
    """Per-entity F0.5. Singleton: 1.0 for an empty prediction, 0.0 otherwise."""
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, truth_map):
    """Average F0.5 over ALL Source 1 ids in truth_map (singletons included)."""
    return sum(f05(pred_map.get(k, []), v) for k, v in truth_map.items()) / len(truth_map)


def breakdown(pred_map, truth_map):
    """Macro F0.5 split into singleton / non-singleton S1 entities."""
    single = {k: v for k, v in truth_map.items() if not v}
    multi = {k: v for k, v in truth_map.items() if v}
    return {
        "macro_f05": macro_f05(pred_map, truth_map),
        "singletons_f05": macro_f05(pred_map, single) if single else None,
        "matched_f05": macro_f05(pred_map, multi) if multi else None,
        "n_singletons": len(single), "n_matched": len(multi),
    }


def blocking_recall(cand_map, truth_map):
    """Share of true (s1, match) pairs present in the candidate set, and mean list size."""
    total = hit = 0
    for k, v in truth_map.items():
        c = set(cand_map.get(k, []))
        total += len(v)
        hit += sum(1 for x in v if x in c)
    sizes = [len(cand_map.get(k, [])) for k in truth_map]
    return {"pair_recall": hit / max(total, 1), "mean_candidates": sum(sizes) / max(len(sizes), 1)}


if __name__ == "__main__":
    # organizers' worked example -> 0.714
    print(round(f05(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"]), 3))
