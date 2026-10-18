"""Single logged-target metrics with retrieval failures retained as zero scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _wilson_interval(hits: int, queries: int) -> list[float] | None:
    if queries == 0:
        return None
    z = 1.959963984540054
    probability = hits / queries
    denominator = 1 + z * z / queries
    center = (probability + z * z / (2 * queries)) / denominator
    radius = (
        z
        * np.sqrt(probability * (1 - probability) / queries + z * z / (4 * queries * queries))
        / denominator
    )
    return [float(max(0, center - radius)), float(min(1, center + radius))]


def _stage_metrics(ranks: np.ndarray, ks: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for k in ks:
        hit = (ranks > 0) & (ranks <= k)
        hits = int(hit.sum())
        output[f"hits@{k}"] = hits
        output[f"hitrate@{k}"] = hits / len(ranks) if len(ranks) else None
        output[f"hitrate@{k}_wilson_95_ci"] = _wilson_interval(hits, len(ranks))
        output[f"mrr@{k}"] = float(np.sum(1 / ranks[hit]) / len(ranks)) if len(ranks) else None
        output[f"ndcg@{k}"] = (
            float(np.sum(1 / np.log2(ranks[hit] + 1)) / len(ranks)) if len(ranks) else None
        )
    return output


def summarize_end_to_end(query_results: pd.DataFrame, ks: list[int]) -> dict[str, Any]:
    """Zero ranks denote absent targets; conditional metrics never replace full metrics."""
    if not ks or any(k <= 0 for k in ks) or query_results.empty:
        raise ValueError("Require queries and positive top-k cutoffs")
    output: dict[str, Any] = {}
    for group in ("all", "warm", "cold"):
        frame = (
            query_results
            if group == "all"
            else query_results[query_results["target_group"].eq(group)]
        )
        before = frame["retrieval_target_rank"].to_numpy(dtype=np.int64)
        after = frame["reranked_target_rank"].to_numpy(dtype=np.int64)
        if np.any((before == 0) != (after == 0)):
            raise ValueError("Reranking must preserve target membership in the candidate set")
        retrieved = before > 0
        hits = int(retrieved.sum())
        output[group] = {
            "queries": len(frame),
            "candidate_hits": hits,
            "candidate_recall": hits / len(frame) if len(frame) else None,
            "candidate_recall_wilson_95_ci": _wilson_interval(hits, len(frame)),
            "oracle_top_k_hitrate_ceiling": hits / len(frame) if len(frame) else None,
            "retrieval_order": _stage_metrics(before, ks),
            "lightgbm_reranked": _stage_metrics(after, ks),
            "conditional_on_retrieved": {
                "queries": hits,
                "retrieval_order": _stage_metrics(before[retrieved], ks),
                "lightgbm_reranked": _stage_metrics(after[retrieved], ks),
            },
        }
    return output
