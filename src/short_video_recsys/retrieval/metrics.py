"""Retrieval metrics with explicit all-target and warm-target denominators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from short_video_recsys.retrieval.protocol import RetrievalPeriodData


@dataclass(frozen=True)
class RetrievalMetrics:
    queries: int
    targets: int
    warm_targets: int
    warm_target_fraction: float
    catalog_coverage_at_k: float
    metrics: dict[str, float]


def evaluate_recommendations(
    recommendations: Mapping[int, Sequence[int]],
    data: RetrievalPeriodData,
    ks: Sequence[int],
    catalog_size: int | None = None,
) -> RetrievalMetrics:
    """Evaluate macro Recall and HitRate without hiding unavailable/cold targets."""

    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    ordered_ks = sorted(set(ks))
    query_users = sorted(data.targets)
    if not query_users:
        raise ValueError("No evaluation queries contain novel targets")

    total_targets = sum(len(data.targets[user]) for user in query_users)
    total_warm_targets = sum(len(data.warm_targets[user]) for user in query_users)
    output = {}
    for k in ordered_ks:
        all_recalls = []
        warm_recalls = []
        hits = []
        for user_id in query_users:
            recommended = set(recommendations.get(user_id, ())[:k])
            targets = data.targets[user_id]
            warm = data.warm_targets[user_id]
            all_recalls.append(len(recommended.intersection(targets)) / len(targets))
            if warm:
                warm_recalls.append(len(recommended.intersection(warm)) / len(warm))
            hits.append(float(bool(recommended.intersection(targets))))
        output[f"recall_all@{k}"] = sum(all_recalls) / len(all_recalls)
        output[f"recall_warm@{k}"] = (
            sum(warm_recalls) / len(warm_recalls) if warm_recalls else 0.0
        )
        output[f"hitrate@{k}"] = sum(hits) / len(hits)

    largest_k = ordered_ks[-1]
    unique_recommended = {
        item
        for user_id in query_users
        for item in recommendations.get(user_id, ())[:largest_k]
    }
    if catalog_size is None:
        coverage = len(unique_recommended.intersection(data.catalog_items)) / len(
            data.catalog_items
        )
    else:
        if catalog_size <= 0:
            raise ValueError("catalog_size must be positive")
        coverage = len(unique_recommended) / catalog_size
    return RetrievalMetrics(
        queries=len(query_users),
        targets=total_targets,
        warm_targets=total_warm_targets,
        warm_target_fraction=total_warm_targets / total_targets,
        catalog_coverage_at_k=coverage,
        metrics=output,
    )
