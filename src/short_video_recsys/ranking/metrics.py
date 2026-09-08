"""Dependency-light binary and grouped ranking metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RankingMetrics:
    rows: int
    positives: int
    positive_rate: float
    roc_auc: float
    log_loss: float
    brier_score: float
    user_day_ndcg_at_10: float
    user_days_with_positive: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both positive and negative labels")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    _, starts, counts = np.unique(
        sorted_scores, return_index=True, return_counts=True
    )
    positive_counts = np.add.reduceat(sorted_labels, starts)
    average_ranks = starts + (counts + 1) / 2.0
    positive_rank_sum = float(np.dot(positive_counts, average_ranks))
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def _grouped_ndcg_at_k(
    labels: np.ndarray, scores: np.ndarray, groups: np.ndarray, k: int
) -> tuple[float, int]:
    if k <= 0:
        raise ValueError("k must be positive")
    order = np.lexsort((-scores, groups))
    ordered_groups = groups[order]
    ordered_labels = labels[order]
    starts = np.flatnonzero(
        np.concatenate(([True], ordered_groups[1:] != ordered_groups[:-1]))
    )
    counts = np.diff(np.append(starts, len(labels)))
    ranks = np.arange(len(labels)) - np.repeat(starts, counts)
    gains = np.where(ranks < k, ordered_labels / np.log2(ranks + 2), 0.0)
    dcg = np.add.reduceat(gains, starts)
    positives = np.add.reduceat(ordered_labels, starts).astype(np.int64)
    discounts = 1 / np.log2(np.arange(k) + 2)
    ideal_lookup = np.concatenate(([0.0], np.cumsum(discounts)))
    ideal = ideal_lookup[np.minimum(positives, k)]
    valid = positives > 0
    return float(np.mean(dcg[valid] / ideal[valid])), int(valid.sum())


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    user_ids: np.ndarray,
    local_days: np.ndarray,
) -> RankingMetrics:
    """Evaluate impression probabilities and ranking within each user-local-day."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    user_ids = np.asarray(user_ids, dtype=np.int64)
    local_days = np.asarray(local_days, dtype=np.int64)
    if not (len(labels) == len(scores) == len(user_ids) == len(local_days)):
        raise ValueError("Metric inputs must have equal lengths")
    if len(labels) == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be a non-empty binary array")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    log_loss = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    groups = user_ids * 100_000 + local_days
    ndcg, valid_groups = _grouped_ndcg_at_k(labels, scores, groups, k=10)
    positives = int(labels.sum())
    return RankingMetrics(
        rows=len(labels),
        positives=positives,
        positive_rate=positives / len(labels),
        roc_auc=_roc_auc(labels, scores),
        log_loss=float(log_loss),
        brier_score=float(np.mean((scores - labels) ** 2)),
        user_day_ndcg_at_10=ndcg,
        user_days_with_positive=valid_groups,
    )
