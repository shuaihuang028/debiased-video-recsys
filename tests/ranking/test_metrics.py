from __future__ import annotations

import numpy as np
import pytest

from short_video_recsys.ranking.metrics import evaluate_scores


def test_perfect_scores_have_perfect_auc_and_ndcg() -> None:
    labels = np.array([1, 0, 1, 0], dtype=np.int8)
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    users = np.array([1, 1, 2, 2])
    days = np.array([10, 10, 10, 10])

    metrics = evaluate_scores(labels, scores, users, days)

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.user_day_ndcg_at_10 == pytest.approx(1.0)
    assert metrics.positive_rate == pytest.approx(0.5)


def test_auc_assigns_average_rank_to_tied_scores() -> None:
    metrics = evaluate_scores(
        np.array([1, 0], dtype=np.int8),
        np.array([0.5, 0.5]),
        np.array([1, 1]),
        np.array([10, 10]),
    )

    assert metrics.roc_auc == pytest.approx(0.5)
