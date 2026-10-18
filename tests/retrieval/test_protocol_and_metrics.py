from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from short_video_recsys.retrieval.metrics import evaluate_recommendations
from short_video_recsys.retrieval.popularity import fit_popularity, recommend_popular
from short_video_recsys.retrieval.protocol import build_retrieval_period

DAY_MS = 86_400_000


def _write(path: Path, rows: list[tuple[int, int, int, int]]) -> None:
    pd.DataFrame(rows, columns=["user_id", "video_id", "time_ms", "long_view"]).to_parquet(
        path, index=False
    )


def test_protocol_retains_cold_targets_in_all_target_denominator(tmp_path: Path) -> None:
    fit = tmp_path / "fit.parquet"
    evaluation = tmp_path / "evaluation.parquet"
    _write(fit, [(1, 10, 1, 1), (1, 11, 2, 1), (2, 10, 3, 1), (2, 12, 4, 0)])
    _write(
        evaluation,
        [
            (1, 11, DAY_MS, 1),
            (1, 12, DAY_MS + 1, 1),
            (1, 99, 2 * DAY_MS, 1),
            (2, 11, DAY_MS, 1),
        ],
    )

    data = build_retrieval_period([fit], evaluation, history_size=1)

    assert data.query_users == {0: 1, 1: 1, 2: 2}
    assert data.query_times == {0: DAY_MS + 1, 1: 2 * DAY_MS, 2: DAY_MS}
    assert data.histories == {0: [11], 1: [12], 2: [10]}
    assert data.warm_histories == {0: [11], 1: [12], 2: [10]}
    assert data.targets == {0: {12}, 1: {99}, 2: {11}}
    assert data.warm_targets == {0: {12}, 1: set(), 2: {11}}
    assert data.catalog_items == {10, 11, 12}


def test_popularity_and_metrics_are_deterministic(tmp_path: Path) -> None:
    fit = tmp_path / "fit.parquet"
    evaluation = tmp_path / "evaluation.parquet"
    _write(fit, [(1, 10, 1, 1), (1, 11, 2, 1), (2, 10, 3, 1), (2, 12, 4, 0)])
    _write(evaluation, [(1, 12, DAY_MS, 1), (2, 11, DAY_MS, 1)])
    data = build_retrieval_period([fit], evaluation, history_size=2)

    popularity = fit_popularity([fit])
    recommendations = recommend_popular(popularity, data.seen_items, sorted(data.targets), k=2)
    metrics = evaluate_recommendations(recommendations, data, ks=[1, 2])

    assert popularity == [10, 11]
    assert recommendations == {0: [], 1: [11]}
    assert metrics.queries == 2
    assert metrics.metrics["recall_all@2"] == pytest.approx(0.5)
    assert metrics.metrics["hitrate@2"] == pytest.approx(0.5)


def test_protocol_excludes_all_simultaneous_events_from_query_history(tmp_path: Path) -> None:
    fit = tmp_path / "fit.parquet"
    evaluation = tmp_path / "evaluation.parquet"
    _write(fit, [(1, 10, 1, 1), (1, 11, 2, 1)])
    _write(evaluation, [
        (1, 10, DAY_MS, 1), (1, 12, DAY_MS, 1), (1, 13, DAY_MS, 1),
        (1, 14, 2 * DAY_MS, 1),
    ])
    data = build_retrieval_period([fit], evaluation, history_size=3)
    assert data.targets == {0: {12}, 1: {14}}
    assert data.histories == {0: [11, 10], 1: [13, 12, 10]}


def test_protocol_rejects_overlapping_fit_and_evaluation_periods(tmp_path: Path) -> None:
    fit = tmp_path / "fit.parquet"
    evaluation = tmp_path / "evaluation.parquet"
    _write(fit, [(1, 10, 100, 1)])
    _write(evaluation, [(1, 12, 100, 1)])
    with pytest.raises(ValueError, match="strictly precede"):
        build_retrieval_period([fit], evaluation)
