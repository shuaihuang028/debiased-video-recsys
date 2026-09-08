from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from short_video_recsys.data.split import TimeBoundaries
from short_video_recsys.features.duckdb_pipeline import (
    build_pit_parquet,
    build_random_pit_parquet,
)
from short_video_recsys.features.point_in_time import build_point_in_time_features


def test_duckdb_pipeline_matches_reference_semantics(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "video_id": [10, 11, 12, 10],
            "date": [20220421, 20220421, 20220422, 20220423],
            "time_ms": [3_000, 1_000, 2_000, 2_000],
            "is_click": [0, 1, 0, 1],
            "is_like": [0, 1, 0, 0],
            "long_view": [0, 1, 0, 1],
            "play_time_ms": [1_000, 10_000, 0, 5_000],
            "duration_ms": [10_000, 10_000, 10_000, 10_000],
            "is_rand": [0, 0, 0, 0],
        }
    )
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "features.parquet"
    source.to_csv(input_path, index=False)

    summary = build_pit_parquet(
        input_paths=[input_path],
        output_path=output_path,
        boundaries=TimeBoundaries(1_000, 2_000, 3_000),
        windows_ms={"window": 1_500},
        watch_ratio_cap=5.0,
        smoothing_alpha=1.0,
        smoothing_beta=9.0,
        temp_directory=tmp_path / "duckdb_tmp",
    )
    actual = pd.read_parquet(output_path).sort_values(["time_ms", "video_id"])
    expected = build_point_in_time_features(
        source,
        windows_ms={"window": 1_500},
        smoothing_alpha=1.0,
        smoothing_beta=9.0,
    ).sort_values(["time_ms", "video_id"])

    assert summary.rows == 4
    assert (summary.train_rows, summary.validation_rows, summary.test_rows) == (1, 2, 1)
    columns = [
        "user_hist_impression_count",
        "user_hist_click_count",
        "user_window_impression_count",
        "item_hist_impression_count",
        "user_hist_ctr",
        "item_window_mean_watch_ratio",
    ]
    for column in columns:
        assert actual[column].tolist() == pytest.approx(expected[column].tolist())


def test_random_features_keep_only_random_rows_with_prior_standard_history(
    tmp_path: Path,
) -> None:
    columns = {
        "user_id": [1, 1, 1],
        "video_id": [10, 11, 12],
        "date": [20220421] * 3,
        "time_ms": [1_000, 2_000, 3_000],
        "is_click": [1, 0, 1],
        "is_like": [0, 0, 0],
        "long_view": [1, 0, 1],
        "play_time_ms": [1_000] * 3,
        "duration_ms": [2_000] * 3,
        "is_rand": [0, 1, 1],
    }
    source = pd.DataFrame(columns)
    standard_path = tmp_path / "standard.parquet"
    random_path = tmp_path / "random.parquet"
    source[source.is_rand == 0].to_parquet(standard_path, index=False)
    source[source.is_rand == 1].to_parquet(random_path, index=False)
    output_path = tmp_path / "random_features.parquet"

    summary = build_random_pit_parquet(
        standard_paths=[standard_path],
        random_paths=[random_path],
        output_path=output_path,
        boundaries=TimeBoundaries(2_000, 2_500, 3_000),
        windows_ms={"window": 10_000},
        watch_ratio_cap=5.0,
        smoothing_alpha=1.0,
        smoothing_beta=9.0,
        temp_directory=tmp_path / "duckdb_random_tmp",
    )
    actual = pd.read_parquet(output_path).sort_values("time_ms")

    assert summary.rows == 2
    assert (summary.calibration_rows, summary.validation_rows, summary.test_rows) == (1, 0, 1)
    assert actual["user_hist_impression_count"].tolist() == [1, 2]
    assert actual["split"].tolist() == ["calibration", "test"]
