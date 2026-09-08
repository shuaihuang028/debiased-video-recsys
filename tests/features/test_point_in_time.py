from __future__ import annotations

import pandas as pd
import pytest

from short_video_recsys.features.point_in_time import build_point_in_time_features


def _pit_input() -> pd.DataFrame:
    # Intentionally unsorted. Rows 1 and 2 share a timestamp and must not see each other.
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "video_id": [10, 11, 12, 10],
            "time_ms": [3_000, 1_000, 2_000, 2_000],
            "is_click": [0, 1, 0, 1],
            "is_like": [0, 1, 0, 0],
            "long_view": [0, 1, 0, 1],
            "play_time_ms": [1_000, 10_000, 0, 5_000],
            "duration_ms": [10_000, 10_000, 10_000, 10_000],
        },
        index=[40, 10, 30, 20],
    )


def test_features_use_only_strictly_earlier_events() -> None:
    output = build_point_in_time_features(_pit_input(), windows_ms={"window": 1_500})

    at_1000 = output.loc[10]
    at_2000_a = output.loc[30]
    at_2000_b = output.loc[20]
    at_3000 = output.loc[40]

    assert at_1000.user_hist_impression_count == 0
    assert at_2000_a.user_hist_impression_count == 1
    assert at_2000_b.user_hist_impression_count == 1
    assert at_2000_a.user_hist_click_count == 1
    assert at_2000_b.user_hist_click_count == 1
    assert at_3000.user_hist_impression_count == 3
    assert at_3000.user_hist_click_count == 2


def test_rolling_window_excludes_current_timestamp_and_old_events() -> None:
    output = build_point_in_time_features(_pit_input(), windows_ms={"window": 1_500})

    at_3000 = output.loc[40]
    assert at_3000.user_window_impression_count == 2
    assert at_3000.user_window_click_count == 1


def test_item_history_is_independent_from_user_history() -> None:
    output = build_point_in_time_features(_pit_input(), windows_ms={"window": 10_000})

    at_3000_item_10 = output.loc[40]
    assert at_3000_item_10.item_hist_impression_count == 1
    assert at_3000_item_10.item_hist_click_count == 1


def test_output_preserves_input_order_and_index() -> None:
    source = _pit_input()
    output = build_point_in_time_features(source, windows_ms={"window": 1_500})

    assert output.index.tolist() == source.index.tolist()
    assert output[["user_id", "video_id", "time_ms"]].equals(
        source[["user_id", "video_id", "time_ms"]]
    )


def test_smoothed_rate_uses_declared_prior() -> None:
    output = build_point_in_time_features(
        _pit_input(), windows_ms={}, smoothing_alpha=1.0, smoothing_beta=9.0
    )

    assert output.loc[10, "user_hist_ctr"] == pytest.approx(0.1)
    assert output.loc[30, "user_hist_ctr"] == pytest.approx(2.0 / 11.0)
