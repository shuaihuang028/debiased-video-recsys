from __future__ import annotations

import pandas as pd
import pytest

from short_video_recsys.serving.snapshot import (
    _aggregate_query,
    _history_columns,
    _history_values,
)


def test_snapshot_history_transform_matches_training_prior() -> None:
    values = {}
    for window in ("hist", "1h", "1d", "7d"):
        values.update(
            {
                f"user_{window}_impression_count": [0, 10],
                f"user_{window}_click_count": [0, 4],
                f"user_{window}_like_count": [0, 2],
                f"user_{window}_long_view_count": [0, 3],
                f"user_{window}_watch_ratio_sum": [0, 5],
            }
        )
    frame = pd.DataFrame(values)

    transformed = _history_values(frame, "user")
    positions = {name: i for i, name in enumerate(_history_columns("user"))}

    assert transformed[0, positions["user_hist_ctr"]] == pytest.approx(1 / 11)
    assert transformed[1, positions["user_hist_ctr"]] == pytest.approx(5 / 21)
    assert transformed[1, positions["user_hist_mean_watch_ratio"]] == pytest.approx(0.5)
    assert transformed[1, positions["log1p_user_hist_impression_count"]] == pytest.approx(
        2.3978953
    )


def test_snapshot_query_uses_model_scope_not_entity_key_in_aliases() -> None:
    query = _aggregate_query("events", "user", "user_id", 10_000)

    assert "AS user_hist_impression_count" in query
    assert "AS user_id_hist_impression_count" not in query
