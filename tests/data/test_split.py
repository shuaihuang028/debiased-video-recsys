from __future__ import annotations

import pandas as pd
import pytest

from short_video_recsys.data.schema import DataValidationError
from short_video_recsys.data.split import TimeBoundaries, temporal_split


def test_temporal_split_is_sorted_disjoint_and_complete(interactions: pd.DataFrame) -> None:
    parts = temporal_split(interactions, TimeBoundaries(1_000, 2_000, 4_000))

    assert {name: len(frame) for name, frame in parts.items()} == {
        "train": 1,
        "validation": 1,
        "test": 2,
    }
    assert parts["train"].time_ms.max() < parts["validation"].time_ms.min()
    assert parts["validation"].time_ms.max() < parts["test"].time_ms.min()
    assert sum(len(frame) for frame in parts.values()) == len(interactions)


def test_temporal_split_rejects_rows_after_final_boundary(interactions: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="exceed"):
        temporal_split(interactions, TimeBoundaries(1_000, 2_000, 3_000))


def test_boundaries_must_be_increasing() -> None:
    with pytest.raises(ValueError, match="increasing"):
        TimeBoundaries(1_000, 1_000, 4_000)
