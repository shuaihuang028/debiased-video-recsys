from __future__ import annotations

import pandas as pd
import pytest

from short_video_recsys.data.schema import DataValidationError, validate_interactions


def test_validate_interactions_returns_summary(interactions: pd.DataFrame) -> None:
    summary = validate_interactions(interactions)

    assert summary.rows == 4
    assert summary.min_date == 20220421
    assert summary.max_date == 20220424
    assert summary.users == 2
    assert summary.videos == 2


def test_validate_interactions_rejects_missing_column(interactions: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="duration_ms"):
        validate_interactions(interactions.drop(columns="duration_ms"))


def test_validate_interactions_rejects_non_binary_label(interactions: pd.DataFrame) -> None:
    interactions.loc[0, "is_click"] = 2

    with pytest.raises(DataValidationError, match="is_click"):
        validate_interactions(interactions)


def test_validate_interactions_rejects_invalid_calendar_date(interactions: pd.DataFrame) -> None:
    interactions.loc[0, "date"] = 20220231

    with pytest.raises(DataValidationError, match="YYYYMMDD"):
        validate_interactions(interactions)


def test_validate_interactions_reports_zero_duration_without_dropping_row(
    interactions: pd.DataFrame,
) -> None:
    interactions.loc[0, "duration_ms"] = 0

    summary = validate_interactions(interactions)

    assert summary.rows == 4
    assert summary.zero_duration_rows == 1
