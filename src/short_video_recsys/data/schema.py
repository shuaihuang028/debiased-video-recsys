"""Schema contract for KuaiRand interaction logs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_INTERACTION_COLUMNS: tuple[str, ...] = (
    "user_id",
    "video_id",
    "date",
    "time_ms",
    "is_click",
    "is_like",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "is_rand",
)

BINARY_INTERACTION_COLUMNS: tuple[str, ...] = (
    "is_click",
    "is_like",
    "long_view",
    "is_rand",
)

DAY_MS = 24 * 60 * 60 * 1_000
DATASET_UTC_OFFSET_MS = 8 * 60 * 60 * 1_000


class DataValidationError(ValueError):
    """Raised when raw interactions violate the declared data contract."""


@dataclass(frozen=True)
class ValidationSummary:
    """Compact validation output suitable for structured logging."""

    rows: int
    min_date: int
    max_date: int
    users: int | None
    videos: int | None
    zero_duration_rows: int
    timestamp_date_mismatch_rows: int


def _missing_columns(columns: Iterable[str]) -> tuple[str, ...]:
    available = set(columns)
    return tuple(name for name in REQUIRED_INTERACTION_COLUMNS if name not in available)


def validate_interactions(frame: pd.DataFrame) -> ValidationSummary:
    """Validate fields needed by retrieval, ranking, and PIT feature generation.

    The function deliberately does not require the input to be sorted: raw KuaiRand files are
    normalized and sorted only after validation.
    """

    missing = _missing_columns(frame.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise DataValidationError("Interaction data is empty")

    identifier_columns = ["user_id", "video_id", "date", "time_ms"]
    null_counts = frame[identifier_columns].isna().sum()
    invalid_nulls = null_counts[null_counts > 0]
    if not invalid_nulls.empty:
        raise DataValidationError(f"Null identifiers/timestamps: {invalid_nulls.to_dict()}")

    for column in BINARY_INTERACTION_COLUMNS:
        values = set(frame[column].dropna().unique().tolist())
        invalid = values.difference({0, 1})
        if invalid:
            raise DataValidationError(f"{column} contains non-binary values: {sorted(invalid)}")

    if (frame["time_ms"] < 0).any():
        raise DataValidationError("time_ms must be non-negative")
    if (frame["play_time_ms"].fillna(0) < 0).any():
        raise DataValidationError("play_time_ms must be non-negative")
    if (frame["duration_ms"].dropna() < 0).any():
        raise DataValidationError("Non-null duration_ms values must be non-negative")

    dates = pd.to_numeric(frame["date"], errors="coerce")
    if dates.isna().any():
        raise DataValidationError("date must contain YYYYMMDD integers")
    parsed_dates = pd.to_datetime(
        dates.astype("int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    if parsed_dates.isna().any():
        raise DataValidationError("date contains invalid YYYYMMDD values")

    date_days = parsed_dates.astype("int64") // (DAY_MS * 1_000_000)
    timestamp_days = (
        pd.to_numeric(frame["time_ms"]).astype("int64") + DATASET_UTC_OFFSET_MS
    ) // DAY_MS

    return ValidationSummary(
        rows=len(frame),
        min_date=int(dates.min()),
        max_date=int(dates.max()),
        users=int(frame["user_id"].nunique()),
        videos=int(frame["video_id"].nunique()),
        zero_duration_rows=int(frame["duration_ms"].eq(0).sum()),
        timestamp_date_mismatch_rows=int((date_days != timestamp_days).sum()),
    )


def validate_csv(path: Path, chunksize: int = 500_000) -> ValidationSummary:
    """Validate a large CSV incrementally and combine its summaries."""

    summaries = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        summaries.append(validate_interactions(chunk))
    if not summaries:
        raise DataValidationError(f"No rows found in {path}")

    # Exact user/item cardinalities require retaining millions of IDs. The incremental validator
    # intentionally omits them rather than reporting a misleading sum across chunks.
    return ValidationSummary(
        rows=sum(item.rows for item in summaries),
        min_date=min(item.min_date for item in summaries),
        max_date=max(item.max_date for item in summaries),
        users=None,
        videos=None,
        zero_duration_rows=sum(item.zero_duration_rows for item in summaries),
        timestamp_date_mismatch_rows=sum(
            item.timestamp_date_mismatch_rows for item in summaries
        ),
    )
