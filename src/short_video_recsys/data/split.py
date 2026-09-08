"""Deterministic chronological splits for standard and randomized logs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from short_video_recsys.data.schema import DataValidationError


@dataclass(frozen=True)
class TimeBoundaries:
    """Inclusive epoch-millisecond boundaries for three chronological partitions."""

    first_end_ms: int
    second_end_ms: int
    third_end_ms: int

    def __post_init__(self) -> None:
        if not self.first_end_ms < self.second_end_ms < self.third_end_ms:
            raise ValueError("Time boundaries must be strictly increasing")


def temporal_split(
    frame: pd.DataFrame,
    boundaries: TimeBoundaries,
    first_name: str = "train",
    second_name: str = "validation",
    third_name: str = "test",
    time_column: str = "time_ms",
) -> dict[str, pd.DataFrame]:
    """Split rows by event timestamp and assert strict chronological separation.

    Rows after the final boundary are rejected instead of silently discarded. Rows before the
    first observed date naturally belong to the first partition.
    """

    if time_column not in frame:
        raise DataValidationError(f"Required split column: {time_column}")
    if frame.empty:
        raise DataValidationError("Cannot split an empty interaction frame")
    if frame[time_column].isna().any():
        raise DataValidationError("Split column cannot contain nulls")
    if (frame[time_column] > boundaries.third_end_ms).any():
        max_time = int(frame[time_column].max())
        raise DataValidationError(
            f"Rows exceed configured final boundary {boundaries.third_end_ms}: max={max_time}"
        )

    ordered = frame.sort_values([time_column, "user_id", "video_id"], kind="mergesort")
    first = ordered[ordered[time_column] <= boundaries.first_end_ms].copy()
    second = ordered[
        (ordered[time_column] > boundaries.first_end_ms)
        & (ordered[time_column] <= boundaries.second_end_ms)
    ].copy()
    third = ordered[
        (ordered[time_column] > boundaries.second_end_ms)
        & (ordered[time_column] <= boundaries.third_end_ms)
    ].copy()

    partitions = {first_name: first, second_name: second, third_name: third}
    if any(part.empty for part in partitions.values()):
        sizes = {name: len(part) for name, part in partitions.items()}
        raise DataValidationError(f"Temporal split produced an empty partition: {sizes}")
    if sum(len(part) for part in partitions.values()) != len(frame):
        raise AssertionError("Temporal split lost rows")

    if not first[time_column].max() < second[time_column].min():
        raise DataValidationError("First and second partitions overlap in event time")
    if not second[time_column].max() < third[time_column].min():
        raise DataValidationError("Second and third partitions overlap in event time")

    return partitions
