"""Point-in-time historical features with strict timestamp semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from short_video_recsys.data.schema import DataValidationError

DEFAULT_WINDOWS_MS: dict[str, int] = {
    "1h": 60 * 60 * 1_000,
    "1d": 24 * 60 * 60 * 1_000,
    "7d": 7 * 24 * 60 * 60 * 1_000,
}

EVENT_VALUE_COLUMNS = (
    "impression_value",
    "click_value",
    "like_value",
    "long_view_value",
    "watch_ratio_value",
)


def _event_values(frame: pd.DataFrame, watch_ratio_cap: float) -> pd.DataFrame:
    values = pd.DataFrame(index=frame.index)
    values["impression_value"] = 1
    values["click_value"] = frame["is_click"].fillna(0).astype("int64")
    values["like_value"] = frame["is_like"].fillna(0).astype("int64")
    values["long_view_value"] = frame["long_view"].fillna(0).astype("int64")
    duration = frame["duration_ms"].where(frame["duration_ms"] > 0)
    ratio = frame["play_time_ms"].fillna(0).div(duration).fillna(0)
    values["watch_ratio_value"] = ratio.clip(lower=0, upper=watch_ratio_cap)
    return values


def _strict_window_sums(times: np.ndarray, values: np.ndarray, window_ms: int) -> np.ndarray:
    """Sum values in [timestamp-window, timestamp), excluding all current-time rows."""

    left = np.searchsorted(times, times - window_ms, side="left")
    right = np.arange(len(times))
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return prefix[right] - prefix[left]


def _entity_features(
    working: pd.DataFrame,
    entity_column: str,
    prefix: str,
    windows_ms: Mapping[str, int],
) -> pd.DataFrame:
    # Aggregate equal timestamps first. This is essential: an ordinary groupby.cumsum().shift(1)
    # leaks labels between rows that share the prediction timestamp.
    grouped = (
        working.groupby([entity_column, "time_ms"], sort=True, as_index=False)[
            list(EVENT_VALUE_COLUMNS)
        ]
        .sum()
        .sort_values([entity_column, "time_ms"], kind="mergesort")
        .reset_index(drop=True)
    )

    entity_group = grouped.groupby(entity_column, sort=False)
    for value_column in EVENT_VALUE_COLUMNS:
        current = grouped[value_column]
        grouped[f"{prefix}_hist_{value_column}"] = (
            entity_group[value_column].cumsum() - current
        )

    for label, window_ms in windows_ms.items():
        if window_ms <= 0:
            raise ValueError(f"Window {label} must be positive")
        for value_column in EVENT_VALUE_COLUMNS:
            output = np.zeros(len(grouped), dtype=np.float64)
            for positions in entity_group.indices.values():
                positions_array = np.asarray(positions)
                timestamps = grouped.iloc[positions_array]["time_ms"].to_numpy(dtype=np.int64)
                values = grouped.iloc[positions_array][value_column].to_numpy(dtype=np.float64)
                output[positions_array] = _strict_window_sums(timestamps, values, window_ms)
            grouped[f"{prefix}_{label}_{value_column}"] = output

    keys_and_current_values = {entity_column, "time_ms", *EVENT_VALUE_COLUMNS}
    feature_columns = [
        column for column in grouped.columns if column not in keys_and_current_values
    ]
    return grouped[[entity_column, "time_ms", *feature_columns]]


def _rename_and_add_rates(
    frame: pd.DataFrame, prefix: str, periods: Sequence[str], alpha: float, beta: float
) -> pd.DataFrame:
    rename_suffixes = {
        "impression_value": "impression_count",
        "click_value": "click_count",
        "like_value": "like_count",
        "long_view_value": "long_view_count",
        "watch_ratio_value": "watch_ratio_sum",
    }
    frame = frame.rename(
        columns={
            f"{prefix}_{period}_{old}": f"{prefix}_{period}_{new}"
            for period in periods
            for old, new in rename_suffixes.items()
        }
    )
    for period in periods:
        impressions = frame[f"{prefix}_{period}_impression_count"]
        denominator = impressions + alpha + beta
        frame[f"{prefix}_{period}_ctr"] = (
            frame[f"{prefix}_{period}_click_count"] + alpha
        ) / denominator
        frame[f"{prefix}_{period}_like_rate"] = (
            frame[f"{prefix}_{period}_like_count"] + alpha
        ) / denominator
        frame[f"{prefix}_{period}_long_view_rate"] = (
            frame[f"{prefix}_{period}_long_view_count"] + alpha
        ) / denominator
        frame[f"{prefix}_{period}_mean_watch_ratio"] = np.divide(
            frame[f"{prefix}_{period}_watch_ratio_sum"],
            impressions,
            out=np.zeros(len(frame), dtype=np.float64),
            where=impressions.to_numpy() > 0,
        )
    return frame


def build_point_in_time_features(
    interactions: pd.DataFrame,
    windows_ms: Mapping[str, int] = DEFAULT_WINDOWS_MS,
    entities: Sequence[str] = ("user_id", "video_id"),
    watch_ratio_cap: float = 5.0,
    smoothing_alpha: float = 1.0,
    smoothing_beta: float = 10.0,
) -> pd.DataFrame:
    """Add expanding and rolling historical features to each interaction.

    Labels at time ``t`` are never visible to features at time ``t``. The returned order and
    index match the input, allowing callers to append labels or persist training examples safely.
    """

    required = {
        "user_id",
        "video_id",
        "time_ms",
        "is_click",
        "is_like",
        "long_view",
        "play_time_ms",
        "duration_ms",
    }
    missing = sorted(required.difference(interactions.columns))
    if missing:
        raise DataValidationError(f"Missing PIT feature columns: {', '.join(missing)}")
    if interactions.empty:
        raise DataValidationError("Cannot build features from an empty frame")
    if interactions["time_ms"].isna().any():
        raise DataValidationError("time_ms cannot be null")
    if watch_ratio_cap <= 0:
        raise ValueError("watch_ratio_cap must be positive")
    if smoothing_alpha < 0 or smoothing_beta < 0:
        raise ValueError("Smoothing parameters must be non-negative")

    working = interactions.copy()
    working["_row_order"] = np.arange(len(working))
    values = _event_values(working, watch_ratio_cap)
    for column in EVENT_VALUE_COLUMNS:
        working[column] = values[column]

    periods = ["hist", *windows_ms.keys()]
    result = working
    prefix_by_entity = {"user_id": "user", "video_id": "item"}
    for entity_column in entities:
        if entity_column not in prefix_by_entity:
            raise ValueError(f"Unsupported entity column: {entity_column}")
        prefix = prefix_by_entity[entity_column]
        features = _entity_features(working, entity_column, prefix, windows_ms)
        result = result.merge(
            features,
            on=[entity_column, "time_ms"],
            how="left",
            validate="many_to_one",
        )
        result = _rename_and_add_rates(
            result, prefix, periods, smoothing_alpha, smoothing_beta
        )

    result = result.sort_values("_row_order", kind="mergesort")
    result.index = interactions.index
    drop_columns = ["_row_order", *EVENT_VALUE_COLUMNS]
    return result.drop(columns=drop_columns)
