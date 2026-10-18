"""Strictly chronological Two-Tower examples and negative samples."""

from __future__ import annotations

import json
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from short_video_recsys.features.recent_history import (
    HISTORY_POLICY,
    TIMESTAMP_POLICY,
    recent_item_ids,
    touch_recent_item,
)
from short_video_recsys.training.item_features import load_item_feature_arrays


@dataclass(frozen=True)
class SamplingSummary:
    eligible_positive_events: int
    saved_examples: int
    examples_with_exposure_negative: int
    random_negative_future_violations: int
    non_prior_history_examples: int
    duplicate_history_examples: int
    random_negative_positive_collisions: int
    random_negative_history_collisions: int
    non_prior_exposure_negatives: int
    history_policy: str = HISTORY_POLICY
    timestamp_policy: str = TIMESTAMP_POLICY


def _choose_random_negative(
    rng: np.random.Generator,
    eligible_items: np.ndarray,
    upper_bound: int,
    positive_item: int,
    history: list[int],
) -> int | None:
    blocked = set(history)
    blocked.add(positive_item)
    for _ in range(20):
        candidate = int(eligible_items[rng.integers(0, upper_bound)])
        if candidate not in blocked:
            return candidate
    remaining = eligible_items[:upper_bound]
    remaining = remaining[~np.isin(remaining, list(blocked))]
    return int(rng.choice(remaining)) if len(remaining) else None


def build_training_examples(
    interactions_path: Path,
    item_feature_dir: Path,
    output_path: Path,
    summary_path: Path,
    seed: int,
    history_length: int,
    exposure_pool_size: int,
    max_examples: int,
) -> SamplingSummary:
    """Reservoir-sample positives while preserving true prior histories and negatives."""

    if min(history_length, exposure_pool_size, max_examples) <= 0:
        raise ValueError("Sampling sizes must be positive")
    arrays = load_item_feature_arrays(item_feature_dir)
    valid_catalog = arrays["present"] & (arrays["upload_time_ms"] > 0)
    eligible_items = np.flatnonzero(valid_catalog)
    order = np.argsort(arrays["upload_time_ms"][eligible_items], kind="stable")
    eligible_items = eligible_items[order]
    eligible_times = np.asarray(arrays["upload_time_ms"][eligible_items])

    columns = [
        "user_id",
        "video_id",
        "time_ms",
        "is_click",
        "is_like",
        "long_view",
    ]
    frame = pd.read_parquet(interactions_path, columns=columns).sort_values(
        ["time_ms", "user_id", "video_id"], kind="stable"
    )
    outputs = {
        "user_id": np.zeros(max_examples, dtype=np.int32),
        "positive_item": np.zeros(max_examples, dtype=np.int32),
        "random_negative_item": np.zeros(max_examples, dtype=np.int32),
        "exposure_negative_item": np.full(max_examples, -1, dtype=np.int32),
        "time_ms": np.zeros(max_examples, dtype=np.int64),
        "history_items": np.full((max_examples, history_length), -1, dtype=np.int32),
        "history_length": np.zeros(max_examples, dtype=np.int16),
        "history_max_time_ms": np.full(max_examples, -1, dtype=np.int64),
        "exposure_negative_time_ms": np.full(max_examples, -1, dtype=np.int64),
    }
    positive_histories: dict[int, OrderedDict[int, int]] = {}
    exposure_negatives: dict[int, deque[tuple[int, int]]] = {}
    rng = np.random.default_rng(seed)
    eligible_count = 0

    values = frame[columns].to_numpy()
    for (time_raw, user_raw), grouped in groupby(values, key=lambda row: (row[2], row[0])):
        timestamp = int(time_raw)
        user = int(user_raw)
        simultaneous = list(grouped)
        history = positive_histories.setdefault(user, OrderedDict())
        negative_pool = exposure_negatives.setdefault(user, deque(maxlen=exposure_pool_size))
        history_values = recent_item_ids(history, history_length)
        eligible_catalog_size = int(np.searchsorted(eligible_times, timestamp, side="left"))
        for _, item_raw, _, _, _, long_view in simultaneous:
            item = int(item_raw)
            if int(long_view) != 1 or not history_values or eligible_catalog_size <= 1:
                continue
            blocked = {*history_values, item}
            blocked_available = sum(
                bool(valid_catalog[candidate] and arrays["upload_time_ms"][candidate] < timestamp)
                for candidate in blocked
            )
            if blocked_available >= eligible_catalog_size:
                continue
            slot = eligible_count
            eligible_count += 1
            if slot >= max_examples:
                slot = int(rng.integers(0, eligible_count))
                if slot >= max_examples:
                    continue
            negative = _choose_random_negative(
                rng,
                eligible_items,
                eligible_catalog_size,
                item,
                history_values,
            )
            if negative is None:
                raise RuntimeError("No eligible negative despite the catalog availability check")
            outputs["user_id"][slot] = user
            outputs["positive_item"][slot] = item
            outputs["random_negative_item"][slot] = negative
            outputs["time_ms"][slot] = timestamp
            outputs["history_length"][slot] = len(history_values)
            outputs["history_items"][slot, : len(history_values)] = history_values
            outputs["history_items"][slot, len(history_values) :] = -1
            outputs["exposure_negative_item"][slot] = -1
            outputs["history_max_time_ms"][slot] = max(history.values())
            outputs["exposure_negative_time_ms"][slot] = -1
            valid_exposure = [
                (candidate, exposure_time)
                for candidate, exposure_time in negative_pool
                if candidate != item
                and candidate not in history
                and valid_catalog[candidate]
                and arrays["upload_time_ms"][candidate] < timestamp
            ]
            if valid_exposure:
                negative_item, exposure_time = valid_exposure[rng.integers(0, len(valid_exposure))]
                outputs["exposure_negative_item"][slot] = negative_item
                outputs["exposure_negative_time_ms"][slot] = exposure_time
        # Update both pools only after every example for this user/timestamp is built.
        for _, item_raw, _, click, like, long_view in simultaneous:
            item = int(item_raw)
            if int(long_view) == 1:
                touch_recent_item(history, item, timestamp, limit=history_length)
            elif int(click) == 0 and int(like) == 0:
                negative_pool.append((item, timestamp))

    saved_examples = min(eligible_count, max_examples)
    trimmed = {name: value[:saved_examples] for name, value in outputs.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **trimmed)

    random_upload_times = arrays["upload_time_ms"][trimmed["random_negative_item"]]
    future_violations = int((random_upload_times >= trimmed["time_ms"]).sum())
    exposures = trimmed["exposure_negative_item"] >= 0
    summary = SamplingSummary(
        eligible_positive_events=eligible_count,
        saved_examples=saved_examples,
        examples_with_exposure_negative=int((trimmed["exposure_negative_item"] >= 0).sum()),
        random_negative_future_violations=future_violations,
        non_prior_history_examples=int(
            (trimmed["history_max_time_ms"] >= trimmed["time_ms"]).sum()
        ),
        duplicate_history_examples=sum(
            len(row[row >= 0]) != len(set(row[row >= 0].tolist()))
            for row in trimmed["history_items"]
        ),
        random_negative_positive_collisions=int(
            (trimmed["random_negative_item"] == trimmed["positive_item"]).sum()
        ),
        random_negative_history_collisions=int(
            (trimmed["history_items"] == trimmed["random_negative_item"][:, None]).any(axis=1).sum()
        ),
        non_prior_exposure_negatives=int(
            (trimmed["exposure_negative_time_ms"][exposures] >= trimmed["time_ms"][exposures]).sum()
        ),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
