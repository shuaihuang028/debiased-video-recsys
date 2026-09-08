"""Strictly chronological Two-Tower examples and negative samples."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from short_video_recsys.training.item_features import load_item_feature_arrays


@dataclass(frozen=True)
class SamplingSummary:
    eligible_positive_events: int
    saved_examples: int
    examples_with_exposure_negative: int
    random_negative_future_violations: int


def _choose_random_negative(
    rng: np.random.Generator,
    eligible_items: np.ndarray,
    upper_bound: int,
    positive_item: int,
    history: deque[int],
) -> int:
    blocked = set(history)
    blocked.add(positive_item)
    for _ in range(20):
        candidate = int(eligible_items[rng.integers(0, upper_bound)])
        if candidate not in blocked:
            return candidate
    return int(eligible_items[rng.integers(0, upper_bound)])


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
    }
    positive_histories: dict[int, deque[int]] = {}
    exposure_negatives: dict[int, deque[int]] = {}
    rng = np.random.default_rng(seed)
    eligible_count = 0

    values = frame[columns].to_numpy()
    for user_raw, item_raw, time_raw, click, like, long_view in values:
        user = int(user_raw)
        item = int(item_raw)
        timestamp = int(time_raw)
        history = positive_histories.setdefault(user, deque(maxlen=history_length))
        negative_pool = exposure_negatives.setdefault(
            user, deque(maxlen=exposure_pool_size)
        )
        if int(long_view) == 1:
            if history:
                eligible_catalog_size = int(
                    np.searchsorted(eligible_times, timestamp, side="right")
                )
                if eligible_catalog_size > 1:
                    slot = eligible_count
                    eligible_count += 1
                    if slot >= max_examples:
                        replacement = int(rng.integers(0, eligible_count))
                        if replacement >= max_examples:
                            history.append(item)
                            continue
                        slot = replacement

                    history_values = list(history)
                    outputs["user_id"][slot] = user
                    outputs["positive_item"][slot] = item
                    outputs["random_negative_item"][slot] = _choose_random_negative(
                        rng,
                        eligible_items,
                        eligible_catalog_size,
                        item,
                        history,
                    )
                    outputs["time_ms"][slot] = timestamp
                    outputs["history_length"][slot] = len(history_values)
                    outputs["history_items"][slot, : len(history_values)] = history_values
                    outputs["history_items"][slot, len(history_values) :] = -1
                    outputs["exposure_negative_item"][slot] = -1
                    valid_exposure = [
                        candidate
                        for candidate in negative_pool
                        if candidate != item and candidate not in history
                    ]
                    if valid_exposure:
                        outputs["exposure_negative_item"][slot] = int(
                            valid_exposure[rng.integers(0, len(valid_exposure))]
                        )
            history.append(item)
        elif int(click) == 0 and int(like) == 0:
            negative_pool.append(item)

    saved_examples = min(eligible_count, max_examples)
    trimmed = {name: value[:saved_examples] for name, value in outputs.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **trimmed)

    random_upload_times = arrays["upload_time_ms"][trimmed["random_negative_item"]]
    future_violations = int((random_upload_times > trimmed["time_ms"]).sum())
    summary = SamplingSummary(
        eligible_positive_events=eligible_count,
        saved_examples=saved_examples,
        examples_with_exposure_negative=int((trimmed["exposure_negative_item"] >= 0).sum()),
        random_negative_future_violations=future_violations,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
