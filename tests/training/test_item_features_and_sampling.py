from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from short_video_recsys.retrieval.protocol import build_retrieval_period
from short_video_recsys.serving.snapshot import _recent_histories
from short_video_recsys.training.item_features import (
    load_item_feature_arrays,
    prepare_item_features,
)
from short_video_recsys.training.sampling import _choose_random_negative, build_training_examples


def _metadata(path: Path) -> None:
    pd.DataFrame(
        {
            "video_id": [0, 1, 2],
            "author_id": [10, 11, 12],
            "video_type": ["NORMAL", "NORMAL", "AD"],
            "upload_dt": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "upload_type": ["Web", "ShortImport", "Web"],
            "video_duration": [10_000, None, 20_000],
            "server_width": [720, 720, 1080],
            "server_height": [1280, 1280, 1920],
            "music_id": [100, 101, None],
            "music_type": [4, 9, None],
            "tag": ["1,2", "3", None],
        }
    ).to_csv(path, index=False)


def test_item_features_and_sampling_are_time_safe(tmp_path: Path) -> None:
    metadata_path = tmp_path / "items.csv"
    feature_dir = tmp_path / "item_features"
    _metadata(metadata_path)
    item_summary = prepare_item_features(
        metadata_path,
        feature_dir,
        author_buckets=16,
        music_buckets=16,
        max_tags=3,
        chunksize=2,
    )
    arrays = load_item_feature_arrays(feature_dir)

    assert item_summary.catalog_rows == 3
    assert arrays["tags"][0].tolist() == [1, 2, 0]
    assert arrays["duration_log_seconds"][1] == 0
    assert arrays["author_bucket"][0] > 0

    base_time = 1_650_000_000_000
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "video_id": [0, 1, 0, 2],
            "time_ms": [base_time + offset for offset in range(4)],
            "is_click": [0, 1, 0, 1],
            "is_like": [0, 0, 0, 0],
            "long_view": [0, 1, 0, 1],
        }
    )
    interactions_path = tmp_path / "interactions.parquet"
    output_path = tmp_path / "examples.npz"
    summary_path = tmp_path / "summary.json"
    interactions.to_parquet(interactions_path, index=False)
    sampling_summary = build_training_examples(
        interactions_path,
        feature_dir,
        output_path,
        summary_path,
        seed=7,
        history_length=2,
        exposure_pool_size=5,
        max_examples=10,
    )
    examples = np.load(output_path)

    assert sampling_summary.saved_examples == 1
    assert sampling_summary.random_negative_future_violations == 0
    assert examples["positive_item"].tolist() == [2]
    assert examples["history_items"][0].tolist() == [1, -1]
    assert examples["exposure_negative_item"].tolist() == [0]


def test_training_retrieval_and_snapshot_share_strict_unique_histories(tmp_path: Path) -> None:
    metadata_path = tmp_path / "items.csv"
    _metadata(metadata_path)
    metadata = pd.read_csv(metadata_path)
    extra = metadata.copy()
    extra["video_id"] += 3
    pd.concat([metadata, extra], ignore_index=True).to_csv(metadata_path, index=False)
    feature_dir = tmp_path / "features"
    prepare_item_features(metadata_path, feature_dir, 16, 16, 5)
    base = 1_650_000_000_000
    events = pd.DataFrame(
        [
            (1, 0, base, 1, 0, 1),
            (1, 1, base + 1, 1, 0, 1),
            (1, 0, base + 2, 1, 0, 1),
            (1, 2, base + 3, 1, 0, 1),
            (1, 3, base + 3, 1, 0, 1),
            (1, 4, base + 3, 0, 0, 0),
            (1, 5, base + 86_400_003, 1, 0, 1),
        ],
        columns=["user_id", "video_id", "time_ms", "is_click", "is_like", "long_view"],
    )
    input_path = tmp_path / "events.parquet"
    events.to_parquet(input_path, index=False)
    output_path = tmp_path / "examples.npz"
    summary = build_training_examples(
        input_path,
        feature_dir,
        output_path,
        tmp_path / "summary.json",
        7,
        3,
        10,
        100,
    )
    examples = np.load(output_path)
    simultaneous = examples["time_ms"] == base + 3
    assert examples["positive_item"][simultaneous].tolist() == [2, 3]
    assert examples["history_items"][simultaneous].tolist() == [[0, 1, -1], [0, 1, -1]]
    assert examples["exposure_negative_item"][simultaneous].tolist() == [-1, -1]
    assert examples["exposure_negative_item"][-1] == 4
    assert (examples["history_max_time_ms"] < examples["time_ms"]).all()
    assert summary.duplicate_history_examples == 0
    assert summary.non_prior_history_examples == 0
    assert summary.non_prior_exposure_negatives == 0
    assert summary.random_negative_positive_collisions == 0
    assert summary.random_negative_history_collisions == 0

    fit_path, evaluation_path = tmp_path / "fit.parquet", tmp_path / "evaluation.parquet"
    events[events["time_ms"] < base + 3].to_parquet(fit_path, index=False)
    events[events["time_ms"] >= base + 3].to_parquet(evaluation_path, index=False)
    data = build_retrieval_period([fit_path], evaluation_path, history_size=3)
    with duckdb.connect() as connection:
        for query_id, timestamp in data.query_times.items():
            target = next(iter(data.targets[query_id]))
            index = np.flatnonzero(
                (examples["time_ms"] == timestamp) & (examples["positive_item"] == target)
            )[0]
            training_history = examples["history_items"][index]
            assert training_history[training_history >= 0].tolist() == data.histories[query_id]
            snapshot = _recent_histories(
                connection,
                f"read_parquet('{input_path}')",
                timestamp,
                1,
                3,
            )
            assert snapshot[1].tolist() == training_history.tolist()


def test_negative_sampling_fallback_never_returns_a_blocked_item() -> None:
    class RepeatedDraw:
        def integers(self, *args):
            return 0

        def choice(self, values):
            return values[0]

    rng = RepeatedDraw()
    assert _choose_random_negative(rng, np.array([0, 1, 2]), 3, 0, [1]) == 2
    assert _choose_random_negative(rng, np.array([0, 1]), 2, 0, [1]) is None
