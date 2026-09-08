from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from short_video_recsys.training.item_features import (
    load_item_feature_arrays,
    prepare_item_features,
)
from short_video_recsys.training.sampling import build_training_examples


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
