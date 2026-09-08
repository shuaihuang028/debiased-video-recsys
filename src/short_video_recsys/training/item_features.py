"""Compact, memory-mappable content features for the 4.37M item catalog."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ItemFeatureSummary:
    catalog_rows: int
    array_size: int
    missing_video_ids: int
    tag_vocabulary_size: int
    upload_type_vocabulary_size: int
    video_type_vocabulary_size: int
    music_type_vocabulary_size: int


def _mapping(values: list[object]) -> dict[str, int]:
    normalized = sorted({str(value) for value in values if pd.notna(value)})
    return {value: index + 1 for index, value in enumerate(normalized)}


def _bucket_numeric(series: pd.Series, buckets: int) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & (numeric >= 0)
    output = np.zeros(len(series), dtype=np.int32)
    output[valid.to_numpy()] = (
        numeric[valid].astype("int64").mod(buckets).to_numpy(dtype=np.int32) + 1
    )
    return output


def prepare_item_features(
    metadata_path: Path,
    output_dir: Path,
    author_buckets: int,
    music_buckets: int,
    max_tags: int,
    chunksize: int = 500_000,
) -> ItemFeatureSummary:
    """Convert CSV metadata into fixed-width arrays indexed directly by video_id."""

    if author_buckets <= 0 or music_buckets <= 0 or max_tags <= 0:
        raise ValueError("Bucket counts and max_tags must be positive")
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    escaped_path = str(metadata_path).replace("'", "''")
    source = f"read_csv_auto('{escaped_path}', header = true)"
    catalog_rows, max_video_id = connection.execute(
        f"SELECT COUNT(*), MAX(video_id) FROM {source}"
    ).fetchone()
    category_rows = connection.execute(
        f"SELECT DISTINCT upload_type, video_type, music_type FROM {source}"
    ).fetchall()
    connection.close()
    upload_mapping = _mapping([row[0] for row in category_rows])
    video_mapping = _mapping([row[1] for row in category_rows])
    music_type_mapping = _mapping([row[2] for row in category_rows])

    size = int(max_video_id) + 1
    arrays = {
        "present": np.zeros(size, dtype=np.bool_),
        "author_bucket": np.zeros(size, dtype=np.int32),
        "music_bucket": np.zeros(size, dtype=np.int32),
        "upload_type": np.zeros(size, dtype=np.int16),
        "video_type": np.zeros(size, dtype=np.int8),
        "music_type": np.zeros(size, dtype=np.int8),
        "tags": np.zeros((size, max_tags), dtype=np.int16),
        "duration_log_seconds": np.zeros(size, dtype=np.float32),
        "aspect_ratio": np.zeros(size, dtype=np.float32),
        "upload_time_ms": np.zeros(size, dtype=np.int64),
    }
    usecols = [
        "video_id",
        "author_id",
        "video_type",
        "upload_dt",
        "upload_type",
        "video_duration",
        "server_width",
        "server_height",
        "music_id",
        "music_type",
        "tag",
    ]
    for chunk in pd.read_csv(metadata_path, usecols=usecols, chunksize=chunksize):
        ids = chunk["video_id"].to_numpy(dtype=np.int64)
        arrays["present"][ids] = True
        arrays["author_bucket"][ids] = _bucket_numeric(chunk["author_id"], author_buckets)
        arrays["music_bucket"][ids] = _bucket_numeric(chunk["music_id"], music_buckets)
        arrays["upload_type"][ids] = (
            chunk["upload_type"].astype(str).map(upload_mapping).fillna(0).to_numpy(dtype=np.int16)
        )
        arrays["video_type"][ids] = (
            chunk["video_type"].astype(str).map(video_mapping).fillna(0).to_numpy(dtype=np.int8)
        )
        arrays["music_type"][ids] = (
            chunk["music_type"].astype(str).map(music_type_mapping).fillna(0).to_numpy(dtype=np.int8)
        )

        split_tags = chunk["tag"].fillna("").astype(str).str.split(",", expand=True)
        for tag_position in range(min(max_tags, split_tags.shape[1])):
            arrays["tags"][ids, tag_position] = (
                pd.to_numeric(split_tags[tag_position], errors="coerce")
                .fillna(0)
                .clip(lower=0)
                .to_numpy(dtype=np.int16)
            )

        duration_seconds = pd.to_numeric(chunk["video_duration"], errors="coerce").fillna(0)
        arrays["duration_log_seconds"][ids] = np.log1p(
            duration_seconds.clip(lower=0).to_numpy(dtype=np.float32) / 1_000.0
        )
        width = pd.to_numeric(chunk["server_width"], errors="coerce")
        height = pd.to_numeric(chunk["server_height"], errors="coerce")
        aspect = width.div(height.where(height > 0)).fillna(0).clip(lower=0, upper=10)
        arrays["aspect_ratio"][ids] = aspect.to_numpy(dtype=np.float32)

        upload_dates = pd.to_datetime(chunk["upload_dt"], errors="coerce")
        upload_ns = upload_dates.astype("int64").to_numpy()
        valid_dates = upload_dates.notna().to_numpy()
        upload_ms = np.zeros(len(chunk), dtype=np.int64)
        # Dataset dates are local UTC+8 midnights; convert naive epoch values accordingly.
        upload_ms[valid_dates] = upload_ns[valid_dates] // 1_000_000 - 8 * 60 * 60 * 1_000
        arrays["upload_time_ms"][ids] = upload_ms

    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array, allow_pickle=False)

    metadata = {
        "author_buckets": author_buckets,
        "music_buckets": music_buckets,
        "max_tags": max_tags,
        "upload_type_mapping": upload_mapping,
        "video_type_mapping": video_mapping,
        "music_type_mapping": music_type_mapping,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary = ItemFeatureSummary(
        catalog_rows=int(catalog_rows),
        array_size=size,
        missing_video_ids=size - int(catalog_rows),
        tag_vocabulary_size=int(arrays["tags"].max()) + 1,
        upload_type_vocabulary_size=len(upload_mapping) + 1,
        video_type_vocabulary_size=len(video_mapping) + 1,
        music_type_vocabulary_size=len(music_type_mapping) + 1,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def load_item_feature_arrays(directory: Path, mmap_mode: str | None = "r") -> dict[str, np.ndarray]:
    """Load all model arrays without materializing copies when mmap_mode is enabled."""

    names = (
        "present",
        "author_bucket",
        "music_bucket",
        "upload_type",
        "video_type",
        "music_type",
        "tags",
        "duration_log_seconds",
        "aspect_ratio",
        "upload_time_ms",
    )
    return {
        name: np.load(directory / f"{name}.npy", mmap_mode=mmap_mode, allow_pickle=False)
        for name in names
    }
