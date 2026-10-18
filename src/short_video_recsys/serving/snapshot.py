"""Build a fixed point-in-time feature snapshot for online serving."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch

from short_video_recsys.features.recent_history import HISTORY_POLICY, TIMESTAMP_POLICY
from short_video_recsys.ranking.train_ranker import (
    COUNT_SUFFIXES,
    LOG_HISTORY_FEATURES,
    RATE_SUFFIXES,
    WINDOWS,
)
from short_video_recsys.retrieval.end_to_end_candidates import file_sha256
from short_video_recsys.retrieval.two_tower_encoding import _load_model
from short_video_recsys.training.item_features import load_item_feature_arrays
from short_video_recsys.training.train_two_tower import _user_batch

DAY_MS = 86_400_000
WATCH_RATIO = (
    "CASE WHEN duration_ms > 0 THEN LEAST("
    "GREATEST(CAST(play_time_ms AS DOUBLE) / duration_ms, 0.0), 5.0) ELSE 0.0 END"
)


def _source(paths: list[Path]) -> str:
    values = ", ".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{values}], union_by_name = true)"


def _aggregate_query(source: str, scope: str, entity: str, snapshot_ms: int) -> str:
    expressions = []
    values = {
        "impression_count": "1",
        "click_count": "CAST(is_click AS BIGINT)",
        "like_count": "CAST(is_like AS BIGINT)",
        "long_view_count": "CAST(long_view AS BIGINT)",
        "watch_ratio_sum": WATCH_RATIO,
    }
    periods: list[tuple[str, int | None]] = [
        ("hist", None),
        ("1h", 3_600_000),
        ("1d", DAY_MS),
        ("7d", 7 * DAY_MS),
    ]
    for period, window_ms in periods:
        condition = (
            "" if window_ms is None else f" FILTER (WHERE time_ms >= {snapshot_ms - window_ms})"
        )
        for suffix, value in values.items():
            aggregate = "COUNT(*)" if suffix == "impression_count" else f"SUM({value})"
            expressions.append(f"COALESCE({aggregate}{condition}, 0) AS {scope}_{period}_{suffix}")
    aggregates = ",\n            ".join(expressions)
    return f"""
        SELECT
            {entity},
            {aggregates}
        FROM {source}
        WHERE time_ms < {snapshot_ms}
        GROUP BY {entity}
        ORDER BY {entity}
    """


def _history_columns(scope: str) -> tuple[str, ...]:
    return tuple(
        name
        for name in LOG_HISTORY_FEATURES
        if name.startswith(f"{scope}_") or name.startswith(f"log1p_{scope}_")
    )


def _history_values(frame: pd.DataFrame, scope: str) -> np.ndarray:
    columns = _history_columns(scope)
    output = np.zeros((len(frame), len(columns)), dtype=np.float32)
    column_positions = {name: position for position, name in enumerate(columns)}
    for window in WINDOWS:
        raw = {
            suffix: frame[f"{scope}_{window}_{suffix}"].to_numpy(dtype=np.float32)
            for suffix in COUNT_SUFFIXES
        }
        for suffix, values in raw.items():
            name = f"log1p_{scope}_{window}_{suffix}"
            output[:, column_positions[name]] = np.log1p(np.maximum(values, 0))
        denominator = raw["impression_count"] + 11.0
        for count_suffix, rate_suffix in zip(COUNT_SUFFIXES[1:4], RATE_SUFFIXES[:3]):
            name = f"{scope}_{window}_{rate_suffix}"
            output[:, column_positions[name]] = (raw[count_suffix] + 1.0) / denominator
        mean_name = f"{scope}_{window}_mean_watch_ratio"
        output[:, column_positions[mean_name]] = np.divide(
            raw["watch_ratio_sum"],
            raw["impression_count"],
            out=np.zeros(len(frame), dtype=np.float32),
            where=raw["impression_count"] > 0,
        )
    return output


def _initialize_history_array(path: Path, rows: int, columns: tuple[str, ...]) -> np.memmap:
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, len(columns)))
    array[:] = 0
    for position, name in enumerate(columns):
        if name.endswith(RATE_SUFFIXES[:3]):
            array[:, position] = 1.0 / 11.0
    return array


def _materialize_entity_history(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    scope: str,
    entity: str,
    snapshot_ms: int,
    output_path: Path,
    array_rows: int,
) -> int:
    columns = _history_columns(scope)
    output = _initialize_history_array(output_path, array_rows, columns)
    reader = connection.execute(
        _aggregate_query(source, scope, entity, snapshot_ms)
    ).fetch_record_batch(rows_per_batch=100_000)
    populated = 0
    for batch in reader:
        frame = batch.to_pandas()
        ids = frame[entity].to_numpy(dtype=np.int64)
        output[ids] = _history_values(frame, scope)
        populated += len(ids)
    output.flush()
    return populated


def _recent_histories(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    snapshot_ms: int,
    max_user_id: int,
    history_length: int,
) -> np.ndarray:
    rows = connection.execute(
        f"""
        WITH latest AS (
            SELECT user_id, video_id, MAX(time_ms) AS last_time_ms
            FROM {source}
            WHERE time_ms < {snapshot_ms} AND long_view = 1
            GROUP BY user_id, video_id
        )
        SELECT user_id, video_id, position FROM (
            SELECT
                user_id,
                video_id,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id ORDER BY last_time_ms DESC, video_id DESC
                ) AS position
            FROM latest
        )
        WHERE position <= {history_length}
        ORDER BY user_id, position
        """
    ).fetch_df()
    histories = np.full((max_user_id + 1, history_length), -1, dtype=np.int32)
    histories[
        rows["user_id"].to_numpy(dtype=np.int64),
        rows["position"].to_numpy(dtype=np.int64) - 1,
    ] = rows["video_id"].to_numpy(dtype=np.int32)
    return histories


def _encode_users(
    checkpoint_path: Path,
    item_arrays: dict[str, np.ndarray],
    histories: np.ndarray,
    output_path: Path,
) -> None:
    model = _load_model(checkpoint_path, torch.device("cpu"))
    user_ids = np.arange(len(histories), dtype=np.int32)
    examples = {"user_id": user_ids, "history_items": histories}
    indices = np.arange(len(user_ids))
    inputs = _user_batch(item_arrays, examples, indices, torch.device("cpu"))
    with torch.no_grad():
        embeddings = model.encode_user(*inputs).cpu().numpy().astype(np.float32)
    np.save(output_path, embeddings, allow_pickle=False)


def build_serving_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Materialize user/item histories and user vectors at one immutable timestamp."""

    paths = config["paths"]
    snapshot = config["snapshot"]
    snapshot_ms = int(snapshot["as_of_ms"])
    history_length = int(snapshot["history_length"])
    input_paths = [Path(path) for path in paths["interaction_files"]]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing snapshot inputs:\n" + "\n".join(missing))
    output_dir = Path(paths["snapshot_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    item_arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
    source = _source(input_paths)
    connection = duckdb.connect()
    max_user_id = int(
        connection.execute(
            f"SELECT MAX(user_id) FROM {source} WHERE time_ms < {snapshot_ms}"
        ).fetchone()[0]
    )
    started = time.perf_counter()
    user_count = _materialize_entity_history(
        connection,
        source,
        "user",
        "user_id",
        snapshot_ms,
        output_dir / "user_history_features.npy",
        max_user_id + 1,
    )
    item_count = _materialize_entity_history(
        connection,
        source,
        "item",
        "video_id",
        snapshot_ms,
        output_dir / "item_history_features.npy",
        len(item_arrays["present"]),
    )
    histories = _recent_histories(connection, source, snapshot_ms, max_user_id, history_length)
    connection.close()
    np.save(output_dir / "recent_long_view_items.npy", histories, allow_pickle=False)
    _encode_users(
        Path(paths["two_tower_checkpoint"]),
        item_arrays,
        histories,
        output_dir / "user_embeddings.npy",
    )
    metadata = {
        "as_of_ms": snapshot_ms,
        "history_length": history_length,
        "max_user_id": max_user_id,
        "item_array_size": len(item_arrays["present"]),
        "populated_users": user_count,
        "populated_items": item_count,
        "user_history_columns": list(_history_columns("user")),
        "item_history_columns": list(_history_columns("item")),
        "seconds": time.perf_counter() - started,
        "history_policy": HISTORY_POLICY,
        "timestamp_policy": TIMESTAMP_POLICY,
        "two_tower_checkpoint_sha256": file_sha256(Path(paths["two_tower_checkpoint"])),
        "ranker_model_sha256": file_sha256(Path(paths["ranker_model"])),
    }
    identity = {
        name: metadata[name]
        for name in (
            "as_of_ms",
            "history_length",
            "history_policy",
            "timestamp_policy",
            "two_tower_checkpoint_sha256",
            "ranker_model_sha256",
        )
    }
    metadata["snapshot_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata
