"""Most-popular engaged-item retrieval baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import duckdb

from short_video_recsys.retrieval.protocol import _sql_paths


def fit_popularity(
    fit_paths: Sequence[Path], target_column: str = "long_view"
) -> list[int]:
    """Rank items by positive-event count with deterministic item-ID tie breaking."""

    if target_column not in {"is_click", "is_like", "long_view"}:
        raise ValueError(f"Unsupported retrieval target: {target_column}")
    source = f"read_parquet({_sql_paths(fit_paths)}, union_by_name = true)"
    connection = duckdb.connect()
    rows = connection.execute(
        f"""
        SELECT video_id, COUNT(*) AS positive_events
        FROM {source}
        WHERE {target_column} = 1
        GROUP BY video_id
        ORDER BY positive_events DESC, video_id ASC
        """
    ).fetchall()
    connection.close()
    return [int(video_id) for video_id, _ in rows]


def recommend_popular(
    popularity: Sequence[int],
    seen_items: Mapping[int, set[int]],
    users: Sequence[int],
    k: int,
) -> dict[int, list[int]]:
    """Recommend globally popular items while filtering prior positive interactions."""

    if k <= 0:
        raise ValueError("k must be positive")
    recommendations = {}
    for user_id in users:
        seen = seen_items.get(user_id, set())
        selected = []
        for item_id in popularity:
            if item_id not in seen:
                selected.append(item_id)
                if len(selected) == k:
                    break
        recommendations[user_id] = selected
    return recommendations
