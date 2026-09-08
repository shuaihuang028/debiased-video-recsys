"""Sequential ItemCF trained and scored with DuckDB."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from short_video_recsys.retrieval.protocol import _sql_paths


@dataclass(frozen=True)
class ItemCFBuildSummary:
    output_path: str
    neighbors: int
    source_items: int


def _sql_string(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def fit_sequential_itemcf(
    fit_paths: Sequence[Path],
    output_path: Path,
    target_column: str = "long_view",
    context_window: int = 5,
    neighbors_per_item: int = 200,
    temp_directory: Path = Path("artifacts/duckdb_tmp"),
) -> ItemCFBuildSummary:
    """Fit directed distance-weighted item transitions with cosine normalization."""

    if context_window <= 0 or neighbors_per_item <= 0:
        raise ValueError("context_window and neighbors_per_item must be positive")
    if target_column not in {"is_click", "is_like", "long_view"}:
        raise ValueError(f"Unsupported retrieval target: {target_column}")
    missing = [str(path) for path in fit_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ItemCF inputs:\n" + "\n".join(missing))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()

    source = f"read_parquet({_sql_paths(fit_paths)}, union_by_name = true)"
    connection = duckdb.connect()
    connection.execute(f"SET temp_directory = {_sql_string(temp_directory)}")
    query = f"""
        WITH positives AS (
            SELECT
                user_id,
                video_id,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id ORDER BY time_ms, video_id
                ) AS position
            FROM {source}
            WHERE {target_column} = 1
        ), frequencies AS (
            SELECT video_id, COUNT(*) AS frequency
            FROM positives
            GROUP BY video_id
        ), pair_weights AS (
            SELECT
                earlier.video_id AS source_item,
                later.video_id AS target_item,
                SUM(1.0 / (later.position - earlier.position)) AS pair_weight
            FROM positives AS earlier
            JOIN positives AS later
              ON earlier.user_id = later.user_id
             AND later.position > earlier.position
             AND later.position <= earlier.position + {context_window}
             AND later.video_id != earlier.video_id
            GROUP BY earlier.video_id, later.video_id
        ), scored AS (
            SELECT
                pairs.source_item,
                pairs.target_item,
                pairs.pair_weight
                    / SQRT(source_freq.frequency * target_freq.frequency) AS similarity
            FROM pair_weights AS pairs
            JOIN frequencies AS source_freq ON pairs.source_item = source_freq.video_id
            JOIN frequencies AS target_freq ON pairs.target_item = target_freq.video_id
        )
        SELECT source_item, target_item, similarity
        FROM scored
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY source_item ORDER BY similarity DESC, target_item ASC
        ) <= {neighbors_per_item}
    """
    connection.execute(
        f"COPY ({query}) TO {_sql_string(temporary_output)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    temporary_output.replace(output_path)
    relation = f"read_parquet({_sql_string(output_path)})"
    neighbors, source_items = connection.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT source_item) FROM {relation}"
    ).fetchone()
    connection.close()
    return ItemCFBuildSummary(str(output_path), int(neighbors), int(source_items))


def recommend_itemcf(
    neighbor_path: Path,
    histories: Mapping[int, Sequence[int]],
    k: int,
) -> dict[int, list[int]]:
    """Aggregate neighbors from recent history using reciprocal-rank recency weights."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not neighbor_path.is_file():
        raise FileNotFoundError(neighbor_path)

    history_rows = [
        (query_id, item_id, 1.0 / (rank + 1))
        for query_id, items in histories.items()
        for rank, item_id in enumerate(items)
    ]
    recommendations = {query_id: [] for query_id in histories}
    if not history_rows:
        return recommendations

    history_frame = pd.DataFrame(
        history_rows, columns=["query_id", "source_item", "recency_weight"]
    )
    connection = duckdb.connect()
    connection.register("query_history", history_frame)
    neighbor_source = f"read_parquet({_sql_string(neighbor_path)})"
    rows = connection.execute(
        f"""
        WITH scores AS (
            SELECT
                history.query_id,
                neighbors.target_item,
                SUM(history.recency_weight * neighbors.similarity) AS score
            FROM query_history AS history
            JOIN {neighbor_source} AS neighbors USING (source_item)
            GROUP BY history.query_id, neighbors.target_item
        ), unseen AS (
            SELECT scores.*
            FROM scores
            ANTI JOIN query_history AS history
              ON scores.query_id = history.query_id
             AND scores.target_item = history.source_item
        )
        SELECT query_id, target_item
        FROM unseen
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY query_id ORDER BY score DESC, target_item ASC
        ) <= {k}
        ORDER BY query_id, score DESC, target_item ASC
        """
    ).fetchall()
    connection.close()
    for query_id, target_item in rows:
        recommendations[int(query_id)].append(int(target_item))
    return recommendations
