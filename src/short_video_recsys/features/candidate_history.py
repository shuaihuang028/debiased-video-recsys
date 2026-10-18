"""Historical features for arbitrary, potentially unexposed candidate items."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from short_video_recsys.features.duckdb_pipeline import _parquet_source_sql, _rate_expressions
from short_video_recsys.features.point_in_time import DEFAULT_WINDOWS_MS
from short_video_recsys.ranking.train_ranker import COUNT_SUFFIXES

LOGGER = logging.getLogger(__name__)


def _entity_history(
    connection: duckdb.DuckDBPyConnection,
    source: str,
    entity: str,
    scope: str,
) -> None:
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE requested_entities AS "
        f"SELECT DISTINCT {entity}, time_ms FROM feature_requests"
    )
    max_time = int(connection.execute("SELECT MAX(time_ms) FROM requested_entities").fetchone()[0])
    counts = {
        "impression_count": "COUNT(*)",
        "click_count": "SUM(COALESCE(is_click, 0))",
        "like_count": "SUM(COALESCE(is_like, 0))",
        "long_view_count": "SUM(COALESCE(long_view, 0))",
        "watch_ratio_sum": "SUM(CASE WHEN duration_ms > 0 THEN "
        "LEAST(GREATEST(COALESCE(play_time_ms, 0)::DOUBLE / duration_ms, "
        "0.0), 5.0) ELSE 0.0 END)",
    }
    aggregates = ", ".join(f"{expression} AS {suffix}" for suffix, expression in counts.items())
    cumulative = ", ".join(
        f"SUM({suffix}) OVER (PARTITION BY {entity} ORDER BY time_ms "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS {suffix}"
        for suffix in COUNT_SUFFIXES
    )
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE event_prefix AS
        WITH grouped AS (
            SELECT {entity}, time_ms, {aggregates} FROM {source}
            WHERE time_ms < {max_time}
              AND {entity} IN (SELECT {entity} FROM requested_entities)
            GROUP BY {entity}, time_ms
        )
        SELECT {entity}, time_ms, {cumulative} FROM grouped
    """)
    cutoffs = " UNION ALL ".join(
        f"SELECT {entity}, time_ms, '{period}' AS period, "
        f"time_ms - {window} AS cutoff_ms FROM requested_entities"
        for period, window in [("hist", 0), *DEFAULT_WINDOWS_MS.items()]
    )
    state_columns = ", ".join(f"COALESCE(p.{suffix}, 0) AS {suffix}" for suffix in COUNT_SUFFIXES)
    # Both ends are exclusive prefix lookups: prefix(t-) - prefix((t-window)-).
    # This includes the lower window boundary and excludes all simultaneous events.
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE prefix_lookups AS
        SELECT b.{entity}, b.time_ms, b.period, {state_columns}
        FROM ({cutoffs}) b ASOF LEFT JOIN event_prefix p
          ON b.{entity} = p.{entity} AND b.cutoff_ms > p.time_ms
    """)
    features = []
    for period in ("hist", *DEFAULT_WINDOWS_MS):
        for suffix in COUNT_SUFFIXES:
            current = f"MAX({suffix}) FILTER (WHERE period = 'hist')"
            if period != "hist":
                current += f" - MAX({suffix}) FILTER (WHERE period = '{period}')"
            features.append(f"GREATEST({current}, 0) AS {scope}_{period}_{suffix}")
    rates = _rate_expressions(DEFAULT_WINDOWS_MS, {entity: scope}, 1.0, 10.0)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE {scope}_candidate_history AS
        WITH counts AS (
            SELECT {entity}, time_ms, {", ".join(features)}
            FROM prefix_lookups GROUP BY {entity}, time_ms
        )
        SELECT *, {", ".join(rates)} FROM counts
    """)
    connection.execute("DROP TABLE event_prefix")
    connection.execute("DROP TABLE prefix_lookups")


def materialize_candidate_history(
    connection: duckdb.DuckDBPyConnection,
    interaction_paths: list[Path],
    requests: pd.DataFrame,
    output_path: Path,
) -> None:
    """Build query-time histories for retrieved items, including unexposed items.

    Uses the ranker's 1h/1d/7d windows, watch-ratio cap of 5, and Beta(1,10) smoothing.
    """
    if requests.empty:
        raise ValueError("Feature requests cannot be empty")
    if requests["request_id"].duplicated().any():
        raise ValueError("request_id must be unique")
    connection.register("feature_requests", requests)
    source = _parquet_source_sql(interaction_paths)
    for entity, scope in (("user_id", "user"), ("video_id", "item")):
        LOGGER.info("Building %s candidate history for %d requests", scope, len(requests))
        _entity_history(connection, source, entity, scope)
    destination = str(output_path).replace("'", "''")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"""
        COPY (
            SELECT r.*, u.* EXCLUDE (user_id, time_ms), i.* EXCLUDE (video_id, time_ms)
            FROM feature_requests r
            JOIN user_candidate_history u USING (user_id, time_ms)
            JOIN item_candidate_history i USING (video_id, time_ms)
            ORDER BY request_id
        ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    connection.execute("DROP TABLE user_candidate_history")
    connection.execute("DROP TABLE item_candidate_history")
    connection.execute("DROP TABLE requested_entities")
    connection.unregister("feature_requests")
