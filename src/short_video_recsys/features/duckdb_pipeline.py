"""Scalable DuckDB implementation of the point-in-time feature contract."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb

from short_video_recsys.data.schema import DataValidationError
from short_video_recsys.data.split import TimeBoundaries

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class FeatureBuildSummary:
    """Materialized feature-file metadata."""

    output_path: str
    rows: int
    columns: int
    train_rows: int
    validation_rows: int
    test_rows: int


@dataclass(frozen=True)
class RandomFeatureBuildSummary:
    """Materialized randomized-row features with observational PIT history."""

    output_path: str
    rows: int
    columns: int
    calibration_rows: int
    validation_rows: int
    test_rows: int


def _sql_string(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _source_sql(paths: Sequence[Path]) -> str:
    quoted = ", ".join(_sql_string(path) for path in paths)
    return f"read_csv_auto([{quoted}], header = true, union_by_name = true)"


def _parquet_source_sql(paths: Sequence[Path]) -> str:
    quoted = ", ".join(_sql_string(path) for path in paths)
    return f"read_parquet([{quoted}], union_by_name = true)"


def _window_clause(entity: str, window_ms: int | None) -> str:
    lower = "UNBOUNDED PRECEDING" if window_ms is None else f"{window_ms} PRECEDING"
    return (
        f"PARTITION BY {entity} ORDER BY time_ms "
        f"RANGE BETWEEN {lower} AND 1 PRECEDING"
    )


def _count_expressions(
    windows_ms: Mapping[str, int], entities: Mapping[str, str]
) -> list[str]:
    values = {
        "impression_count": "1",
        "click_count": "CAST(is_click AS BIGINT)",
        "like_count": "CAST(is_like AS BIGINT)",
        "long_view_count": "CAST(long_view AS BIGINT)",
        "watch_ratio_sum": "event_watch_ratio",
    }
    periods: list[tuple[str, int | None]] = [("hist", None), *windows_ms.items()]
    expressions = []
    for entity, prefix in entities.items():
        for label, window_ms in periods:
            if not _SAFE_LABEL.fullmatch(label):
                raise ValueError(f"Unsafe window label: {label}")
            if window_ms is not None and window_ms <= 0:
                raise ValueError(f"Window {label} must be positive")
            window = _window_clause(entity, window_ms)
            for suffix, value in values.items():
                alias = f"{prefix}_{label}_{suffix}"
                aggregate = "COUNT(*)" if suffix == "impression_count" else f"SUM({value})"
                expressions.append(f"COALESCE({aggregate} OVER ({window}), 0) AS {alias}")
    return expressions


def _rate_expressions(
    windows_ms: Mapping[str, int],
    entities: Mapping[str, str],
    alpha: float,
    beta: float,
) -> list[str]:
    expressions = []
    for prefix in entities.values():
        for label in ("hist", *windows_ms.keys()):
            impressions = f"{prefix}_{label}_impression_count"
            denominator = f"({impressions} + {alpha} + {beta})"
            for count_name, rate_name in (
                ("click_count", "ctr"),
                ("like_count", "like_rate"),
                ("long_view_count", "long_view_rate"),
            ):
                count = f"{prefix}_{label}_{count_name}"
                expressions.append(
                    f"({count} + {alpha}) / {denominator} AS {prefix}_{label}_{rate_name}"
                )
            watch_sum = f"{prefix}_{label}_watch_ratio_sum"
            expressions.append(
                f"CASE WHEN {impressions} > 0 THEN {watch_sum} / {impressions} "
                f"ELSE 0.0 END AS {prefix}_{label}_mean_watch_ratio"
            )
    return expressions


def feature_query(
    source: str,
    boundaries: TimeBoundaries,
    windows_ms: Mapping[str, int],
    watch_ratio_cap: float,
    smoothing_alpha: float,
    smoothing_beta: float,
) -> str:
    """Build SQL whose RANGE windows exactly match the tested Pandas reference."""

    if watch_ratio_cap <= 0:
        raise ValueError("watch_ratio_cap must be positive")
    if smoothing_alpha < 0 or smoothing_beta < 0:
        raise ValueError("Smoothing parameters must be non-negative")

    entities = {"user_id": "user", "video_id": "item"}
    counts = ",\n                ".join(_count_expressions(windows_ms, entities))
    rates = ",\n            ".join(
        _rate_expressions(windows_ms, entities, smoothing_alpha, smoothing_beta)
    )
    split = (
        f"CASE WHEN time_ms <= {boundaries.first_end_ms} THEN 'train' "
        f"WHEN time_ms <= {boundaries.second_end_ms} THEN 'validation' "
        f"ELSE 'test' END AS split"
    )
    return f"""
        WITH prepared AS (
            SELECT
                *,
                CASE
                    WHEN duration_ms > 0 THEN LEAST(
                        GREATEST(CAST(play_time_ms AS DOUBLE) / duration_ms, 0.0),
                        {watch_ratio_cap}
                    )
                    ELSE 0.0
                END AS event_watch_ratio
            FROM {source}
        ), counts AS (
            SELECT
                *,
                {counts}
            FROM prepared
        )
        SELECT
            * EXCLUDE (event_watch_ratio),
            {rates},
            {split}
        FROM counts
    """


def build_pit_parquet(
    input_paths: Sequence[Path],
    output_path: Path,
    boundaries: TimeBoundaries,
    windows_ms: Mapping[str, int],
    watch_ratio_cap: float,
    smoothing_alpha: float,
    smoothing_beta: float,
    temp_directory: Path,
) -> FeatureBuildSummary:
    """Build all full-data PIT features in one out-of-core query and write Parquet."""

    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)

    source = _source_sql(input_paths)
    connection = duckdb.connect()
    connection.execute(f"SET temp_directory = {_sql_string(temp_directory)}")
    rows, max_time = connection.execute(
        f"SELECT COUNT(*), MAX(time_ms) FROM {source}"
    ).fetchone()
    if rows == 0:
        raise DataValidationError("Cannot build features from empty inputs")
    if max_time > boundaries.third_end_ms:
        raise DataValidationError(
            f"Rows exceed configured final boundary {boundaries.third_end_ms}: max={max_time}"
        )

    query = feature_query(
        source,
        boundaries,
        windows_ms,
        watch_ratio_cap,
        smoothing_alpha,
        smoothing_beta,
    )
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()
    connection.execute(
        f"COPY ({query}) TO {_sql_string(temporary_output)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    temporary_output.replace(output_path)

    relation = f"read_parquet({_sql_string(output_path)})"
    output_rows = connection.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
    columns = len(connection.sql(f"SELECT * FROM {relation} LIMIT 0").columns)
    partition_counts = dict(
        connection.execute(f"SELECT split, COUNT(*) FROM {relation} GROUP BY split").fetchall()
    )
    connection.close()
    if output_rows != rows:
        raise AssertionError(f"Feature build changed row count: input={rows}, output={output_rows}")

    return FeatureBuildSummary(
        output_path=str(output_path),
        rows=int(output_rows),
        columns=int(columns),
        train_rows=int(partition_counts.get("train", 0)),
        validation_rows=int(partition_counts.get("validation", 0)),
        test_rows=int(partition_counts.get("test", 0)),
    )


def build_random_pit_parquet(
    standard_paths: Sequence[Path],
    random_paths: Sequence[Path],
    output_path: Path,
    boundaries: TimeBoundaries,
    windows_ms: Mapping[str, int],
    watch_ratio_cap: float,
    smoothing_alpha: float,
    smoothing_beta: float,
    temp_directory: Path,
) -> RandomFeatureBuildSummary:
    """Build random-exposure rows using all strictly prior observed events as history."""

    inputs = [*standard_paths, *random_paths]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)
    source = _parquet_source_sql(inputs)
    query = feature_query(
        source,
        boundaries,
        windows_ms,
        watch_ratio_cap,
        smoothing_alpha,
        smoothing_beta,
    )
    random_query = f"""
        SELECT
            * EXCLUDE (split),
            CASE
                WHEN time_ms <= {boundaries.first_end_ms} THEN 'calibration'
                WHEN time_ms <= {boundaries.second_end_ms} THEN 'validation'
                ELSE 'test'
            END AS split
        FROM ({query})
        WHERE is_rand = 1
    """
    connection = duckdb.connect()
    connection.execute(f"SET temp_directory = {_sql_string(temp_directory)}")
    expected_rows = connection.execute(
        f"SELECT COUNT(*) FROM {_parquet_source_sql(random_paths)}"
    ).fetchone()[0]
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()
    connection.execute(
        f"COPY ({random_query}) TO {_sql_string(temporary_output)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    temporary_output.replace(output_path)
    relation = f"read_parquet({_sql_string(output_path)})"
    output_rows = connection.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
    columns = len(connection.sql(f"SELECT * FROM {relation} LIMIT 0").columns)
    partition_counts = dict(
        connection.execute(f"SELECT split, COUNT(*) FROM {relation} GROUP BY split").fetchall()
    )
    connection.close()
    if output_rows != expected_rows:
        raise AssertionError(
            f"Random feature build changed row count: input={expected_rows}, output={output_rows}"
        )
    return RandomFeatureBuildSummary(
        output_path=str(output_path),
        rows=int(output_rows),
        columns=int(columns),
        calibration_rows=int(partition_counts.get("calibration", 0)),
        validation_rows=int(partition_counts.get("validation", 0)),
        test_rows=int(partition_counts.get("test", 0)),
    )
