"""Fixed offline retrieval protocol shared by every retriever."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import duckdb

from short_video_recsys.features.recent_history import recent_item_ids, touch_recent_item


@dataclass(frozen=True)
class RetrievalPeriodData:
    """Daily next-item queries with histories available at each query timestamp."""

    query_users: dict[int, int]
    query_times: dict[int, int]
    histories: dict[int, list[int]]
    warm_histories: dict[int, list[int]]
    seen_items: dict[int, set[int]]
    targets: dict[int, set[int]]
    warm_targets: dict[int, set[int]]
    catalog_items: set[int]


def _sql_paths(paths: Sequence[Path]) -> str:
    quoted = ", ".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"[{quoted}]"


def build_retrieval_period(
    fit_paths: Sequence[Path],
    evaluation_path: Path,
    target_column: str = "long_view",
    history_size: int = 50,
) -> RetrievalPeriodData:
    """Build user-period queries without removing cold targets from the main denominator."""

    if history_size <= 0:
        raise ValueError("history_size must be positive")
    if target_column not in {"is_click", "is_like", "long_view"}:
        raise ValueError(f"Unsupported retrieval target: {target_column}")
    missing = [str(path) for path in [*fit_paths, evaluation_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing retrieval inputs:\n" + "\n".join(missing))

    connection = duckdb.connect()
    fit_source = f"read_parquet({_sql_paths(fit_paths)}, union_by_name = true)"
    eval_source = "read_parquet('" + str(evaluation_path).replace("'", "''") + "')"

    catalog_items = set(
        connection.execute(f"SELECT DISTINCT video_id FROM {fit_source}").fetchnumpy()[
            "video_id"
        ].tolist()
    )
    history_rows = connection.execute(
        f"""
        SELECT user_id, video_id, time_ms
        FROM {fit_source}
        WHERE {target_column} = 1
        ORDER BY user_id, time_ms, video_id
        """
    ).fetch_df()
    fit_max_time = connection.execute(f"SELECT MAX(time_ms) FROM {fit_source}").fetchone()[0]
    eval_min_time = connection.execute(f"SELECT MIN(time_ms) FROM {eval_source}").fetchone()[0]
    if fit_max_time is not None and eval_min_time is not None and fit_max_time >= eval_min_time:
        connection.close()
        raise ValueError("Fit interactions must strictly precede the evaluation period")
    target_rows = connection.execute(
        f"""
        SELECT user_id, video_id, time_ms
        FROM {eval_source}
        WHERE {target_column} = 1
        ORDER BY user_id, time_ms, video_id
        """
    ).fetch_df()
    connection.close()

    user_state: dict[int, OrderedDict[int, int]] = {}

    def touch(user: int, item: int, timestamp: int) -> None:
        state = user_state.setdefault(user, OrderedDict())
        touch_recent_item(state, item, timestamp)

    for user_id, video_id, time_ms in history_rows.itertuples(index=False, name=None):
        user = int(user_id)
        item = int(video_id)
        touch(user, item, int(time_ms))

    query_users = {}
    query_times = {}
    histories = {}
    warm_histories = {}
    recent_seen = {}
    targets = {}
    warm_targets = {}
    emitted_user_days: set[tuple[int, int]] = set()
    rows = target_rows.itertuples(index=False, name=None)
    for (user_id, time_ms), simultaneous in groupby(rows, key=lambda row: (row[0], row[2])):
        user = int(user_id)
        timestamp = int(time_ms)
        items = [int(row[1]) for row in simultaneous]
        state = user_state.setdefault(user, OrderedDict())
        local_day = (timestamp + 8 * 60 * 60 * 1_000) // (24 * 60 * 60 * 1_000)
        user_day = (user, local_day)
        novel = [item for item in items if item not in state]
        if user_day not in emitted_user_days and novel:
            item = novel[0]
            query_id = len(targets)
            history = recent_item_ids(state, history_size)
            warm_history = [
                history_item
                for history_item in reversed(state.keys())
                if history_item in catalog_items
            ][:history_size]
            query_users[query_id] = user
            query_times[query_id] = timestamp
            histories[query_id] = history
            warm_histories[query_id] = warm_history
            recent_seen[query_id] = set(history)
            targets[query_id] = {item}
            warm_targets[query_id] = {item} if item in catalog_items else set()
            emitted_user_days.add(user_day)
        # Every query at t sees only events strictly before t, including tied rows.
        for item in items:
            touch(user, item, timestamp)

    return RetrievalPeriodData(
        query_users=query_users,
        query_times=query_times,
        histories=histories,
        warm_histories=warm_histories,
        seen_items=recent_seen,
        targets=targets,
        warm_targets=warm_targets,
        catalog_items=catalog_items,
    )
