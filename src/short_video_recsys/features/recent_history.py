"""Positive-item histories ordered by last event time."""

from __future__ import annotations

from collections import OrderedDict

HISTORY_POLICY = "most_recent_unique_positive_items_newest_first"
TIMESTAMP_POLICY = "events_strictly_before_example_timestamp"


def touch_recent_item(
    history: OrderedDict[int, int],
    item: int,
    timestamp: int,
    limit: int | None = None,
) -> None:
    """Keep each item once, ordered by its latest positive event."""
    history.pop(item, None)
    history[item] = timestamp
    if limit is not None and len(history) > limit:
        history.popitem(last=False)


def recent_item_ids(history: OrderedDict[int, int], limit: int) -> list[int]:
    return list(reversed(history))[:limit]
