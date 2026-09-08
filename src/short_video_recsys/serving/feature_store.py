"""User-feature stores backed by local mmap files or Redis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class UserFeatures:
    embedding: np.ndarray
    history_features: np.ndarray
    recent_items: np.ndarray


class FeatureStore(Protocol):
    backend_name: str

    def get_user(self, user_id: int) -> UserFeatures | None: ...

    def close(self) -> None: ...


class MemoryFeatureStore:
    backend_name = "memory_mmap"

    def __init__(self, snapshot_dir: Path):
        self.embeddings = np.load(snapshot_dir / "user_embeddings.npy", mmap_mode="r")
        self.histories = np.load(
            snapshot_dir / "user_history_features.npy", mmap_mode="r"
        )
        self.recent = np.load(
            snapshot_dir / "recent_long_view_items.npy", mmap_mode="r"
        )

    def get_user(self, user_id: int) -> UserFeatures | None:
        if user_id < 0 or user_id >= len(self.embeddings):
            return None
        recent = np.asarray(self.recent[user_id])
        return UserFeatures(
            embedding=np.asarray(self.embeddings[user_id], dtype=np.float32).copy(),
            history_features=np.asarray(
                self.histories[user_id], dtype=np.float32
            ).copy(),
            recent_items=recent[recent >= 0].astype(np.int32, copy=True),
        )

    def close(self) -> None:
        return None


class RedisFeatureStore:
    backend_name = "redis"

    def __init__(self, snapshot_dir: Path, redis_url: str, seed_if_empty: bool = True):
        import redis

        self.client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.client.ping()
        with (snapshot_dir / "metadata.json").open(encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        if seed_if_empty and not self.client.exists("recsys:snapshot:metadata"):
            self._seed(snapshot_dir)

    def _seed(self, snapshot_dir: Path) -> None:
        embeddings = np.load(snapshot_dir / "user_embeddings.npy", mmap_mode="r")
        histories = np.load(
            snapshot_dir / "user_history_features.npy", mmap_mode="r"
        )
        recent = np.load(
            snapshot_dir / "recent_long_view_items.npy", mmap_mode="r"
        )
        pipeline = self.client.pipeline(transaction=False)
        for user_id in range(len(embeddings)):
            pipeline.hset(
                f"recsys:user:{user_id}",
                mapping={
                    "embedding": np.asarray(embeddings[user_id], dtype=np.float32).tobytes(),
                    "history": np.asarray(histories[user_id], dtype=np.float32).tobytes(),
                    "recent": np.asarray(recent[user_id], dtype=np.int32).tobytes(),
                },
            )
            if user_id % 200 == 199:
                pipeline.execute()
        pipeline.set("recsys:snapshot:metadata", json.dumps(self.metadata))
        pipeline.execute()

    def get_user(self, user_id: int) -> UserFeatures | None:
        values = self.client.hmget(
            f"recsys:user:{user_id}", ("embedding", "history", "recent")
        )
        if any(value is None for value in values):
            return None
        embedding_raw, history_raw, recent_raw = values
        embedding = np.frombuffer(embedding_raw, dtype=np.float32).copy()
        history = np.frombuffer(history_raw, dtype=np.float32).copy()
        recent = np.frombuffer(recent_raw, dtype=np.int32)
        return UserFeatures(
            embedding=embedding,
            history_features=history,
            recent_items=recent[recent >= 0].copy(),
        )

    def close(self) -> None:
        self.client.close()
