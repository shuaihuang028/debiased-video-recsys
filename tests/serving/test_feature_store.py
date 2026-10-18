from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from short_video_recsys.serving.feature_store import MemoryFeatureStore, RedisFeatureStore


def test_memory_feature_store_filters_padding(tmp_path: Path) -> None:
    np.save(tmp_path / "user_embeddings.npy", np.ones((2, 3), dtype=np.float32))
    np.save(tmp_path / "user_history_features.npy", np.ones((2, 4), dtype=np.float32))
    np.save(
        tmp_path / "recent_long_view_items.npy",
        np.array([[10, -1], [11, 12]], dtype=np.int32),
    )
    store = MemoryFeatureStore(tmp_path)

    user = store.get_user(0)

    assert user is not None
    assert user.recent_items.tolist() == [10]
    assert store.get_user(2) is None


def test_redis_snapshot_namespace_does_not_reuse_legacy_vectors(tmp_path, monkeypatch) -> None:
    import redis

    class FakeRedis:
        def __init__(self):
            self.values = {"recsys:snapshot:metadata": "legacy"}
            self.hashes = {"recsys:user:0": {"embedding": b"old-vector"}}

        def ping(self):
            return True

        def exists(self, key):
            return key in self.values

        def pipeline(self, **kwargs):
            return self

        def hset(self, key, mapping):
            self.hashes[key] = mapping

        def set(self, key, value):
            self.values[key] = value

        def execute(self):
            return []

        def hmget(self, key, names):
            return [self.hashes.get(key, {}).get(name) for name in names]

    client = FakeRedis()
    monkeypatch.setattr(redis.Redis, "from_url", lambda *args, **kwargs: client)
    np.save(tmp_path / "user_embeddings.npy", np.array([[1.0, 2.0]], dtype=np.float32))
    np.save(tmp_path / "user_history_features.npy", np.ones((1, 3), dtype=np.float32))
    np.save(tmp_path / "recent_long_view_items.npy", np.array([[10, -1]], dtype=np.int32))
    (tmp_path / "metadata.json").write_text(json.dumps({"snapshot_id": "corrected"}))
    store = RedisFeatureStore(tmp_path, "redis://example")
    assert store.get_user(0).embedding.tolist() == [1.0, 2.0]
    assert client.hashes["recsys:user:0"]["embedding"] == b"old-vector"
    assert "recsys:corrected:snapshot:metadata" in client.values
