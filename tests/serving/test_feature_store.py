from __future__ import annotations

from pathlib import Path

import numpy as np

from short_video_recsys.serving.feature_store import MemoryFeatureStore


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
