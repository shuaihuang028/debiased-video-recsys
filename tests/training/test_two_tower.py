from __future__ import annotations

import pytest
import torch

from short_video_recsys.training.two_tower import TwoTower, TwoTowerConfig


def test_two_tower_outputs_normalized_vectors_and_handles_empty_history() -> None:
    config = TwoTowerConfig(
        num_users=3,
        author_buckets=8,
        music_buckets=8,
        tag_vocabulary_size=6,
        upload_type_vocabulary_size=4,
        video_type_vocabulary_size=3,
        music_type_vocabulary_size=3,
        embedding_dim=8,
        categorical_dim=8,
    )
    model = TwoTower(config)
    item = {
        "author_bucket": torch.tensor([1, 2]),
        "music_bucket": torch.tensor([2, 3]),
        "upload_type": torch.tensor([1, 2]),
        "video_type": torch.tensor([1, 1]),
        "music_type": torch.tensor([1, 2]),
        "tags": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "duration_log_seconds": torch.tensor([2.0, 3.0]),
        "aspect_ratio": torch.tensor([0.5, 1.0]),
    }
    users = model.encode_user(
        torch.tensor([0, 99]),
        torch.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 0]]]),
        torch.tensor([[1, 0], [0, 0]]),
        torch.tensor([[True, False], [False, False]]),
    )
    items = model.encode_item(item)

    assert users.shape == items.shape == (2, 8)
    assert torch.linalg.vector_norm(users, dim=1).tolist() == pytest.approx([1.0, 1.0])
    assert torch.linalg.vector_norm(items, dim=1).tolist() == pytest.approx([1.0, 1.0])
