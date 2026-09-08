"""Content-aware Two-Tower retrieval model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TwoTowerConfig:
    num_users: int
    author_buckets: int
    music_buckets: int
    tag_vocabulary_size: int
    upload_type_vocabulary_size: int
    video_type_vocabulary_size: int
    music_type_vocabulary_size: int
    embedding_dim: int = 64
    categorical_dim: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


class TwoTower(nn.Module):
    """Shared content embeddings align recent user interests with cold-capable items."""

    def __init__(self, config: TwoTowerConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.categorical_dim
        self.user_embedding = nn.Embedding(config.num_users + 1, dim * 2, padding_idx=0)
        self.author_embedding = nn.Embedding(config.author_buckets + 1, dim, padding_idx=0)
        self.music_embedding = nn.Embedding(config.music_buckets + 1, dim, padding_idx=0)
        self.tag_embedding = nn.Embedding(config.tag_vocabulary_size, dim, padding_idx=0)
        self.upload_type_embedding = nn.Embedding(
            config.upload_type_vocabulary_size, dim // 2, padding_idx=0
        )
        self.video_type_embedding = nn.Embedding(
            config.video_type_vocabulary_size, dim // 4, padding_idx=0
        )
        self.music_type_embedding = nn.Embedding(
            config.music_type_vocabulary_size, dim // 4, padding_idx=0
        )
        item_input_dim = dim * 3 + dim // 2 + dim // 4 + dim // 4 + 2
        self.item_mlp = nn.Sequential(
            nn.Linear(item_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.embedding_dim),
        )
        user_input_dim = dim * 2 + dim + dim
        self.user_mlp = nn.Sequential(
            nn.Linear(user_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.embedding_dim),
        )

    @staticmethod
    def _masked_mean(
        embeddings: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]
    ) -> torch.Tensor:
        expanded_mask = mask.unsqueeze(-1).to(embeddings.dtype)
        numerator = (embeddings * expanded_mask).sum(dim=dims)
        denominator = expanded_mask.sum(dim=dims).clamp_min(1.0)
        return numerator / denominator

    def encode_item(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        tags = features["tags"]
        tag_vector = self._masked_mean(self.tag_embedding(tags), tags.ne(0), dims=(1,))
        numeric = torch.stack(
            (
                features["duration_log_seconds"] / 8.0,
                features["aspect_ratio"] / 2.0,
            ),
            dim=1,
        )
        combined = torch.cat(
            (
                self.author_embedding(features["author_bucket"]),
                self.music_embedding(features["music_bucket"]),
                tag_vector,
                self.upload_type_embedding(features["upload_type"]),
                self.video_type_embedding(features["video_type"]),
                self.music_type_embedding(features["music_type"]),
                numeric,
            ),
            dim=1,
        )
        return F.normalize(self.item_mlp(combined), dim=1)

    def encode_user(
        self,
        user_ids: torch.Tensor,
        history_tags: torch.Tensor,
        history_authors: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_users = torch.where(
            (user_ids >= 0) & (user_ids < self.config.num_users), user_ids + 1, 0
        )
        tag_mask = history_tags.ne(0) & history_mask.unsqueeze(-1)
        history_tag_vector = self._masked_mean(
            self.tag_embedding(history_tags), tag_mask, dims=(1, 2)
        )
        history_author_vector = self._masked_mean(
            self.author_embedding(history_authors), history_mask, dims=(1,)
        )
        combined = torch.cat(
            (
                self.user_embedding(safe_users),
                history_tag_vector,
                history_author_vector,
            ),
            dim=1,
        )
        return F.normalize(self.user_mlp(combined), dim=1)
