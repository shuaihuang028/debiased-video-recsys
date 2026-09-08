"""Neural ranking baselines for controlled multi-task comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


class FeatureEncoder(nn.Module):
    """Embed categorical columns and concatenate standardized numeric features."""

    def __init__(
        self,
        cardinalities: Sequence[int],
        embedding_dims: Sequence[int],
        numeric_features: int,
    ) -> None:
        super().__init__()
        if len(cardinalities) != len(embedding_dims) or not cardinalities:
            raise ValueError("cardinalities and embedding_dims must have equal non-zero length")
        if any(value <= 0 for value in cardinalities):
            raise ValueError("categorical cardinalities must be positive")
        if numeric_features < 0:
            raise ValueError("numeric_features cannot be negative")
        self.cardinalities = tuple(int(value) for value in cardinalities)
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dimension)
            for cardinality, dimension in zip(cardinalities, embedding_dims)
        )
        self.output_dim = int(sum(embedding_dims) + numeric_features)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        if categorical.ndim != 2 or categorical.shape[1] != len(self.embeddings):
            raise ValueError("categorical tensor has the wrong shape")
        if numeric.ndim != 2:
            raise ValueError("numeric tensor must be two-dimensional")
        encoded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        encoded.append(numeric)
        return torch.cat(encoded, dim=1)


def _mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        if width <= 0:
            raise ValueError("hidden dimensions must be positive")
        layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)))
        previous = width
    return nn.Sequential(*layers), previous


class SingleTaskMLP(nn.Module):
    """Long-view-only neural baseline with the common feature encoder."""

    task_names = ("long_view",)

    def __init__(self, encoder: FeatureEncoder, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.backbone, output_dim = _mlp(encoder.output_dim, hidden_dims, dropout)
        self.output = nn.Linear(output_dim, 1)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(categorical, numeric)
        return self.output(self.backbone(encoded))


class SharedBottom(nn.Module):
    """A shared representation with a separate prediction tower per task."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        task_names: Sequence[str],
        shared_dims: Sequence[int],
        tower_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(task_names) < 2:
            raise ValueError("shared-bottom requires at least two tasks")
        self.encoder = encoder
        self.task_names = tuple(task_names)
        self.shared, shared_output = _mlp(encoder.output_dim, shared_dims, dropout)
        self.towers = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(shared_output, tower_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(tower_dim, 1),
                )
                for task in self.task_names
            }
        )

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        shared = self.shared(self.encoder(categorical, numeric))
        return torch.cat([self.towers[task](shared) for task in self.task_names], dim=1)


class MMoE(nn.Module):
    """Multi-gate mixture-of-experts with task-specific soft expert routing."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        task_names: Sequence[str],
        num_experts: int,
        expert_dims: Sequence[int],
        tower_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(task_names) < 2:
            raise ValueError("MMoE requires at least two tasks")
        if num_experts < 2:
            raise ValueError("MMoE requires at least two experts")
        self.encoder = encoder
        self.task_names = tuple(task_names)
        self.num_experts = int(num_experts)
        expert_modules = []
        expert_output = None
        for _ in range(num_experts):
            expert, expert_output = _mlp(encoder.output_dim, expert_dims, dropout)
            expert_modules.append(expert)
        assert expert_output is not None
        self.experts = nn.ModuleList(expert_modules)
        self.gates = nn.ModuleDict(
            {task: nn.Linear(encoder.output_dim, num_experts) for task in self.task_names}
        )
        self.towers = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(expert_output, tower_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(tower_dim, 1),
                )
                for task in self.task_names
            }
        )

    def gate_weights(
        self, categorical: torch.Tensor, numeric: torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        encoded = self.encoder(categorical, numeric)
        return {task: torch.softmax(self.gates[task](encoded), dim=1) for task in self.task_names}

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(categorical, numeric)
        expert_outputs = torch.stack([expert(encoded) for expert in self.experts], dim=1)
        outputs = []
        for task in self.task_names:
            weights = torch.softmax(self.gates[task](encoded), dim=1).unsqueeze(-1)
            mixture = torch.sum(expert_outputs * weights, dim=1)
            outputs.append(self.towers[task](mixture))
        return torch.cat(outputs, dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
