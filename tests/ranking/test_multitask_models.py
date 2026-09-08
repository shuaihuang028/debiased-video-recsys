from __future__ import annotations

import pytest
import torch

from short_video_recsys.ranking.multitask_models import (
    FeatureEncoder,
    MMoE,
    SharedBottom,
    SingleTaskMLP,
    parameter_count,
)


def _encoder() -> FeatureEncoder:
    return FeatureEncoder(cardinalities=[5, 7], embedding_dims=[3, 4], numeric_features=6)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    categorical = torch.tensor([[0, 1], [2, 3], [4, 6]], dtype=torch.long)
    numeric = torch.randn(3, 6)
    return categorical, numeric


@pytest.mark.parametrize(
    ("model", "expected_tasks"),
    [
        (SingleTaskMLP(_encoder(), [8, 4], 0.0), 1),
        (SharedBottom(_encoder(), ["click", "long", "like"], [8, 4], 3, 0.0), 3),
        (MMoE(_encoder(), ["click", "long", "like"], 4, [6, 4], 3, 0.0), 3),
    ],
)
def test_multitask_models_produce_trainable_logits(
    model: torch.nn.Module, expected_tasks: int
) -> None:
    categorical, numeric = _inputs()

    logits = model(categorical, numeric)
    logits.sum().backward()

    assert logits.shape == (3, expected_tasks)
    assert torch.isfinite(logits).all()
    assert parameter_count(model) > 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mmoe_gate_weights_are_task_specific_probabilities() -> None:
    model = MMoE(_encoder(), ["click", "long", "like"], 4, [6, 4], 3, 0.0)
    categorical, numeric = _inputs()

    weights = model.gate_weights(categorical, numeric)

    assert set(weights) == {"click", "long", "like"}
    for task_weights in weights.values():
        assert task_weights.shape == (3, 4)
        assert torch.allclose(task_weights.sum(dim=1), torch.ones(3))
        assert torch.all(task_weights >= 0)


def test_feature_encoder_rejects_invalid_specification() -> None:
    with pytest.raises(ValueError, match="equal non-zero length"):
        FeatureEncoder(cardinalities=[5], embedding_dims=[3, 4], numeric_features=2)
