"""Deterministic pairwise training for random and exposure negatives."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from short_video_recsys.features.recent_history import HISTORY_POLICY, TIMESTAMP_POLICY
from short_video_recsys.training.item_features import load_item_feature_arrays
from short_video_recsys.training.two_tower import TwoTower, TwoTowerConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_pairwise_accuracy: float
    validation_loss: float
    validation_pairwise_accuracy: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _item_batch(
    arrays: dict[str, np.ndarray], item_ids: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    categorical = (
        "author_bucket",
        "music_bucket",
        "upload_type",
        "video_type",
        "music_type",
        "tags",
    )
    numeric = ("duration_log_seconds", "aspect_ratio")
    output = {
        name: torch.as_tensor(np.asarray(arrays[name][item_ids]), dtype=torch.long, device=device)
        for name in categorical
    }
    output.update(
        {
            name: torch.as_tensor(
                np.asarray(arrays[name][item_ids]), dtype=torch.float32, device=device
            )
            for name in numeric
        }
    )
    return output


def _user_batch(
    arrays: dict[str, np.ndarray],
    examples: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = examples["history_items"][indices]
    history_mask_np = history >= 0
    safe_history = np.where(history_mask_np, history, 0)
    history_tags_np = np.asarray(arrays["tags"][safe_history]).copy()
    history_authors_np = np.asarray(arrays["author_bucket"][safe_history]).copy()
    history_tags_np[~history_mask_np] = 0
    history_authors_np[~history_mask_np] = 0
    return (
        torch.as_tensor(examples["user_id"][indices], dtype=torch.long, device=device),
        torch.as_tensor(history_tags_np, dtype=torch.long, device=device),
        torch.as_tensor(history_authors_np, dtype=torch.long, device=device),
        torch.as_tensor(history_mask_np, dtype=torch.bool, device=device),
    )


def _run_batches(
    model: TwoTower,
    arrays: dict[str, np.ndarray],
    examples: dict[str, np.ndarray],
    indices: np.ndarray,
    negative_field: str,
    batch_size: int,
    temperature: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    ordered = rng.permutation(indices) if training else indices
    loss_sum = 0.0
    correct = 0
    observations = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, len(ordered), batch_size):
            batch_indices = ordered[start : start + batch_size]
            user_inputs = _user_batch(arrays, examples, batch_indices, device)
            positive = _item_batch(arrays, examples["positive_item"][batch_indices], device)
            negative = _item_batch(arrays, examples[negative_field][batch_indices], device)
            user_vector = model.encode_user(*user_inputs)
            positive_score = (user_vector * model.encode_item(positive)).sum(dim=1)
            negative_score = (user_vector * model.encode_item(negative)).sum(dim=1)
            loss = F.softplus(-(positive_score - negative_score) / temperature).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_count = len(batch_indices)
            loss_sum += float(loss.detach().cpu()) * batch_count
            correct += int((positive_score > negative_score).sum().detach().cpu())
            observations += batch_count
    return loss_sum / observations, correct / observations


def train_one_model(config: dict[str, Any], negative_mode: str) -> dict[str, Any]:
    """Train one negative-sampling variant with a shared chronological validation split."""

    if negative_mode not in {"random", "exposure"}:
        raise ValueError(f"Unsupported negative mode: {negative_mode}")
    seed = int(config["seed"])
    _seed_everything(seed)
    paths = config["paths"]
    model_values = config["model"]
    torch.set_num_threads(int(model_values.get("num_threads", 4)))
    item_values = config["item_features"]
    device = _device(str(model_values["device"]))

    loaded = np.load(Path(paths["training_examples"]), allow_pickle=False)
    examples = {name: loaded[name] for name in loaded.files}
    if "history_max_time_ms" not in examples:
        raise ValueError("Regenerate training examples with the strict history sampler")
    if np.any(examples["history_max_time_ms"] >= examples["time_ms"]):
        raise ValueError("Training histories must strictly precede their example timestamps")
    arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
    with (Path(paths["item_feature_dir"]) / "summary.json").open(encoding="utf-8") as handle:
        item_summary = json.load(handle)

    negative_field = f"{negative_mode}_negative_item"
    valid_indices = np.arange(len(examples["time_ms"]))
    if negative_mode == "exposure":
        valid_indices = valid_indices[examples[negative_field] >= 0]
    validation_fraction = float(model_values["validation_fraction"])
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    split_time = int(np.quantile(examples["time_ms"], 1 - validation_fraction))
    train_indices = valid_indices[examples["time_ms"][valid_indices] <= split_time]
    validation_indices = valid_indices[examples["time_ms"][valid_indices] > split_time]

    model_config = TwoTowerConfig(
        num_users=int(examples["user_id"].max()) + 1,
        author_buckets=int(item_values["author_buckets"]),
        music_buckets=int(item_values["music_buckets"]),
        tag_vocabulary_size=int(item_summary["tag_vocabulary_size"]),
        upload_type_vocabulary_size=int(item_summary["upload_type_vocabulary_size"]),
        video_type_vocabulary_size=int(item_summary["video_type_vocabulary_size"]),
        music_type_vocabulary_size=int(item_summary["music_type_vocabulary_size"]),
        embedding_dim=int(model_values["embedding_dim"]),
        categorical_dim=int(model_values["categorical_dim"]),
    )
    model = TwoTower(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_values["learning_rate"]),
        weight_decay=float(model_values["weight_decay"]),
    )
    batch_size = int(model_values["batch_size"])
    temperature = float(model_values["temperature"])
    rng = np.random.default_rng(seed)
    history = []
    best_loss = float("inf")
    checkpoint_dir = Path(paths["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"two_tower_{negative_mode}.pt"
    for epoch in range(1, int(model_values["epochs"]) + 1):
        train_loss, train_accuracy = _run_batches(
            model,
            arrays,
            examples,
            train_indices,
            negative_field,
            batch_size,
            temperature,
            device,
            optimizer,
            rng,
        )
        validation_loss, validation_accuracy = _run_batches(
            model,
            arrays,
            examples,
            validation_indices,
            negative_field,
            batch_size,
            temperature,
            device,
            None,
            rng,
        )
        metrics = EpochMetrics(
            epoch, train_loss, train_accuracy, validation_loss, validation_accuracy
        )
        history.append(asdict(metrics))
        LOGGER.info(
            "%s epoch %d: train loss %.5f, validation loss %.5f, validation accuracy %.4f",
            negative_mode,
            epoch,
            train_loss,
            validation_loss,
            validation_accuracy,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config.to_dict(),
                    "negative_mode": negative_mode,
                    "split_time_ms": split_time,
                    "best_validation_loss": best_loss,
                    "history_policy": HISTORY_POLICY,
                    "timestamp_policy": TIMESTAMP_POLICY,
                },
                checkpoint_path,
            )
    return {
        "negative_mode": negative_mode,
        "device": str(device),
        "train_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "split_time_ms": split_time,
        "best_validation_loss": best_loss,
        "checkpoint": str(checkpoint_path),
        "epochs": history,
        "history_policy": HISTORY_POLICY,
        "timestamp_policy": TIMESTAMP_POLICY,
    }


def train_negative_ablation(config: dict[str, Any]) -> dict[str, Any]:
    """Train both variants from identical seeded initialization and data split."""

    return {
        "objective": "pairwise_bpr",
        "config": config,
        "models": {mode: train_one_model(config, mode) for mode in ("random", "exposure")},
    }
