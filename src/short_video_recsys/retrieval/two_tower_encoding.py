"""PyTorch-only item and query encoding job for Two-Tower retrieval."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from short_video_recsys.retrieval.protocol import RetrievalPeriodData, build_retrieval_period
from short_video_recsys.training.item_features import load_item_feature_arrays
from short_video_recsys.training.train_two_tower import _item_batch, _user_batch
from short_video_recsys.training.two_tower import TwoTower, TwoTowerConfig


def _load_model(checkpoint_path: Path, device: torch.device) -> TwoTower:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TwoTower(TwoTowerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def _encode_catalog(
    model: TwoTower,
    arrays: dict[str, np.ndarray],
    max_query_time: int,
    output_dir: Path,
    mode: str,
    batch_size: int = 16_384,
) -> tuple[int, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_mask = (
        arrays["present"]
        & (arrays["upload_time_ms"] > 0)
        & (arrays["upload_time_ms"] <= max_query_time)
    )
    item_ids = np.flatnonzero(candidate_mask).astype(np.int32)
    np.save(output_dir / f"item_ids_{mode}.npy", item_ids, allow_pickle=False)
    embeddings = np.lib.format.open_memmap(
        output_dir / f"item_embeddings_{mode}.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(item_ids), model.config.embedding_dim),
    )
    device = next(model.parameters()).device
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(item_ids), batch_size):
            end = min(start + batch_size, len(item_ids))
            batch = _item_batch(arrays, item_ids[start:end], device)
            embeddings[start:end] = model.encode_item(batch).cpu().numpy()
    embeddings.flush()
    return len(item_ids), time.perf_counter() - started


def _encode_queries(
    model: TwoTower,
    arrays: dict[str, np.ndarray],
    data: RetrievalPeriodData,
    history_length: int,
    output_dir: Path,
    mode: str,
) -> int:
    query_ids = sorted(data.targets)
    histories = np.full((len(query_ids), history_length), -1, dtype=np.int32)
    users = np.zeros(len(query_ids), dtype=np.int32)
    for position, query_id in enumerate(query_ids):
        history = data.histories[query_id][:history_length]
        histories[position, : len(history)] = history
        users[position] = data.query_users[query_id]
    examples = {"user_id": users, "history_items": histories}
    indices = np.arange(len(query_ids))
    device = next(model.parameters()).device
    user_inputs = _user_batch(arrays, examples, indices, device)
    with torch.no_grad():
        embeddings = model.encode_user(*user_inputs).cpu().numpy().astype(np.float32)
    np.save(output_dir / f"query_ids_{mode}.npy", np.asarray(query_ids, dtype=np.int32))
    np.save(output_dir / f"query_embeddings_{mode}.npy", embeddings)
    return len(query_ids)


def encode_two_tower_models(config: dict[str, Any]) -> dict[str, Any]:
    """Materialize item/query vectors without importing FAISS in this process."""

    paths = config["paths"]
    evaluation = config["evaluation"]
    if evaluation["period"] != "validation":
        raise ValueError("Current checkpoints are train-only; only validation evaluation is valid")
    processed = Path("data/processed")
    data = build_retrieval_period(
        [processed / "standard_train.parquet"],
        processed / "standard_validation.parquet",
        target_column=config["sampling"]["target_column"],
        history_size=int(config["sampling"]["history_length"]),
    )
    arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
    max_query_time = max(data.query_times.values())
    output_dir = Path(paths["embedding_dir"])
    results = {"period": "validation", "models": {}}
    for mode in ("random", "exposure"):
        model = _load_model(
            Path(paths["checkpoint_dir"]) / f"two_tower_{mode}.pt", torch.device("cpu")
        )
        candidate_items, encode_seconds = _encode_catalog(
            model, arrays, max_query_time, output_dir, mode
        )
        queries = _encode_queries(
            model,
            arrays,
            data,
            int(config["sampling"]["history_length"]),
            output_dir,
            mode,
        )
        results["models"][mode] = {
            "candidate_items": candidate_items,
            "queries": queries,
            "encode_catalog_seconds": encode_seconds,
        }
    return results
