"""FAISS-only indexing and evaluation job, isolated from PyTorch's OpenMP runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from short_video_recsys.retrieval.metrics import evaluate_recommendations
from short_video_recsys.retrieval.protocol import RetrievalPeriodData, build_retrieval_period
from short_video_recsys.training.item_features import load_item_feature_arrays


def _build_ivf_index(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    output_path: Path,
    nlist: int,
    nprobe: int,
    seed: int,
) -> tuple[faiss.IndexIVFFlat, float]:
    if nlist <= 0 or nprobe <= 0 or nprobe > nlist:
        raise ValueError("Require 0 < nprobe <= nlist")
    dimension = embeddings.shape[1]
    index = faiss.IndexIVFFlat(
        faiss.IndexFlatIP(dimension), dimension, nlist, faiss.METRIC_INNER_PRODUCT
    )
    index.nprobe = nprobe
    rng = np.random.default_rng(seed)
    training_size = min(len(embeddings), max(nlist * 40, 100_000))
    training_positions = rng.choice(len(embeddings), size=training_size, replace=False)
    started = time.perf_counter()
    index.train(np.ascontiguousarray(embeddings[training_positions]))
    for start in range(0, len(embeddings), 100_000):
        end = min(start + 100_000, len(embeddings))
        index.add_with_ids(
            np.ascontiguousarray(embeddings[start:end]),
            np.ascontiguousarray(item_ids[start:end].astype(np.int64)),
        )
    build_seconds = time.perf_counter() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    return index, build_seconds


def _filter_search_results(
    raw_ids: np.ndarray,
    query_ids: list[int],
    data: RetrievalPeriodData,
    upload_times: np.ndarray,
    k: int,
) -> dict[int, list[int]]:
    recommendations = {}
    for position, query_id in enumerate(query_ids):
        seen = data.seen_items[query_id]
        query_time = data.query_times[query_id]
        selected = []
        for item_raw in raw_ids[position]:
            item = int(item_raw)
            if item < 0 or item in seen or upload_times[item] >= query_time:
                continue
            selected.append(item)
            if len(selected) == k:
                break
        recommendations[query_id] = selected
    return recommendations


def _exact_ann_recall(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    query_ids: list[int],
    query_embeddings: np.ndarray,
    approximate: Mapping[int, list[int]],
    data: RetrievalPeriodData,
    upload_times: np.ndarray,
    k: int,
    sample_queries: int,
    seed: int,
) -> float:
    if sample_queries <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    positions = rng.choice(
        len(query_ids), size=min(sample_queries, len(query_ids)), replace=False
    )
    item_positions = np.full(int(item_ids.max()) + 1, -1, dtype=np.int32)
    item_positions[item_ids] = np.arange(len(item_ids), dtype=np.int32)
    recalls = []
    for position in positions:
        query_id = query_ids[int(position)]
        # np.matmul can emit spurious overflow warnings for large read-only
        # memory maps with Apple's BLAS. The row-wise dot product is equivalent
        # here and remains finite for the normalized float32 embeddings.
        scores = np.einsum(
            "ij,j->i", embeddings, query_embeddings[int(position)], optimize=False
        )
        scores[upload_times[item_ids] >= data.query_times[query_id]] = -np.inf
        for seen_item in data.seen_items[query_id]:
            seen_position = item_positions[seen_item] if seen_item < len(item_positions) else -1
            if seen_position >= 0:
                scores[seen_position] = -np.inf
        top_positions = np.argpartition(scores, -k)[-k:]
        top_positions = top_positions[np.argsort(scores[top_positions])[::-1]]
        exact = set(item_ids[top_positions].tolist())
        approx = set(approximate[query_id][:k])
        recalls.append(len(exact.intersection(approx)) / len(exact))
    return float(np.mean(recalls))


def evaluate_faiss_models(config: dict[str, Any]) -> dict[str, Any]:
    """Index and evaluate encoded models without importing PyTorch."""

    paths = config["paths"]
    evaluation = config["evaluation"]
    faiss.omp_set_num_threads(int(evaluation.get("threads", 4)))
    processed = Path("data/processed")
    data = build_retrieval_period(
        [processed / "standard_train.parquet"],
        processed / "standard_validation.parquet",
        target_column=config["sampling"]["target_column"],
        history_size=int(config["sampling"]["history_length"]),
    )
    arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
    embedding_dir = Path(paths["embedding_dir"])
    ks = [int(value) for value in evaluation["top_k"]]
    results: dict[str, Any] = {"period": "validation", "queries": len(data.targets), "models": {}}
    for mode in ("random", "exposure"):
        item_ids = np.load(embedding_dir / f"item_ids_{mode}.npy", mmap_mode="r")
        item_embeddings = np.load(
            embedding_dir / f"item_embeddings_{mode}.npy", mmap_mode="r"
        )
        query_ids_array = np.load(embedding_dir / f"query_ids_{mode}.npy")
        query_ids = [int(value) for value in query_ids_array]
        query_embeddings = np.load(embedding_dir / f"query_embeddings_{mode}.npy")
        index_path = embedding_dir / f"faiss_{mode}.index"
        index, build_seconds = _build_ivf_index(
            item_embeddings,
            item_ids,
            index_path,
            int(evaluation["nlist"]),
            int(evaluation["nprobe"]),
            int(config["seed"]),
        )
        search_started = time.perf_counter()
        _, raw_ids = index.search(query_embeddings, int(evaluation["search_k"]))
        search_seconds = time.perf_counter() - search_started
        recommendations = _filter_search_results(
            raw_ids, query_ids, data, arrays["upload_time_ms"], max(ks)
        )
        metrics = evaluate_recommendations(
            recommendations, data, ks, catalog_size=len(item_ids)
        )
        ann_recall = _exact_ann_recall(
            item_embeddings,
            item_ids,
            query_ids,
            query_embeddings,
            recommendations,
            data,
            arrays["upload_time_ms"],
            max(ks),
            int(evaluation["exact_check_queries"]),
            int(config["seed"]),
        )
        scorable = 0
        for query_id, targets in data.targets.items():
            target = next(iter(targets))
            if (
                target < len(arrays["present"])
                and arrays["present"][target]
                and arrays["upload_time_ms"][target] < data.query_times[query_id]
            ):
                scorable += 1
        results["models"][mode] = {
            "metrics": asdict(metrics),
            "candidate_items": len(item_ids),
            "scorable_target_fraction": scorable / len(data.targets),
            "ann_recall_at_max_k": ann_recall,
            "timing_seconds": {
                "build_index": build_seconds,
                "search_all_queries": search_seconds,
            },
            "artifact": str(index_path),
        }
    return results
