"""Query encoding and FAISS candidate export in separate runtime processes."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from short_video_recsys.retrieval.protocol import build_retrieval_period


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_end_to_end_queries(config: dict[str, Any]) -> dict[str, Any]:
    # Do not import this runtime in the FAISS export process on macOS.
    import torch

    from short_video_recsys.retrieval.two_tower_encoding import _encode_queries, _load_model
    from short_video_recsys.training.item_features import load_item_feature_arrays

    paths = config["paths"]
    torch.set_num_threads(int(config["retrieval"].get("threads", 4)))
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history_length = int(config["retrieval"]["history_length"])
    data = build_retrieval_period(
        [Path(paths["train_interactions"])],
        Path(paths["evaluation_interactions"]),
        history_size=history_length,
    )
    arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
    model = _load_model(Path(paths["two_tower_checkpoint"]), torch.device("cpu"))
    _encode_queries(model, arrays, data, history_length, output_dir, "e2e")
    queries = pd.DataFrame(
        [
            {
                "query_id": query_id,
                "user_id": data.query_users[query_id],
                "time_ms": data.query_times[query_id],
                "target_id": next(iter(targets)),
                "target_group": "warm" if data.warm_targets[query_id] else "cold",
            }
            for query_id, targets in sorted(data.targets.items())
        ]
    )
    with duckdb.connect() as connection:
        connection.register("queries", queries)
        context = connection.execute(
            "SELECT q.query_id, MIN(e.tab) AS tab, COUNT(DISTINCT e.tab) AS tabs "
            "FROM queries q JOIN read_parquet(?) e ON q.user_id = e.user_id "
            "AND q.target_id = e.video_id AND q.time_ms = e.time_ms "
            "WHERE e.long_view = 1 GROUP BY q.query_id ORDER BY q.query_id",
            [str(paths["evaluation_interactions"])],
        ).fetch_df()
    if len(context) != len(queries) or context["tabs"].ne(1).any():
        raise ValueError("Each query must have exactly one unambiguous request context")
    queries = queries.merge(context[["query_id", "tab"]], on="query_id", validate="one_to_one")
    queries.to_parquet(output_dir / "queries.parquet", index=False)
    np.save(
        output_dir / "seen_items.npy",
        np.array(
            [
                data.histories[q] + [-1] * (history_length - len(data.histories[q]))
                for q in queries["query_id"]
            ],
            dtype=np.int32,
        ),
        allow_pickle=False,
    )
    return {
        "queries": len(queries),
        "history_rule": "strictly_before_query_timestamp",
        "checkpoint_sha256": file_sha256(Path(paths["two_tower_checkpoint"])),
        "queries_sha256": file_sha256(output_dir / "queries.parquet"),
        "checkpoint_history_policy": model.history_policy,
        "checkpoint_timestamp_policy": model.timestamp_policy,
    }


def select_candidates(
    raw_ids: np.ndarray,
    raw_scores: np.ndarray,
    query_time_ms: int,
    seen_items: set[int],
    upload_times: np.ndarray,
    k: int,
) -> tuple[list[int], list[float]]:
    selected: list[int] = []
    scores: list[float] = []
    used: set[int] = set()
    for raw_id, raw_score in zip(raw_ids, raw_scores):
        item = int(raw_id)
        if (
            item < 0
            or item in seen_items
            or item in used
            or upload_times[item] <= 0
            or upload_times[item] >= query_time_ms
        ):
            continue
        selected.append(item)
        scores.append(float(raw_score))
        used.add(item)
        if len(selected) == k:
            break
    return selected, scores


def export_end_to_end_candidates(config: dict[str, Any]) -> dict[str, Any]:
    import faiss

    paths = config["paths"]
    retrieval = config["retrieval"]
    k = int(retrieval["candidates"])
    search_k = int(retrieval["search_k"])
    if k <= 0 or search_k < k:
        raise ValueError("Require search_k >= candidates > 0")
    output_dir = Path(paths["output_dir"])
    queries = pd.read_parquet(output_dir / "queries.parquet")
    query_ids = np.load(output_dir / "query_ids_e2e.npy")
    if not np.array_equal(query_ids, queries["query_id"].to_numpy()):
        raise ValueError("Encoded query IDs do not match query metadata")
    vectors = np.load(output_dir / "query_embeddings_e2e.npy")
    histories = np.load(output_dir / "seen_items.npy")
    upload_times = np.load(Path(paths["item_feature_dir"]) / "upload_time_ms.npy", mmap_mode="r")
    index = faiss.read_index(str(paths["faiss_index"]))
    index.nprobe = int(retrieval["nprobe"])
    faiss.omp_set_num_threads(int(retrieval["threads"]))
    started = time.perf_counter()
    raw_scores, raw_ids = index.search(vectors, search_k)
    search_seconds = time.perf_counter() - started
    rows = []
    for position, query in enumerate(queries.itertuples(index=False)):
        items, scores = select_candidates(
            raw_ids[position],
            raw_scores[position],
            query.time_ms,
            set(histories[position].tolist()),
            upload_times,
            k,
        )
        rows.extend(
            (query.query_id, item, rank, score)
            for rank, (item, score) in enumerate(zip(items, scores), start=1)
        )
    candidates = pd.DataFrame(
        rows, columns=["query_id", "video_id", "candidate_rank", "retrieval_score"]
    )
    candidates.to_parquet(output_dir / "candidates.parquet", index=False)
    return {
        "queries": len(queries),
        "candidate_rows": len(candidates),
        "indexed_items": int(index.ntotal),
        "nprobe": int(index.nprobe),
        "candidates": k,
        "search_k": search_k,
        "search_seconds": search_seconds,
        "target_injection": False,
        "upload_rule": "upload_time_ms < query_time_ms",
        "queries_sha256": file_sha256(output_dir / "queries.parquet"),
        "candidates_sha256": file_sha256(output_dir / "candidates.parquet"),
        "index_sha256": file_sha256(Path(paths["faiss_index"])),
    }
