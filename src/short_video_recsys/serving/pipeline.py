"""Two-stage online recommendation orchestration."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from short_video_recsys.serving.feature_store import (
    FeatureStore,
    MemoryFeatureStore,
    RedisFeatureStore,
)
from short_video_recsys.serving.workers import (
    IsolatedWorker,
    WorkerPool,
    ranking_worker,
    retrieval_worker,
)


class RecommendationPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        feature_store: FeatureStore | None = None,
        retrieval: IsolatedWorker | WorkerPool | None = None,
        ranker: IsolatedWorker | None = None,
    ):
        self.config = config
        self.as_of_ms = int(config["snapshot"]["as_of_ms"])
        backend = os.environ.get(
            "RECSYS_FEATURE_STORE", str(config["api"]["feature_store"])
        )
        snapshot_dir = Path(config["paths"]["snapshot_dir"])
        if feature_store is not None:
            self.feature_store = feature_store
        elif backend == "memory":
            self.feature_store = MemoryFeatureStore(snapshot_dir)
        elif backend == "redis":
            redis_url = os.environ.get(
                "RECSYS_REDIS_URL", str(config["api"]["redis_url"])
            )
            self.feature_store = RedisFeatureStore(snapshot_dir, redis_url)
        else:
            raise ValueError(f"Unsupported feature store backend: {backend}")
        self.retrieval = retrieval or retrieval_worker(config)
        self.ranker = ranker or ranking_worker(config)

    @property
    def backend_name(self) -> str:
        return self.feature_store.backend_name

    def recommend(self, user_id: int, top_k: int, tab: int) -> dict[str, Any] | None:
        started = time.perf_counter()
        feature_started = time.perf_counter()
        user = self.feature_store.get_user(user_id)
        feature_ms = (time.perf_counter() - feature_started) * 1_000
        if user is None:
            return None

        retrieval_started = time.perf_counter()
        candidates = self.retrieval.call(
            {
                "embedding": user.embedding,
                "seen_items": user.recent_items,
                "as_of_ms": self.as_of_ms,
                "search_k": int(self.config["retrieval"]["search_k"]),
                "candidates": int(self.config["retrieval"]["candidates"]),
            }
        )
        retrieval_total_ms = (time.perf_counter() - retrieval_started) * 1_000
        ranking_started = time.perf_counter()
        ranked = self.ranker.call(
            {
                "item_ids": candidates["item_ids"],
                "user_id": user_id,
                "user_history": user.history_features,
                "as_of_ms": self.as_of_ms,
                "tab": tab,
            }
        )
        ranking_total_ms = (time.perf_counter() - ranking_started) * 1_000
        item_ids = ranked["item_ids"][:top_k]
        ranking_scores = ranked["ranking_scores"][:top_k]
        total_ms = (time.perf_counter() - started) * 1_000
        return {
            "user_id": user_id,
            "as_of_ms": self.as_of_ms,
            "items": [
                {"video_id": item, "score": score}
                for item, score in zip(item_ids, ranking_scores)
            ],
            "feature_store": self.backend_name,
            "latency_ms": {
                "feature_fetch": feature_ms,
                "faiss_search": candidates["worker_ms"],
                "retrieval_queue_and_ipc": max(
                    0.0, retrieval_total_ms - candidates["worker_ms"]
                ),
                "ranking_feature_assembly": ranked["feature_assembly_ms"],
                "ranking_inference": ranked["inference_ms"],
                "ranking_sort": ranked["sort_ms"],
                "ranking_queue_and_ipc": max(
                    0.0,
                    ranking_total_ms
                    - ranked["feature_assembly_ms"]
                    - ranked["inference_ms"]
                    - ranked["sort_ms"],
                ),
                "total": total_ms,
            },
        }

    def close(self) -> None:
        self.retrieval.close()
        self.ranker.close()
        self.feature_store.close()
