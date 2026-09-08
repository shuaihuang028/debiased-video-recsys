"""Process-isolated FAISS and LightGBM inference workers."""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

DAY_MS = 86_400_000


def _retrieval_worker(
    connection: Any,
    index_path: str,
    item_feature_dir: str,
    nprobe: int,
) -> None:
    try:
        import faiss

        index = faiss.read_index(index_path)
        index.nprobe = nprobe
        upload_times = np.load(
            Path(item_feature_dir) / "upload_time_ms.npy", mmap_mode="r"
        )
        connection.send({"ready": True})
        while True:
            request = connection.recv()
            if request is None:
                break
            started = time.perf_counter()
            vector = np.asarray(request["embedding"], dtype=np.float32).reshape(1, -1)
            scores, ids = index.search(vector, int(request["search_k"]))
            seen = set(int(item) for item in request["seen_items"])
            selected_ids = []
            selected_scores = []
            for item_raw, score_raw in zip(ids[0], scores[0]):
                item = int(item_raw)
                if (
                    item < 0
                    or item in seen
                    or upload_times[item] >= int(request["as_of_ms"])
                ):
                    continue
                selected_ids.append(item)
                selected_scores.append(float(score_raw))
                if len(selected_ids) == int(request["candidates"]):
                    break
            connection.send(
                {
                    "item_ids": selected_ids,
                    "retrieval_scores": selected_scores,
                    "worker_ms": (time.perf_counter() - started) * 1_000,
                }
            )
    except BaseException as error:
        connection.send({"ready": False, "error": repr(error)})
    finally:
        connection.close()


def _ranker_matrix(
    item_ids: np.ndarray,
    user_id: int,
    user_history: np.ndarray,
    as_of_ms: int,
    tab: int,
    item_arrays: dict[str, np.ndarray],
    item_history: np.ndarray,
    user_columns: list[str],
    item_columns: list[str],
) -> Any:
    import pandas as pd

    from short_video_recsys.ranking.train_ranker import MODEL_FEATURES

    size = len(item_ids)
    local_ms = as_of_ms + 8 * 60 * 60 * 1_000
    local_hour = (local_ms % DAY_MS) / (DAY_MS / 24)
    local_weekday = (local_ms // DAY_MS + 3) % 7
    data: dict[str, Any] = {
        "user_id": np.full(size, user_id, dtype=np.int32),
        "tab": np.full(size, tab, dtype=np.int16),
        "local_hour_sin": np.full(size, np.sin(2 * np.pi * local_hour / 24), np.float32),
        "local_hour_cos": np.full(size, np.cos(2 * np.pi * local_hour / 24), np.float32),
        "local_weekday_sin": np.full(
            size, np.sin(2 * np.pi * local_weekday / 7), np.float32
        ),
        "local_weekday_cos": np.full(
            size, np.cos(2 * np.pi * local_weekday / 7), np.float32
        ),
    }
    for name in ("author_bucket", "music_bucket", "upload_type", "video_type", "music_type"):
        data[name] = np.asarray(item_arrays[name][item_ids], dtype=np.int32)
    tags = np.asarray(item_arrays["tags"][item_ids], dtype=np.int32)
    for position in range(tags.shape[1]):
        data[f"tag_{position}"] = tags[:, position]
    data["duration_log_seconds"] = np.asarray(
        item_arrays["duration_log_seconds"][item_ids], dtype=np.float32
    )
    data["aspect_ratio"] = np.asarray(
        item_arrays["aspect_ratio"][item_ids], dtype=np.float32
    )
    upload_times = np.asarray(item_arrays["upload_time_ms"][item_ids], dtype=np.int64)
    present = np.asarray(item_arrays["present"][item_ids]) & (upload_times > 0)
    data["item_age_days"] = np.where(
        present, np.maximum(as_of_ms - upload_times, 0) / DAY_MS, 0
    ).astype(np.float32)
    data["item_metadata_missing"] = (~present).astype(np.float32)
    for position, name in enumerate(user_columns):
        data[name] = np.full(size, user_history[position], dtype=np.float32)
    selected_item_history = np.asarray(item_history[item_ids], dtype=np.float32)
    for position, name in enumerate(item_columns):
        data[name] = selected_item_history[:, position]
    return pd.DataFrame(data).loc[:, MODEL_FEATURES]


def _ranking_worker(
    connection: Any,
    model_path: str,
    item_feature_dir: str,
    snapshot_dir: str,
) -> None:
    try:
        if sys.platform == "darwin":
            import torch  # noqa: F401
        import lightgbm as lgb

        from short_video_recsys.training.item_features import load_item_feature_arrays

        snapshot_path = Path(snapshot_dir)
        with (snapshot_path / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        item_arrays = load_item_feature_arrays(Path(item_feature_dir))
        item_history = np.load(
            snapshot_path / "item_history_features.npy", mmap_mode="r"
        )
        model = lgb.Booster(model_file=model_path)
        connection.send({"ready": True})
        while True:
            request = connection.recv()
            if request is None:
                break
            assembly_started = time.perf_counter()
            item_ids = np.asarray(request["item_ids"], dtype=np.int64)
            matrix = _ranker_matrix(
                item_ids,
                int(request["user_id"]),
                np.asarray(request["user_history"], dtype=np.float32),
                int(request["as_of_ms"]),
                int(request["tab"]),
                item_arrays,
                item_history,
                metadata["user_history_columns"],
                metadata["item_history_columns"],
            )
            assembly_ms = (time.perf_counter() - assembly_started) * 1_000
            inference_started = time.perf_counter()
            scores = np.asarray(model.predict(matrix), dtype=np.float32)
            inference_ms = (time.perf_counter() - inference_started) * 1_000
            sort_started = time.perf_counter()
            order = np.argsort(-scores, kind="stable")
            connection.send(
                {
                    "item_ids": item_ids[order].astype(int).tolist(),
                    "ranking_scores": scores[order].astype(float).tolist(),
                    "feature_assembly_ms": assembly_ms,
                    "inference_ms": inference_ms,
                    "sort_ms": (time.perf_counter() - sort_started) * 1_000,
                }
            )
    except BaseException as error:
        connection.send({"ready": False, "error": repr(error)})
    finally:
        connection.close()


class IsolatedWorker:
    """Synchronous RPC over a Pipe, with heavy runtimes confined to a child process."""

    def __init__(
        self,
        target: Callable[..., None],
        args: tuple[Any, ...],
        startup_timeout: float = 60.0,
    ):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=target, args=(child, *args), daemon=True)
        self.lock = threading.Lock()
        self.process.start()
        if not parent.poll(startup_timeout):
            self.close()
            raise TimeoutError("Inference worker did not start in time")
        status = parent.recv()
        if not status.get("ready"):
            self.close()
            raise RuntimeError(f"Inference worker failed: {status.get('error')}")

    def call(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        with self.lock:
            self.connection.send(payload)
            if not self.connection.poll(timeout):
                raise TimeoutError("Inference worker request timed out")
            response = self.connection.recv()
        if response.get("ready") is False:
            raise RuntimeError(f"Inference worker failed: {response.get('error')}")
        return response

    def close(self) -> None:
        if getattr(self, "process", None) is None:
            return
        if self.process.is_alive():
            try:
                self.connection.send(None)
                self.process.join(timeout=5)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()


class WorkerPool:
    """Lease independent model processes so concurrent requests need not serialize."""

    def __init__(self, workers: list[IsolatedWorker]):
        if not workers:
            raise ValueError("WorkerPool requires at least one worker")
        self.workers = workers
        self.available: queue.Queue[IsolatedWorker] = queue.Queue()
        for worker in workers:
            self.available.put(worker)

    def call(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        try:
            worker = self.available.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("No inference worker became available") from error
        try:
            return worker.call(payload, timeout=timeout)
        finally:
            self.available.put(worker)

    def close(self) -> None:
        for worker in self.workers:
            worker.close()


def retrieval_worker(config: dict[str, Any]) -> WorkerPool:
    arguments = (
        str(config["paths"]["faiss_index"]),
        str(config["paths"]["item_feature_dir"]),
        int(config["retrieval"]["nprobe"]),
    )
    return WorkerPool(
        [
            IsolatedWorker(_retrieval_worker, arguments)
            for _ in range(int(config["retrieval"].get("workers", 1)))
        ]
    )


def ranking_worker(config: dict[str, Any]) -> IsolatedWorker:
    return IsolatedWorker(
        _ranking_worker,
        (
            str(config["paths"]["ranker_model"]),
            str(config["paths"]["item_feature_dir"]),
            str(config["paths"]["snapshot_dir"]),
        ),
    )
