#!/usr/bin/env python3
"""Concurrent HTTP benchmark with server-stage latency aggregation."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


async def _request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    user_id: int,
) -> tuple[int, float, dict[str, Any] | None]:
    async with semaphore:
        started = time.perf_counter()
        response = await client.get(f"{url}/recommend/{user_id}", params={"top_k": 10})
        elapsed_ms = (time.perf_counter() - started) * 1_000
        payload = response.json() if response.status_code == 200 else None
        return response.status_code, elapsed_ms, payload


async def benchmark(config: dict[str, Any]) -> dict[str, Any]:
    api = config["api"]
    benchmark_config = config["benchmark"]
    url = f"http://{api['host']}:{api['port']}"
    requests = int(benchmark_config["requests"])
    trials = int(benchmark_config["trials"])
    concurrency = int(benchmark_config["concurrency"])
    warmup = int(benchmark_config["warmup_requests"])
    rng = np.random.default_rng(int(config["seed"]))
    timeout = httpx.Timeout(60.0)
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        health = await client.get(f"{url}/health")
        health.raise_for_status()
        health_payload = health.json()
        for user_id in rng.integers(0, 1_000, size=warmup):
            response = await client.get(f"{url}/recommend/{int(user_id)}")
            response.raise_for_status()

        outputs = []
        trial_wall_seconds = []
        for _ in range(trials):
            users = rng.integers(0, 1_000, size=requests)
            semaphore = asyncio.Semaphore(concurrency)
            started = time.perf_counter()
            trial_outputs = await asyncio.gather(
                *(
                    _request(client, semaphore, url, int(user_id))
                    for user_id in users
                )
            )
            trial_wall_seconds.append(time.perf_counter() - started)
            outputs.extend(trial_outputs)

    statuses = Counter(status for status, _, _ in outputs)
    client_latencies = [latency for status, latency, _ in outputs if status == 200]
    stages: dict[str, list[float]] = defaultdict(list)
    for status, _, payload in outputs:
        if status != 200 or payload is None:
            continue
        for name, value in payload["latency_ms"].items():
            stages[name].append(float(value))
    if not client_latencies:
        raise RuntimeError(f"No successful requests: {dict(statuses)}")
    return {
        "endpoint": "/recommend/{user_id}?top_k=10",
        "feature_store": health_payload["feature_store"],
        "trials": trials,
        "requests_per_trial": requests,
        "requests": requests * trials,
        "successful_requests": len(client_latencies),
        "concurrency": concurrency,
        "warmup_requests": warmup,
        "wall_seconds": float(sum(trial_wall_seconds)),
        "throughput_qps": len(client_latencies) / sum(trial_wall_seconds),
        "trial_throughput_qps": _percentiles(
            [requests / seconds for seconds in trial_wall_seconds]
        ),
        "status_counts": {str(key): value for key, value in sorted(statuses.items())},
        "client_round_trip_ms": _percentiles(client_latencies),
        "server_stage_ms": {
            name: _percentiles(values) for name, values in sorted(stages.items())
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "faiss_nprobe": int(config["retrieval"]["nprobe"]),
            "retrieval_candidates": int(config["retrieval"]["candidates"]),
            "retrieval_workers": int(config["retrieval"].get("workers", 1)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    results = asyncio.run(benchmark(config))
    base_output_path = Path(config["paths"]["benchmark_results"])
    backend = results["feature_store"]
    workers = results["environment"]["retrieval_workers"]
    output_path = base_output_path.with_name(
        f"{base_output_path.stem}_{backend}_w{workers}{base_output_path.suffix}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(output_path)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
