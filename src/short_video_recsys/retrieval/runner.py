"""Run fixed-protocol retrieval baselines and return serializable results."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from short_video_recsys.retrieval.itemcf import fit_sequential_itemcf, recommend_itemcf
from short_video_recsys.retrieval.metrics import evaluate_recommendations
from short_video_recsys.retrieval.popularity import fit_popularity, recommend_popular
from short_video_recsys.retrieval.protocol import build_retrieval_period


def _period_paths(processed_dir: Path, period: str) -> tuple[list[Path], Path]:
    train = processed_dir / "standard_train.parquet"
    validation = processed_dir / "standard_validation.parquet"
    test = processed_dir / "standard_test.parquet"
    if period == "validation":
        return [train], validation
    if period == "test":
        return [train, validation], test
    raise ValueError(f"Unsupported evaluation period: {period}")


def run_popularity_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Fit/evaluate validation and test independently under the frozen protocol."""

    processed_dir = Path(config["paths"]["processed_dir"])
    target_column = str(config["target_column"])
    history_size = int(config["history_size"])
    ks = [int(k) for k in config["top_k"]]
    max_k = max(ks)

    results: dict[str, Any] = {
        "model": "popularity_positive_count",
        "target_column": target_column,
        "history_size": history_size,
        "top_k": ks,
        "periods": {},
    }
    for period in ("validation", "test"):
        fit_paths, evaluation_path = _period_paths(processed_dir, period)
        data = build_retrieval_period(
            fit_paths,
            evaluation_path,
            target_column=target_column,
            history_size=history_size,
        )
        popularity = fit_popularity(fit_paths, target_column=target_column)
        recommendations = recommend_popular(
            popularity, data.seen_items, sorted(data.targets), k=max_k
        )
        metrics = evaluate_recommendations(recommendations, data, ks)
        results["periods"][period] = asdict(metrics)
    return results


def run_retrieval_baselines(config: dict[str, Any]) -> dict[str, Any]:
    """Run Popularity and Sequential ItemCF under identical period queries."""

    processed_dir = Path(config["paths"]["processed_dir"])
    target_column = str(config["target_column"])
    history_size = int(config["history_size"])
    ks = [int(k) for k in config["top_k"]]
    max_k = max(ks)
    itemcf_config = config["itemcf"]
    context_window = int(itemcf_config["context_window"])
    neighbors_per_item = int(itemcf_config["neighbors_per_item"])
    neighbor_base = Path(itemcf_config["neighbor_file"])

    results: dict[str, Any] = {
        "protocol": {
            "query": "first_novel_long_view_per_user_utc8_day",
            "target_column": target_column,
            "history_size": history_size,
            "top_k": ks,
            "test_refit": "train_plus_validation",
        },
        "models": {
            "popularity_positive_count": {"periods": {}},
            "sequential_itemcf": {
                "context_window": context_window,
                "neighbors_per_item": neighbors_per_item,
                "periods": {},
            },
        },
    }
    for period in ("validation", "test"):
        fit_paths, evaluation_path = _period_paths(processed_dir, period)
        data = build_retrieval_period(
            fit_paths,
            evaluation_path,
            target_column=target_column,
            history_size=history_size,
        )

        popularity = fit_popularity(fit_paths, target_column=target_column)
        popularity_recommendations = recommend_popular(
            popularity, data.seen_items, sorted(data.targets), k=max_k
        )
        popularity_metrics = evaluate_recommendations(popularity_recommendations, data, ks)
        results["models"]["popularity_positive_count"]["periods"][period] = asdict(
            popularity_metrics
        )

        neighbor_path = neighbor_base.with_name(
            f"{neighbor_base.stem}_{period}{neighbor_base.suffix}"
        )
        build_summary = fit_sequential_itemcf(
            fit_paths,
            neighbor_path,
            target_column=target_column,
            context_window=context_window,
            neighbors_per_item=neighbors_per_item,
            temp_directory=neighbor_base.parent / "duckdb_tmp",
        )
        itemcf_recommendations = recommend_itemcf(neighbor_path, data.warm_histories, k=max_k)
        itemcf_metrics = evaluate_recommendations(itemcf_recommendations, data, ks)
        results["models"]["sequential_itemcf"]["periods"][period] = {
            **asdict(itemcf_metrics),
            "artifact": asdict(build_summary),
        }
    return results
