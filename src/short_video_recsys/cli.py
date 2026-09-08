"""Unified command line entry point for reproducible recommendation experiments."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from short_video_recsys.config import load_yaml, require_mapping
from short_video_recsys.data.schema import validate_csv, validate_interactions
from short_video_recsys.data.split import TimeBoundaries, temporal_split
from short_video_recsys.features.duckdb_pipeline import (
    build_pit_parquet,
    build_random_pit_parquet,
)
from short_video_recsys.logging_utils import configure_logging
from short_video_recsys.retrieval.runner import (
    run_popularity_baseline,
    run_retrieval_baselines,
)
from short_video_recsys.training.item_features import prepare_item_features
from short_video_recsys.training.sampling import build_training_examples

LOGGER = logging.getLogger(__name__)


def _resolve_raw_files(config: dict[str, Any]) -> dict[str, Any]:
    paths = require_mapping(config, "paths")
    raw_dir = Path(paths["raw_dir"])
    standard_logs = [raw_dir / name for name in paths["standard_logs"]]
    return {
        "standard": standard_logs,
        "random": raw_dir / paths["random_log"],
        "processed": Path(paths["processed_dir"]),
        "results": Path(paths["results_dir"]),
    }


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))


def _standard_boundaries(config: dict[str, Any]) -> TimeBoundaries:
    split = require_mapping(config, "standard_split")
    return TimeBoundaries(
        first_end_ms=int(split["train_end_ms"]),
        second_end_ms=int(split["validation_end_ms"]),
        third_end_ms=int(split["test_end_ms"]),
    )


def _random_boundaries(config: dict[str, Any]) -> TimeBoundaries:
    split = require_mapping(config, "random_split")
    return TimeBoundaries(
        first_end_ms=int(split["calibration_end_ms"]),
        second_end_ms=int(split["validation_end_ms"]),
        third_end_ms=int(split["test_end_ms"]),
    )


def _write_partitions(partitions: Mapping[str, pd.DataFrame], output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in partitions.items():
        destination = output_dir / f"{stem}_{name}.parquet"
        frame.to_parquet(destination, index=False)
        LOGGER.info(
            "wrote_partition",
            extra={"fields": {"path": destination, "partition": name, "rows": len(frame)}},
        )


def _partition_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "users": int(frame["user_id"].nunique()),
        "items": int(frame["video_id"].nunique()),
        "min_time_ms": int(frame["time_ms"].min()),
        "max_time_ms": int(frame["time_ms"].max()),
        "click_rate": float(frame["is_click"].mean()),
        "like_rate": float(frame["is_like"].mean()),
        "long_view_rate": float(frame["long_view"].mean()),
        "zero_duration_rows": int(frame["duration_ms"].eq(0).sum()),
    }


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)
    LOGGER.info("wrote_audit", extra={"fields": {"path": path}})


def validate_data(config: dict[str, Any]) -> None:
    paths = _resolve_raw_files(config)
    inputs = [*paths["standard"], paths["random"]]
    _require_files(inputs)
    audit = {}
    for path in inputs:
        summary = validate_csv(path)
        LOGGER.info("validated_file", extra={"fields": {"path": path, **asdict(summary)}})
        audit[path.name] = asdict(summary)
    _write_json(audit, paths["results"] / "raw_validation.json")


def split_data(config: dict[str, Any]) -> None:
    paths = _resolve_raw_files(config)
    inputs = [*paths["standard"], paths["random"]]
    _require_files(inputs)

    standard = pd.concat((pd.read_csv(path) for path in paths["standard"]), ignore_index=True)
    validate_interactions(standard)
    standard_parts = temporal_split(standard, _standard_boundaries(config))
    _write_partitions(standard_parts, paths["processed"], "standard")

    random = pd.read_csv(paths["random"])
    validate_interactions(random)
    random_parts = temporal_split(
        random,
        _random_boundaries(config),
        first_name="calibration",
        second_name="validation",
        third_name="test",
    )
    _write_partitions(random_parts, paths["processed"], "random")
    audit = {
        "standard": {
            name: _partition_summary(frame) for name, frame in standard_parts.items()
        },
        "random": {name: _partition_summary(frame) for name, frame in random_parts.items()},
    }
    _write_json(audit, paths["results"] / "data_splits.json")


def build_features(config: dict[str, Any]) -> None:
    paths = _resolve_raw_files(config)
    _require_files(paths["standard"])
    feature_config = require_mapping(config, "features")
    engine = feature_config.get("engine", "duckdb")
    if engine != "duckdb":
        raise ValueError(f"Unsupported full-data feature engine: {engine}")
    smoothing = feature_config["smoothing"]
    output_path = paths["processed"] / feature_config["output_file"]
    summary = build_pit_parquet(
        input_paths=paths["standard"],
        output_path=output_path,
        boundaries=_standard_boundaries(config),
        windows_ms=feature_config["windows_ms"],
        watch_ratio_cap=float(feature_config["watch_ratio_cap"]),
        smoothing_alpha=float(smoothing["alpha"]),
        smoothing_beta=float(smoothing["beta"]),
        temp_directory=Path(feature_config["temp_dir"]),
    )
    LOGGER.info("built_features", extra={"fields": asdict(summary)})


def build_random_features(config: dict[str, Any]) -> None:
    paths = _resolve_raw_files(config)
    processed = paths["processed"]
    feature_config = require_mapping(config, "features")
    smoothing = feature_config["smoothing"]
    standard_paths = [
        processed / f"standard_{split}.parquet"
        for split in ("train", "validation", "test")
    ]
    random_paths = [
        processed / f"random_{split}.parquet"
        for split in ("calibration", "validation", "test")
    ]
    summary = build_random_pit_parquet(
        standard_paths=standard_paths,
        random_paths=random_paths,
        output_path=processed / feature_config["random_output_file"],
        boundaries=_random_boundaries(config),
        windows_ms=feature_config["windows_ms"],
        watch_ratio_cap=float(feature_config["watch_ratio_cap"]),
        smoothing_alpha=float(smoothing["alpha"]),
        smoothing_beta=float(smoothing["beta"]),
        temp_directory=Path(feature_config["temp_dir"]),
    )
    LOGGER.info("built_random_features", extra={"fields": asdict(summary)})


def eval_popularity(config: dict[str, Any]) -> None:
    results = run_popularity_baseline(config)
    output_path = Path(config["paths"]["results_file"])
    _write_json(results, output_path)


def eval_retrieval(config: dict[str, Any]) -> None:
    results = run_retrieval_baselines(config)
    output_path = Path(config["paths"]["results_file"])
    _write_json(results, output_path)


def prepare_items(config: dict[str, Any]) -> None:
    paths = config["paths"]
    features = config["item_features"]
    summary = prepare_item_features(
        metadata_path=Path(paths["item_metadata"]),
        output_dir=Path(paths["item_feature_dir"]),
        author_buckets=int(features["author_buckets"]),
        music_buckets=int(features["music_buckets"]),
        max_tags=int(features["max_tags"]),
    )
    LOGGER.info("prepared_item_features", extra={"fields": asdict(summary)})


def build_examples(config: dict[str, Any]) -> None:
    paths = config["paths"]
    sampling = config["sampling"]
    summary = build_training_examples(
        interactions_path=Path(paths["train_interactions"]),
        item_feature_dir=Path(paths["item_feature_dir"]),
        output_path=Path(paths["training_examples"]),
        summary_path=Path(paths["sampling_summary"]),
        seed=int(config["seed"]),
        history_length=int(sampling["history_length"]),
        exposure_pool_size=int(sampling["exposure_pool_size"]),
        max_examples=int(sampling["max_examples"]),
    )
    LOGGER.info("built_training_examples", extra={"fields": asdict(summary)})


def train_towers(config: dict[str, Any]) -> None:
    # Keep PyTorch out of commands that use FAISS. On macOS, importing both
    # runtimes in one process can initialize duplicate OpenMP libraries.
    from short_video_recsys.training.train_two_tower import train_negative_ablation

    results = train_negative_ablation(config)
    _write_json(results, Path(config["paths"]["training_results"]))


def encode_towers(config: dict[str, Any]) -> None:
    from short_video_recsys.retrieval.two_tower_encoding import encode_two_tower_models

    results = encode_two_tower_models(config)
    _write_json(results, Path(config["paths"]["encoding_results"]))


def eval_towers(config: dict[str, Any]) -> None:
    from short_video_recsys.retrieval.faiss_eval import evaluate_faiss_models

    results = evaluate_faiss_models(config)
    _write_json(results, Path(config["paths"]["retrieval_results"]))


def train_single_task_ranker(config: dict[str, Any]) -> None:
    from short_video_recsys.ranking.train_ranker import train_ranker

    results = train_ranker(config)
    _write_json(results, Path(config["paths"]["results"]))


def train_multitask_rankers(config: dict[str, Any]) -> None:
    from short_video_recsys.ranking.train_multitask import train_multitask_comparison

    results = train_multitask_comparison(config)
    _write_json(results, Path(config["paths"]["results"]))


def eval_randomized(config: dict[str, Any]) -> None:
    from short_video_recsys.ranking.train_ranker import evaluate_saved_ranker

    results = evaluate_saved_ranker(config)
    _write_json(results, Path(config["paths"]["randomized_results"]))


def train_debiased_ranker(config: dict[str, Any]) -> None:
    from short_video_recsys.debias.density_ratio import train_reweighted_ranker

    results = train_reweighted_ranker(config)
    _write_json(results, Path(config["paths"]["debias_results"]))


def train_random_adaptation(config: dict[str, Any]) -> None:
    from short_video_recsys.debias.randomized_adaptation import (
        train_randomized_adaptation,
    )

    results = train_randomized_adaptation(config)
    _write_json(results, Path(config["paths"]["adaptation_results"]))


def prepare_serving_snapshot(config: dict[str, Any]) -> None:
    from short_video_recsys.serving.snapshot import build_serving_snapshot

    results = build_serving_snapshot(config)
    _write_json(results, Path(config["paths"]["snapshot_dir"]) / "build_summary.json")


def serve_recommender(config: dict[str, Any]) -> None:
    from short_video_recsys.serving.app import serve

    serve(config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "validate-data",
        "split-data",
        "build-features",
        "build-random-features",
        "eval-popularity",
        "eval-retrieval",
        "prepare-item-features",
        "build-training-examples",
        "train-two-tower",
        "encode-two-tower",
        "eval-two-tower",
        "train-ranker",
        "train-multitask-rankers",
        "eval-randomized",
        "train-debiased-ranker",
        "train-random-adaptation",
        "prepare-serving-snapshot",
        "serve",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    configure_logging()
    args = _parser().parse_args()
    config = load_yaml(args.config)
    commands = {
        "validate-data": validate_data,
        "split-data": split_data,
        "build-features": build_features,
        "build-random-features": build_random_features,
        "eval-popularity": eval_popularity,
        "eval-retrieval": eval_retrieval,
        "prepare-item-features": prepare_items,
        "build-training-examples": build_examples,
        "train-two-tower": train_towers,
        "encode-two-tower": encode_towers,
        "eval-two-tower": eval_towers,
        "train-ranker": train_single_task_ranker,
        "train-multitask-rankers": train_multitask_rankers,
        "eval-randomized": eval_randomized,
        "train-debiased-ranker": train_debiased_ranker,
        "train-random-adaptation": train_random_adaptation,
        "prepare-serving-snapshot": prepare_serving_snapshot,
        "serve": serve_recommender,
    }
    commands[args.command](config)


if __name__ == "__main__":
    main()
