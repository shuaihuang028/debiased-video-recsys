"""Compare the legacy and corrected training pipelines on identical replay queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from short_video_recsys.retrieval.end_to_end_candidates import file_sha256


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_history_alignment(config: dict[str, Any]) -> dict[str, Any]:
    comparison = config["comparison"]
    baseline = _load(comparison["baseline_results"])
    corrected = _load(config["paths"]["results"])
    baseline_dir = Path(baseline["config"]["paths"]["output_dir"])
    corrected_dir = Path(corrected["config"]["paths"]["output_dir"])
    identity_columns = ["query_id", "user_id", "time_ms", "target_id", "target_group", "tab"]
    old_queries = pd.read_parquet(baseline_dir / "queries.parquet")
    new_queries = pd.read_parquet(corrected_dir / "queries.parquet")
    old_queries = old_queries.sort_values("query_id").reset_index(drop=True)
    new_queries = new_queries.sort_values("query_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(old_queries[identity_columns], new_queries[identity_columns])
    controls = ["indexed_items", "candidates", "search_k", "nprobe"]
    for name in controls:
        if baseline["candidate_export"][name] != corrected["candidate_export"][name]:
            raise ValueError(f"Replay comparison must hold {name} fixed")
    if baseline["ranker_sha256"] != corrected["ranker_sha256"]:
        raise ValueError("Replay comparison must use the same LightGBM ranker")
    if baseline["config"]["seed"] != corrected["config"]["seed"]:
        raise ValueError("Replay comparison must use the same seed")
    if baseline["config"]["evaluation"]["top_k"] != corrected["config"]["evaluation"]["top_k"]:
        raise ValueError("Replay comparison must use the same metric cutoffs")
    for result, directory in ((baseline, baseline_dir), (corrected, corrected_dir)):
        query_hash = file_sha256(directory / "queries.parquet")
        if query_hash != result["candidate_export"]["queries_sha256"]:
            raise ValueError("Query artifact hash does not match the saved replay result")
        if result["feature_parity_audit"]["mismatched_values"]:
            raise ValueError("Both replays must pass feature parity checks")

    old_examples = np.load(comparison["baseline_examples"], allow_pickle=False)
    old_histories = old_examples["history_items"]
    old_duplicates = sum(
        len(row[row >= 0]) != len(set(row[row >= 0].tolist())) for row in old_histories
    )
    sample_audit = _load(comparison["corrected_sampling_results"])
    training = _load(comparison["corrected_training_results"])
    for field in (
        "duplicate_history_examples",
        "non_prior_history_examples",
        "non_prior_exposure_negatives",
        "random_negative_future_violations",
        "random_negative_history_collisions",
        "random_negative_positive_collisions",
    ):
        if sample_audit[field] != 0:
            raise ValueError(f"Corrected training samples failed audit: {field}")
    old_ranks = pd.read_parquet(baseline_dir / "query_results.parquet").sort_values("query_id")
    new_ranks = pd.read_parquet(corrected_dir / "query_results.parquet").sort_values("query_id")
    if not np.array_equal(old_ranks["query_id"], new_ranks["query_id"]):
        raise ValueError("Per-query results are not aligned")
    groups = {}
    for group in ("all", "warm", "cold"):
        before, after = baseline["groups"][group], corrected["groups"][group]
        groups[group] = {
            "queries": before["queries"],
            "baseline_candidate_hits": before["candidate_hits"],
            "corrected_candidate_hits": after["candidate_hits"],
            "baseline_top10_hits": before["lightgbm_reranked"]["hits@10"],
            "corrected_top10_hits": after["lightgbm_reranked"]["hits@10"],
            "baseline_candidate_recall": before["candidate_recall"],
            "corrected_candidate_recall": after["candidate_recall"],
            "baseline_end_to_end_ndcg_at_10": before["lightgbm_reranked"]["ndcg@10"],
            "corrected_end_to_end_ndcg_at_10": after["lightgbm_reranked"]["ndcg@10"],
        }
    old_hit = old_ranks["retrieval_target_rank"].to_numpy() > 0
    new_hit = new_ranks["retrieval_target_rank"].to_numpy() > 0
    return {
        "comparison_type": "single_seed_training_pipeline_correction",
        "controls": {
            "identical_query_identities": True,
            "queries": len(old_queries),
            "identical_ranker_sha256": baseline["ranker_sha256"],
            "seed": config["seed"],
            **{name: baseline["candidate_export"][name] for name in controls},
        },
        "baseline_results": comparison["baseline_results"],
        "corrected_results": config["paths"]["results"],
        "baseline_checkpoint_sha256": baseline["query_encoding"]["checkpoint_sha256"],
        "corrected_checkpoint_sha256": corrected["query_encoding"]["checkpoint_sha256"],
        "training_audit": {
            "baseline_sample_rows": len(old_histories),
            "baseline_duplicate_history_examples": old_duplicates,
            "corrected": sample_audit,
            "corrected_training_config": training["config"],
        },
        "groups": groups,
        "retrieval_target_changes": {
            "retained_hits": int((old_hit & new_hit).sum()),
            "gained_hits": int((~old_hit & new_hit).sum()),
            "lost_hits": int((old_hit & ~new_hit).sum()),
        },
        "limitations": [
            "A single seed cannot establish a robust recommendation quality gain.",
            "Correcting sample eligibility and history also changes the reservoir sample.",
            "The combined correction does not isolate causal effects of individual fixes.",
            "Logged-impression ranker metrics and serving benchmarks are separate results.",
        ],
    }
