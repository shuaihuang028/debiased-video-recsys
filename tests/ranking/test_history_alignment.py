from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from short_video_recsys.ranking.end_to_end_metrics import summarize_end_to_end
from short_video_recsys.ranking.history_alignment import compare_history_alignment
from short_video_recsys.retrieval.end_to_end_candidates import file_sha256


def test_comparison_counts_gained_and_lost_queries_and_checks_controls(tmp_path: Path) -> None:
    queries = pd.DataFrame(
        {
            "query_id": [0, 1],
            "user_id": [10, 11],
            "time_ms": [100, 200],
            "target_id": [3, 4],
            "target_group": ["warm", "cold"],
            "tab": [1, 1],
        }
    )
    paths = []
    for mode, ranks in (("baseline", [1, 0]), ("corrected", [0, 2])):
        directory = tmp_path / mode
        directory.mkdir()
        queries.to_parquet(directory / "queries.parquet", index=False)
        per_query = queries[["query_id", "target_group"]].copy()
        per_query["retrieval_target_rank"] = ranks
        per_query["reranked_target_rank"] = ranks
        per_query.to_parquet(directory / "query_results.parquet", index=False)
        results = {
            "config": {
                "paths": {"output_dir": str(directory)},
                "seed": 2026,
                "evaluation": {"top_k": [1, 5, 10]},
            },
            "candidate_export": {
                "queries_sha256": file_sha256(directory / "queries.parquet"),
                "indexed_items": 1000,
                "candidates": 100,
                "search_k": 1000,
                "nprobe": 16,
            },
            "ranker_sha256": "same-ranker",
            "feature_parity_audit": {"mismatched_values": 0},
            "query_encoding": {"checkpoint_sha256": mode},
            "groups": summarize_end_to_end(per_query, [1, 5, 10]),
        }
        result_path = directory / "results.json"
        result_path.write_text(json.dumps(results))
        paths.append(result_path)
    np.savez(tmp_path / "examples.npz", history_items=np.array([[1, 1], [2, -1]]))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                field: 0
                for field in (
                    "duplicate_history_examples",
                    "non_prior_history_examples",
                    "non_prior_exposure_negatives",
                    "random_negative_future_violations",
                    "random_negative_history_collisions",
                    "random_negative_positive_collisions",
                )
            }
        )
    )
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps({"config": {"seed": 2026}}))
    config = {
        "seed": 2026,
        "paths": {"results": str(paths[1])},
        "comparison": {
            "baseline_results": str(paths[0]),
            "baseline_examples": str(tmp_path / "examples.npz"),
            "corrected_sampling_results": str(audit_path),
            "corrected_training_results": str(training_path),
        },
    }
    result = compare_history_alignment(config)
    assert result["controls"]["identical_query_identities"]
    assert result["training_audit"]["baseline_duplicate_history_examples"] == 1
    assert result["retrieval_target_changes"] == {
        "retained_hits": 0,
        "gained_hits": 1,
        "lost_hits": 1,
    }
    assert result["groups"]["all"]["baseline_top10_hits"] == 1
    assert result["groups"]["cold"]["corrected_top10_hits"] == 1
    changed = json.loads(paths[1].read_text())
    changed["ranker_sha256"] = "different-ranker"
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="same LightGBM ranker"):
        compare_history_alignment(config)
    changed["ranker_sha256"] = "same-ranker"
    paths[1].write_text(json.dumps(changed))
    audit_path.write_text(json.dumps({"duplicate_history_examples": 1}))
    with pytest.raises(ValueError, match="failed audit"):
        compare_history_alignment(config)
