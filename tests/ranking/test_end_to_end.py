from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from short_video_recsys.features.candidate_history import materialize_candidate_history
from short_video_recsys.features.point_in_time import build_point_in_time_features
from short_video_recsys.ranking.end_to_end import build_query_results
from short_video_recsys.ranking.end_to_end import evaluate_end_to_end as replay
from short_video_recsys.ranking.end_to_end_metrics import summarize_end_to_end
from short_video_recsys.ranking.train_ranker import MODEL_FEATURES, RAW_HISTORY_FEATURES
from short_video_recsys.retrieval.end_to_end_candidates import file_sha256, select_candidates


def test_candidate_history_matches_reference_at_arbitrary_query_times(tmp_path: Path) -> None:
    events = pd.DataFrame(
        [
            (1, 10, 1000, 1, 0, 1, 2000, 1000),
            (1, 10, 1000, 0, 1, 0, 9000, 1000),
            (2, 11, 1001, 1, 0, 1, 5000, 0),
            (1, 11, 3_601_000, 1, 1, 1, 1000, 1000),
            (1, 10, 900_000_000, 1, 1, 1, 1000, 1000),
        ],
        columns=[
            "user_id",
            "video_id",
            "time_ms",
            "is_click",
            "is_like",
            "long_view",
            "play_time_ms",
            "duration_ms",
        ],
    )
    path = tmp_path / "events.parquet"
    events.to_parquet(path, index=False)
    requests = pd.DataFrame(
        [
            (0, 1, 10, 1000),
            (1, 1, 10, 1001),
            (2, 1, 10, 3_601_000),
            (3, 1, 10, 3_601_001),
            (4, 2, 11, 604_801_002),
            (5, 99, 99, 10_000),
        ],
        columns=["request_id", "user_id", "video_id", "time_ms"],
    )
    output = tmp_path / "features.parquet"
    with duckdb.connect() as connection:
        materialize_candidate_history(connection, [path], requests, output)
    actual = pd.read_parquet(output).set_index("request_id")
    for request in requests.itertuples(index=False):
        probe = {
            "user_id": request.user_id,
            "video_id": request.video_id,
            "time_ms": request.time_ms,
            "is_click": 0,
            "is_like": 0,
            "long_view": 0,
            "play_time_ms": 0,
            "duration_ms": 1000,
        }
        reference = build_point_in_time_features(
            pd.concat([events, pd.DataFrame([probe])], ignore_index=True)
        ).iloc[-1]
        np.testing.assert_allclose(
            actual.loc[request.request_id, list(RAW_HISTORY_FEATURES)].to_numpy(float),
            reference.loc[list(RAW_HISTORY_FEATURES)].to_numpy(float),
            rtol=1e-8,
            atol=1e-8,
        )
    assert actual.loc[0, "item_hist_impression_count"] == 0
    assert actual.loc[2, "item_1h_impression_count"] == 2
    assert actual.loc[3, "item_1h_impression_count"] == 0
    assert actual.loc[5, "item_hist_long_view_rate"] == pytest.approx(1 / 11)


def test_candidate_selection_filters_unavailable_seen_and_duplicate_items() -> None:
    ids, scores = select_candidates(
        np.array([-1, 1, 2, 3, 4, 4, 5, 6]),
        np.arange(8, dtype=float),
        query_time_ms=100,
        seen_items={1},
        upload_times=np.array([0, 10, 100, 101, 99, 0, 20]),
        k=2,
    )
    assert ids == [4, 6]
    assert scores == [4.0, 7.0]


def test_end_to_end_metrics_keep_retrieval_misses_and_empty_queries() -> None:
    queries = pd.DataFrame(
        {
            "query_id": [0, 1, 2, 3],
            "user_id": [10, 11, 12, 13],
            "target_id": [100, 200, 300, 400],
            "target_group": ["warm", "warm", "cold", "cold"],
        }
    )
    scores = pd.DataFrame(
        [
            (0, 101, 1, 0.2),
            (0, 100, 2, 0.9),
            (1, 201, 1, 0.5),
            (2, 300, 1, 0.4),
            (2, 301, 2, 0.4),
        ],
        columns=["query_id", "video_id", "candidate_rank", "ranking_score"],
    )
    per_query = build_query_results(queries, scores)
    assert per_query["retrieval_target_rank"].tolist() == [2, 0, 1, 0]
    assert per_query["reranked_target_rank"].tolist() == [1, 0, 1, 0]
    assert per_query["candidate_count"].tolist() == [2, 1, 2, 0]
    results = summarize_end_to_end(per_query, [1, 2])
    assert results["all"]["candidate_hits"] == 2
    assert results["all"]["retrieval_order"]["hitrate@1"] == 0.25
    assert results["all"]["lightgbm_reranked"]["hitrate@1"] == 0.5
    assert results["all"]["lightgbm_reranked"]["ndcg@2"] == 0.5
    assert results["all"]["retrieval_order"]["mrr@2"] == 0.375
    assert results["all"]["conditional_on_retrieved"]["lightgbm_reranked"]["hitrate@1"] == 1
    assert results["warm"]["queries"] + results["cold"]["queries"] == results["all"]["queries"]


def test_empty_groups_are_undefined_and_membership_changes_are_rejected() -> None:
    queries = pd.DataFrame(
        {
            "target_group": ["warm"],
            "retrieval_target_rank": [0],
            "reranked_target_rank": [0],
        }
    )
    results = summarize_end_to_end(queries, [10])
    assert results["cold"]["candidate_recall"] is None
    assert results["warm"]["conditional_on_retrieved"]["retrieval_order"]["ndcg@10"] is None
    assert results["all"]["candidate_recall_wilson_95_ci"][1] > 0
    queries.loc[0, "reranked_target_rank"] = 1
    with pytest.raises(ValueError, match="preserve target membership"):
        summarize_end_to_end(queries, [10])


def test_replay_scores_only_exported_candidates_and_audits_every_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from short_video_recsys.ranking import end_to_end

    columns = [
        "user_id",
        "video_id",
        "time_ms",
        "is_click",
        "is_like",
        "long_view",
        "play_time_ms",
        "duration_ms",
        "tab",
    ]
    train = pd.DataFrame(
        [
            (1, 0, 1, 1, 0, 1, 1000, 1000, 0),
            (2, 1, 2, 1, 0, 1, 1000, 1000, 0),
        ],
        columns=columns,
    )
    evaluation = pd.DataFrame(
        [
            (1, 1, 100, 1, 0, 1, 1000, 1000, 0),
            (2, 2, 101, 1, 0, 1, 1000, 1000, 0),
        ],
        columns=columns,
    )
    train.to_parquet(tmp_path / "train.parquet", index=False)
    evaluation.to_parquet(tmp_path / "evaluation.parquet", index=False)
    reference = build_point_in_time_features(pd.concat([train, evaluation], ignore_index=True))
    reference["split"] = ["train", "train", "validation", "validation"]
    reference.to_parquet(tmp_path / "reference.parquet", index=False)
    queries = pd.DataFrame(
        {
            "query_id": [0, 1],
            "user_id": [1, 2],
            "time_ms": [100, 101],
            "target_id": [1, 2],
            "target_group": ["warm", "cold"],
            "tab": [0, 0],
        }
    )
    queries.to_parquet(tmp_path / "queries.parquet", index=False)
    candidates = pd.DataFrame(
        [
            (0, 0, 1, 0.9),
            (0, 1, 2, 0.8),
            (1, 0, 1, 0.9),
        ],
        columns=["query_id", "video_id", "candidate_rank", "retrieval_score"],
    )
    candidates.to_parquet(tmp_path / "candidates.parquet", index=False)
    provenance = {"queries_sha256": file_sha256(tmp_path / "queries.parquet")}
    (tmp_path / "encoding_summary.json").write_text(json.dumps(provenance))
    provenance["candidates_sha256"] = file_sha256(tmp_path / "candidates.parquet")
    (tmp_path / "candidate_summary.json").write_text(json.dumps(provenance))
    (tmp_path / "model.txt").write_text("test-model")
    arrays = {
        name: np.ones(3, dtype=np.int32)
        for name in ("author_bucket", "music_bucket", "upload_type", "video_type", "music_type")
    }
    arrays.update(
        {
            "present": np.ones(3, dtype=bool),
            "tags": np.zeros((3, 5), dtype=np.int32),
            "upload_time_ms": np.ones(3, dtype=np.int64),
            "duration_log_seconds": np.ones(3, dtype=np.float32),
            "aspect_ratio": np.arange(3, dtype=np.float32),
        }
    )
    scored_rows = []

    class Model:
        def feature_name(self):
            return list(MODEL_FEATURES)

        def predict(self, matrix, **kwargs):
            scored_rows.append(len(matrix))
            return matrix["aspect_ratio"].to_numpy()

    monkeypatch.setattr(end_to_end, "load_item_feature_arrays", lambda _: arrays)
    monkeypatch.setattr(
        end_to_end, "_lightgbm", lambda: SimpleNamespace(Booster=lambda **_: Model())
    )
    config = {
        "paths": {
            "output_dir": str(tmp_path),
            "train_interactions": str(tmp_path / "train.parquet"),
            "evaluation_interactions": str(tmp_path / "evaluation.parquet"),
            "reference_features": str(tmp_path / "reference.parquet"),
            "item_feature_dir": str(tmp_path),
            "ranker_model": str(tmp_path / "model.txt"),
        },
        "evaluation": {"threads": 1, "memory_limit": "256MB", "batch_rows": 2, "top_k": [1]},
        "retrieval": {"candidates": 2},
    }
    result = replay(config)
    assert sum(scored_rows) == 3
    assert result["feature_parity_audit"]["queries_checked"] == 2
    assert result["feature_parity_audit"]["mismatched_values"] == 0
    assert result["groups"]["all"]["candidate_hits"] == 1
    assert result["groups"]["all"]["lightgbm_reranked"]["hitrate@1"] == 0.5
    assert result["groups"]["cold"]["candidate_hits"] == 0
