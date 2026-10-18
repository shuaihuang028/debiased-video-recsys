"""Offline replay of retrieved candidates through the saved LightGBM ranker."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from short_video_recsys.features.candidate_history import materialize_candidate_history
from short_video_recsys.ranking.end_to_end_metrics import summarize_end_to_end
from short_video_recsys.ranking.train_ranker import (
    MODEL_FEATURES,
    RAW_HISTORY_FEATURES,
    _feature_frame,
    _lightgbm,
)
from short_video_recsys.retrieval.end_to_end_candidates import file_sha256
from short_video_recsys.training.item_features import load_item_feature_arrays

LOGGER = logging.getLogger(__name__)


def _audit_feature_parity(
    connection: duckdb.DuckDBPyConnection,
    queries: pd.DataFrame,
    features_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    connection.register("audit_queries", queries)
    columns = ", ".join(f"f.{name}" for name in RAW_HISTORY_FEATURES)
    reference = connection.execute(
        f"SELECT DISTINCT q.query_id, {columns} FROM audit_queries q "
        "JOIN read_parquet(?) f ON q.user_id = f.user_id AND q.target_id = f.video_id "
        "AND q.time_ms = f.time_ms AND q.tab = f.tab "
        "WHERE f.split = 'validation' ORDER BY q.query_id",
        [str(reference_path)],
    ).fetch_df()
    replay = connection.execute(
        f"SELECT query_id, {', '.join(RAW_HISTORY_FEATURES)} FROM read_parquet(?) "
        "WHERE is_audit ORDER BY query_id",
        [str(features_path)],
    ).fetch_df()
    connection.unregister("audit_queries")
    if (
        len(reference) != len(queries)
        or len(replay) != len(queries)
        or not np.array_equal(reference["query_id"], replay["query_id"])
    ):
        raise ValueError("Feature parity audit must cover every query exactly once")
    expected = reference.loc[:, RAW_HISTORY_FEATURES].to_numpy(dtype=np.float64)
    actual = replay.loc[:, RAW_HISTORY_FEATURES].to_numpy(dtype=np.float64)
    matches = np.isclose(actual, expected, rtol=1e-6, atol=1e-5)
    if not matches.all():
        failed_columns = [RAW_HISTORY_FEATURES[i] for i in np.flatnonzero(~matches.all(axis=0))]
        raise ValueError(
            f"Replayed historical features differ from training features: {failed_columns}"
        )
    return {
        "queries_checked": len(queries),
        "historical_features_checked": len(RAW_HISTORY_FEATURES),
        "mismatched_values": int((~matches).sum()),
        "max_absolute_error": float(np.max(np.abs(actual - expected))),
        "rtol": 1e-6,
        "atol": 1e-5,
        "audit_targets_scored_or_added_to_candidates": False,
    }


def build_query_results(queries: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Keep empty candidate sets and break score ties by original retrieval rank."""
    rows = []
    candidate_groups = dict(tuple(scores.groupby("query_id", sort=False))) if len(scores) else {}
    for query in queries.itertuples(index=False):
        frame = candidate_groups.get(query.query_id)
        before = after = count = 0
        if frame is not None:
            count = len(frame)
            if frame["video_id"].duplicated().any():
                raise ValueError("Candidates must not contain duplicate items")
            frame = frame.sort_values("candidate_rank", kind="stable")
            target = frame["video_id"].eq(query.target_id)
            if target.any():
                before = int(frame.loc[target, "candidate_rank"].iloc[0])
                reranked = frame.sort_values(
                    ["ranking_score", "candidate_rank"], ascending=[False, True], kind="stable"
                )
                after = int(np.flatnonzero(reranked["video_id"].eq(query.target_id))[0]) + 1
        rows.append(
            {
                "query_id": query.query_id,
                "user_id": query.user_id,
                "target_group": query.target_group,
                "candidate_count": count,
                "retrieval_target_rank": before,
                "reranked_target_rank": after,
            }
        )
    return pd.DataFrame(rows)


def evaluate_end_to_end(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    paths = config["paths"]
    evaluation = config["evaluation"]
    output_dir = Path(paths["output_dir"])
    queries_path = output_dir / "queries.parquet"
    candidates_path = output_dir / "candidates.parquet"
    encoding = json.loads((output_dir / "encoding_summary.json").read_text())
    candidate_summary = json.loads((output_dir / "candidate_summary.json").read_text())
    for key, path in (("queries_sha256", queries_path), ("candidates_sha256", candidates_path)):
        if candidate_summary[key] != file_sha256(path):
            raise ValueError(f"Candidate provenance mismatch: {key}")
    if encoding["queries_sha256"] != candidate_summary["queries_sha256"]:
        raise ValueError("Candidate export does not match the encoded queries")
    queries = pd.read_parquet(queries_path).sort_values("query_id").reset_index(drop=True)
    candidates = pd.read_parquet(candidates_path)
    if candidates.duplicated(["query_id", "video_id"]).any():
        raise ValueError("Candidate pairs must be unique")
    if not candidates["query_id"].isin(queries["query_id"]).all():
        raise ValueError("Unknown query IDs in candidate export")
    requests = candidates.merge(queries, on="query_id", validate="many_to_one")
    requests["is_audit"] = False
    audit_requests = queries.rename(columns={"target_id": "video_id"}).copy()
    audit_requests["is_audit"] = True
    audit_requests["candidate_rank"] = 0
    audit_requests["retrieval_score"] = np.nan
    requests = pd.concat([requests, audit_requests], ignore_index=True)
    # Target IDs are used only for outcome metrics and separate feature parity rows.
    requests = requests.drop(columns=["target_id", "target_group"], errors="ignore")
    requests["request_id"] = np.arange(len(requests), dtype=np.int64)
    feature_path = output_dir / "candidate_features.parquet"
    connection = duckdb.connect()
    connection.execute("SET threads = ?", [int(evaluation["threads"])])
    connection.execute("SET memory_limit = ?", [str(evaluation["memory_limit"])])
    temp_dir = output_dir / "duckdb_tmp"
    temp_dir.mkdir(exist_ok=True)
    connection.execute("SET temp_directory = ?", [str(temp_dir)])
    try:
        feature_started = time.perf_counter()
        materialize_candidate_history(
            connection,
            [Path(paths["train_interactions"]), Path(paths["evaluation_interactions"])],
            requests,
            feature_path,
        )
        feature_seconds = time.perf_counter() - feature_started
        LOGGER.info("Checking replay features against all logged query targets")
        audit = _audit_feature_parity(
            connection, queries, feature_path, Path(paths["reference_features"])
        )
        arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
        lgb = _lightgbm()
        model = lgb.Booster(model_file=str(paths["ranker_model"]))
        if model.feature_name() != list(MODEL_FEATURES):
            raise ValueError("Saved ranker feature contract does not match replay code")
        scored = []
        prediction_started = time.perf_counter()
        reader = connection.execute(
            "SELECT * FROM read_parquet(?) WHERE NOT is_audit ORDER BY request_id",
            [str(feature_path)],
        ).fetch_record_batch(rows_per_batch=int(evaluation["batch_rows"]))
        for batch in reader:
            frame = batch.to_pandas()
            matrix = _feature_frame(frame, arrays)
            if not np.isfinite(matrix.to_numpy()).all():
                raise ValueError("Candidate model features must all be finite")
            scores = model.predict(matrix, num_threads=int(evaluation["threads"]))
            if not np.isfinite(scores).all():
                raise ValueError("Ranker returned non-finite scores")
            part = frame[["query_id", "video_id", "candidate_rank", "retrieval_score"]].copy()
            part["ranking_score"] = scores
            scored.append(part)
        scored_frame = (
            pd.concat(scored, ignore_index=True)
            if scored
            else pd.DataFrame(
                columns=[
                    "query_id",
                    "video_id",
                    "candidate_rank",
                    "retrieval_score",
                    "ranking_score",
                ]
            )
        )
        prediction_seconds = time.perf_counter() - prediction_started
    finally:
        connection.close()
    if len(scored_frame) != len(candidates):
        raise ValueError("Exactly the exported candidates must be scored")
    scored_frame.to_parquet(output_dir / "scored_candidates.parquet", index=False)
    query_results = build_query_results(queries, scored_frame)
    query_results.to_parquet(output_dir / "query_results.parquet", index=False)
    ks = sorted(set(int(k) for k in evaluation["top_k"]))
    if any(k > int(config["retrieval"]["candidates"]) for k in ks):
        raise ValueError("Top-k cutoffs cannot exceed candidate count")
    return {
        "period": "standard_validation",
        "evaluation_type": "single_logged_target_replay",
        "protocol": {
            "query": "first novel long-view target per user per UTC+8 day",
            "history": "standard events strictly before each query timestamp; tied labels excluded",
            "warm_target": "target appeared in standard-train interactions, regardless of label",
            "cold_target": "target absent from standard-train interactions",
            "candidate_policy": "actual FAISS candidates; no target injection",
            "request_context": "tab from the logged target request, shared by all its candidates",
            "missing_target_rank": 0,
            "score_tie_break": "original retrieval rank",
            "confidence_intervals": "95% Wilson query-proportion intervals; not user-clustered",
        },
        "limitations": [
            "One logged positive is judged per query; other candidates have unknown relevance.",
            "These metrics do not estimate online engagement or causal policy value.",
            "Sparse candidate hits limit conclusions about reranking quality.",
            "Historical retrieval used row-wise updates at tied timestamps; replay fixes them.",
        ],
        "config": config,
        "query_encoding": encoding,
        "candidate_export": candidate_summary,
        "ranker_sha256": file_sha256(Path(paths["ranker_model"])),
        "feature_parity_audit": audit,
        "candidate_counts": {
            "min": int(query_results["candidate_count"].min()),
            "max": int(query_results["candidate_count"].max()),
            "empty_queries": int(query_results["candidate_count"].eq(0).sum()),
            "total_rows": len(scored_frame),
        },
        "groups": summarize_end_to_end(query_results, ks),
        "retrieved_target_rank_transitions": query_results.loc[
            query_results["retrieval_target_rank"].gt(0),
            ["target_group", "retrieval_target_rank", "reranked_target_rank"],
        ].to_dict(orient="records"),
        "timing_seconds": {
            "build_features": feature_seconds,
            "ranker_prediction": prediction_seconds,
            "total_replay": time.perf_counter() - started,
        },
    }
