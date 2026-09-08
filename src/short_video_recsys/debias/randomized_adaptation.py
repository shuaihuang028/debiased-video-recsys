"""Outcome adaptation using only the reserved randomized calibration period."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from short_video_recsys.ranking.metrics import evaluate_scores
from short_video_recsys.ranking.train_ranker import (
    _escaped,
    _feature_frame,
    _fit_lightgbm,
    _lightgbm,
    _predict_split,
    _sample,
)
from short_video_recsys.training.item_features import load_item_feature_arrays

DAY_MS = 86_400_000


def randomized_example_weight(
    standard_rows: int, randomized_rows: int, randomized_mass: float
) -> float:
    """Return per-randomized-row weight for a requested total training mass."""

    if standard_rows <= 0 or randomized_rows <= 0:
        raise ValueError("Training row counts must be positive")
    if not 0 < randomized_mass < 1:
        raise ValueError("randomized_mass must be between zero and one")
    return randomized_mass * standard_rows / ((1 - randomized_mass) * randomized_rows)


def _calibration_metrics(
    model: Any, matrix: pd.DataFrame, frame: pd.DataFrame
) -> dict[str, int | float]:
    best_iteration = model.best_iteration if model.best_iteration > 0 else -1
    predictions = model.predict(matrix, num_iteration=best_iteration)
    timestamps = frame["time_ms"].to_numpy(dtype=np.int64)
    local_days = (timestamps + 8 * 60 * 60 * 1_000) // DAY_MS
    return evaluate_scores(
        frame["long_view"].to_numpy(dtype=np.int8),
        predictions,
        frame["user_id"].to_numpy(dtype=np.int32),
        local_days,
    ).to_dict()


def train_randomized_adaptation(config: dict[str, Any]) -> dict[str, Any]:
    """Select random-calibration mixing strength without touching later periods."""

    lgb = _lightgbm()
    seed = int(config["seed"])
    paths = config["paths"]
    standard_path = Path(paths["standard_features"])
    random_path = Path(paths["random_features"])
    arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
    connection = duckdb.connect()
    training = config["training"]
    holdout_fraction = float(training["internal_validation_fraction"])
    standard_cutoff = int(
        connection.execute(
            f"""
            SELECT quantile_disc(time_ms, {1 - holdout_fraction})
            FROM read_parquet('{_escaped(standard_path)}')
            WHERE split = 'train'
            """
        ).fetchone()[0]
    )
    standard_frame = _sample(
        connection,
        standard_path,
        f"split = 'train' AND time_ms <= {standard_cutoff}",
        int(training["max_train_rows"]),
        seed,
    )
    calibration_fraction = float(config["adaptation"]["calibration_train_fraction"])
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_train_fraction must be between zero and one")
    random_cutoff = int(
        connection.execute(
            f"""
            SELECT quantile_disc(time_ms, {calibration_fraction})
            FROM read_parquet('{_escaped(random_path)}')
            WHERE split = 'calibration'
            """
        ).fetchone()[0]
    )
    random_train = _sample(
        connection,
        random_path,
        f"split = 'calibration' AND time_ms <= {random_cutoff}",
        1_000_000,
        seed,
    )
    random_holdout = _sample(
        connection,
        random_path,
        f"split = 'calibration' AND time_ms > {random_cutoff}",
        1_000_000,
        seed,
    )
    connection.close()
    standard_matrix = _feature_frame(standard_frame, arrays)
    random_train_matrix = _feature_frame(random_train, arrays)
    random_holdout_matrix = _feature_frame(random_holdout, arrays)
    train_matrix = pd.concat([standard_matrix, random_train_matrix], ignore_index=True)
    train_labels = np.concatenate(
        [
            standard_frame["long_view"].to_numpy(dtype=np.int8),
            random_train["long_view"].to_numpy(dtype=np.int8),
        ]
    )
    validation_labels = random_holdout["long_view"].to_numpy(dtype=np.int8)
    candidates = []
    best_model = None
    best_ndcg = -np.inf
    selected_mass = None
    for mass_raw in config["adaptation"]["randomized_mass_candidates"]:
        mass = float(mass_raw)
        random_weight = randomized_example_weight(
            len(standard_frame), len(random_train), mass
        )
        weights = np.concatenate(
            [
                np.ones(len(standard_frame), dtype=np.float32),
                np.full(len(random_train), random_weight, dtype=np.float32),
            ]
        )
        model, seconds = _fit_lightgbm(
            lgb,
            train_matrix,
            train_labels,
            random_holdout_matrix,
            validation_labels,
            config["ranking_model"],
            seed,
            train_weights=weights,
        )
        metrics = _calibration_metrics(model, random_holdout_matrix, random_holdout)
        candidates.append(
            {
                "randomized_mass": mass,
                "randomized_example_weight": random_weight,
                "best_iteration": model.best_iteration,
                "seconds": seconds,
                "calibration_holdout": metrics,
            }
        )
        if metrics["user_day_ndcg_at_10"] > best_ndcg:
            best_ndcg = float(metrics["user_day_ndcg_at_10"])
            best_model = model
            selected_mass = mass
        gc.collect()
    if best_model is None or selected_mass is None:
        raise AssertionError("No randomized-mass candidate was trained")
    model_path = Path(paths["adapted_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_model.save_model(model_path, num_iteration=best_model.best_iteration)
    batch_rows = int(config["evaluation"]["batch_rows"])
    return {
        "method": "randomized_calibration_outcome_adaptation",
        "selection_metric": "calibration_holdout_user_day_ndcg_at_10",
        "seed": seed,
        "standard_train_rows": len(standard_frame),
        "random_calibration_train_rows": len(random_train),
        "random_calibration_holdout_rows": len(random_holdout),
        "standard_split_time_ms": standard_cutoff,
        "random_calibration_split_time_ms": random_cutoff,
        "candidates": candidates,
        "selected_randomized_mass": selected_mass,
        "random_validation": _predict_split(
            best_model, random_path, arrays, batch_rows, "validation"
        ),
        "random_test": _predict_split(
            best_model, random_path, arrays, batch_rows, "test"
        ),
        "artifact": str(model_path),
    }
