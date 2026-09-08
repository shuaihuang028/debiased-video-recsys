"""Randomized-calibration density ratios and reweighted ranker training."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from short_video_recsys.ranking.train_ranker import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    SOURCE_COLUMNS,
    _escaped,
    _feature_frame,
    _fit_lightgbm,
    _lightgbm,
    _predict_split,
    _sample,
)
from short_video_recsys.training.item_features import load_item_feature_arrays


def density_ratio_weights(
    random_probability: np.ndarray, lower: float, upper: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Convert balanced-domain probabilities to clipped, mean-one density ratios."""

    if not 0 < lower <= 1 <= upper:
        raise ValueError("Require 0 < lower <= 1 <= upper")
    probability = np.asarray(random_probability, dtype=np.float64)
    if len(probability) == 0 or not np.isfinite(probability).all():
        raise ValueError("Probabilities must be a non-empty finite array")
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    raw = probability / (1 - probability)
    clipped = np.clip(raw, lower, upper)
    normalized = clipped / clipped.mean()
    effective_sample_size = float(normalized.sum() ** 2 / np.square(normalized).sum())
    diagnostics = {
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_p01": float(np.quantile(raw, 0.01)),
        "raw_p50": float(np.quantile(raw, 0.50)),
        "raw_p99": float(np.quantile(raw, 0.99)),
        "fraction_clipped_low": float(np.mean(raw < lower)),
        "fraction_clipped_high": float(np.mean(raw > upper)),
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
        "effective_sample_size": effective_sample_size,
        "effective_sample_fraction": effective_sample_size / len(normalized),
    }
    return normalized.astype(np.float32), diagnostics


def _load_rows(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    predicate: str,
) -> pd.DataFrame:
    columns = ", ".join(SOURCE_COLUMNS)
    return connection.execute(
        f"""
        SELECT {columns}
        FROM read_parquet('{_escaped(path)}')
        WHERE {predicate}
        """
    ).fetch_df()


def _fit_domain_classifier(
    lgb: Any,
    standard: pd.DataFrame,
    randomized: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    standard_matrix = _feature_frame(standard, arrays)
    randomized_matrix = _feature_frame(randomized, arrays)
    matrix = pd.concat([standard_matrix, randomized_matrix], ignore_index=True)
    labels = np.concatenate(
        [
            np.zeros(len(standard_matrix), dtype=np.int8),
            np.ones(len(randomized_matrix), dtype=np.int8),
        ]
    )
    rng = np.random.default_rng(seed)
    train_positions = []
    validation_positions = []
    validation_fraction = float(config["validation_fraction"])
    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        positions = rng.permutation(positions)
        validation_size = max(1, int(len(positions) * validation_fraction))
        validation_positions.append(positions[:validation_size])
        train_positions.append(positions[validation_size:])
    train_indices = np.concatenate(train_positions)
    validation_indices = np.concatenate(validation_positions)

    parameters = dict(config["model"])
    num_boost_round = int(parameters.pop("num_boost_round"))
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds"))
    log_period = int(parameters.pop("log_period"))
    parameters.update(
        {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "force_col_wise": True,
        }
    )
    train_dataset = lgb.Dataset(
        matrix.iloc[train_indices],
        label=labels[train_indices],
        categorical_feature=list(CATEGORICAL_FEATURES),
    )
    validation_dataset = lgb.Dataset(
        matrix.iloc[validation_indices],
        label=labels[validation_indices],
        categorical_feature=list(CATEGORICAL_FEATURES),
        reference=train_dataset,
    )
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[validation_dataset],
        valid_names=["domain_validation"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=True),
            lgb.log_evaluation(log_period),
        ],
    )
    summary = {
        "standard_rows": len(standard),
        "randomized_rows": len(randomized),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "best_iteration": model.best_iteration,
        "validation_auc": float(model.best_score["domain_validation"]["auc"]),
        "seconds": time.perf_counter() - started,
    }
    importance = model.feature_importance(importance_type="gain")
    summary["top_features_by_gain"] = [
        {"feature": name, "gain": float(gain)}
        for name, gain in sorted(
            zip(model.feature_name(), importance.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )[:15]
    ]
    return model, summary


def train_reweighted_ranker(config: dict[str, Any]) -> dict[str, Any]:
    """Train a density-ratio-weighted ranker using random calibration only."""

    lgb = _lightgbm()
    seed = int(config["seed"])
    paths = config["paths"]
    standard_path = Path(paths["standard_features"])
    random_path = Path(paths["random_features"])
    arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
    connection = duckdb.connect()
    randomized_calibration = _load_rows(connection, random_path, "split = 'calibration'")
    calibration_end_ms = int(randomized_calibration["time_ms"].max())
    standard_calibration = _sample(
        connection,
        standard_path,
        f"split = 'validation' AND time_ms <= {calibration_end_ms}",
        len(randomized_calibration),
        seed,
    )
    domain_model, domain_summary = _fit_domain_classifier(
        lgb,
        standard_calibration,
        randomized_calibration,
        arrays,
        config["propensity"],
        seed,
    )
    density_model_path = Path(paths["density_model"])
    density_model_path.parent.mkdir(parents=True, exist_ok=True)
    domain_model.save_model(density_model_path, num_iteration=domain_model.best_iteration)

    training = config["training"]
    holdout_fraction = float(training["internal_validation_fraction"])
    cutoff = int(
        connection.execute(
            f"""
            SELECT quantile_disc(time_ms, {1 - holdout_fraction})
            FROM read_parquet('{_escaped(standard_path)}')
            WHERE split = 'train'
            """
        ).fetchone()[0]
    )
    train_frame = _sample(
        connection,
        standard_path,
        f"split = 'train' AND time_ms <= {cutoff}",
        int(training["max_train_rows"]),
        seed,
    )
    validation_frame = _sample(
        connection,
        standard_path,
        f"split = 'train' AND time_ms > {cutoff}",
        int(training["max_internal_validation_rows"]),
        seed + 1,
    )
    connection.close()
    train_matrix = _feature_frame(train_frame, arrays)
    validation_matrix = _feature_frame(validation_frame, arrays)
    best_iteration = domain_model.best_iteration if domain_model.best_iteration > 0 else -1
    train_probability = domain_model.predict(train_matrix, num_iteration=best_iteration)
    validation_probability = domain_model.predict(
        validation_matrix, num_iteration=best_iteration
    )
    lower = float(config["weighting"]["clip_min"])
    upper = float(config["weighting"]["clip_max"])
    train_weights, train_weight_summary = density_ratio_weights(
        train_probability, lower, upper
    )
    validation_weights, validation_weight_summary = density_ratio_weights(
        validation_probability, lower, upper
    )
    train_labels = train_frame["long_view"].to_numpy(dtype=np.int8)
    validation_labels = validation_frame["long_view"].to_numpy(dtype=np.int8)
    ranker, training_seconds = _fit_lightgbm(
        lgb,
        train_matrix,
        train_labels,
        validation_matrix,
        validation_labels,
        config["ranking_model"],
        seed,
        train_weights,
        validation_weights,
    )
    ranker_path = Path(paths["debiased_model"])
    ranker_path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_model(ranker_path, num_iteration=ranker.best_iteration)
    batch_rows = int(config["evaluation"]["batch_rows"])
    random_validation = _predict_split(
        ranker, random_path, arrays, batch_rows, "validation"
    )
    random_test = _predict_split(ranker, random_path, arrays, batch_rows, "test")
    return {
        "method": "clipped_self_normalized_density_ratio",
        "estimand": "randomized_exposure_covariate_distribution",
        "assumption": "conditional outcome mechanism is stable after observed covariates",
        "seed": seed,
        "feature_count": len(MODEL_FEATURES),
        "domain_classifier": domain_summary,
        "weighting": {
            "clip_min": lower,
            "clip_max": upper,
            "train": train_weight_summary,
            "internal_validation": validation_weight_summary,
            "weighted_train_positive_rate": float(
                np.average(train_labels, weights=train_weights)
            ),
        },
        "training": {
            "rows": len(train_frame),
            "internal_validation_rows": len(validation_frame),
            "split_time_ms": cutoff,
            "best_iteration": ranker.best_iteration,
            "seconds": training_seconds,
        },
        "random_validation": random_validation,
        "random_test": random_test,
        "artifacts": {
            "density_model": str(density_model_path),
            "debiased_ranker": str(ranker_path),
        },
    }
