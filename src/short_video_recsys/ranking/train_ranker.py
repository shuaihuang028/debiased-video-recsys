"""Leakage-safe LightGBM ranker training and standard-traffic evaluation."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from short_video_recsys.ranking.metrics import evaluate_scores
from short_video_recsys.training.item_features import load_item_feature_arrays

DAY_MS = 86_400_000

WINDOWS = ("hist", "1h", "1d", "7d")
SCOPES = ("user", "item")
COUNT_SUFFIXES = (
    "impression_count",
    "click_count",
    "like_count",
    "long_view_count",
    "watch_ratio_sum",
)
RATE_SUFFIXES = ("ctr", "like_rate", "long_view_rate", "mean_watch_ratio")
RAW_HISTORY_FEATURES = tuple(
    f"{scope}_{window}_{suffix}"
    for scope in SCOPES
    for window in WINDOWS
    for suffix in (*COUNT_SUFFIXES, *RATE_SUFFIXES)
)
LOG_HISTORY_FEATURES = tuple(
    f"log1p_{name}" if name.endswith(COUNT_SUFFIXES) else name
    for name in RAW_HISTORY_FEATURES
)
CONTENT_FEATURES = (
    "author_bucket",
    "music_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "tag_0",
    "tag_1",
    "tag_2",
    "tag_3",
    "tag_4",
    "duration_log_seconds",
    "aspect_ratio",
    "item_age_days",
    "item_metadata_missing",
)
TIME_FEATURES = ("local_hour_sin", "local_hour_cos", "local_weekday_sin", "local_weekday_cos")
MODEL_FEATURES = (
    "user_id",
    "tab",
    *TIME_FEATURES,
    *CONTENT_FEATURES,
    *LOG_HISTORY_FEATURES,
)
CATEGORICAL_FEATURES = (
    "user_id",
    "tab",
    "author_bucket",
    "music_bucket",
    "upload_type",
    "video_type",
    "music_type",
    "tag_0",
    "tag_1",
    "tag_2",
    "tag_3",
    "tag_4",
)
SOURCE_COLUMNS = ("user_id", "video_id", "time_ms", "tab", "long_view", *RAW_HISTORY_FEATURES)


def _lightgbm() -> Any:
    # The macOS LightGBM wheel expects libomp in Homebrew/MacPorts locations.
    # PyTorch already ships the same runtime in this environment, so load it
    # first rather than requiring a machine-global package. This job never
    # imports FAISS, whose separate OpenMP runtime is isolated in its own CLI.
    if sys.platform == "darwin":
        import torch  # noqa: F401

    import lightgbm

    return lightgbm


def _escaped(path: Path) -> str:
    return str(path).replace("'", "''")


def _sample(
    connection: duckdb.DuckDBPyConnection,
    feature_path: Path,
    predicate: str,
    rows: int,
    seed: int,
) -> pd.DataFrame:
    columns = ", ".join(SOURCE_COLUMNS)
    query = f"""
        SELECT {columns}
        FROM (
            SELECT {columns}
            FROM read_parquet('{_escaped(feature_path)}')
            WHERE {predicate}
        )
        USING SAMPLE reservoir({rows} ROWS) REPEATABLE({seed})
    """
    return connection.execute(query).fetch_df()


def _feature_frame(
    frame: pd.DataFrame, arrays: dict[str, np.ndarray]
) -> pd.DataFrame:
    item_ids = frame["video_id"].to_numpy(dtype=np.int64)
    if len(item_ids) and (item_ids.min() < 0 or item_ids.max() >= len(arrays["present"])):
        raise ValueError("Encountered video_id outside item feature arrays")
    output = pd.DataFrame(index=frame.index)
    output["user_id"] = frame["user_id"].to_numpy(dtype=np.int32)
    output["tab"] = frame["tab"].to_numpy(dtype=np.int16)

    timestamps = frame["time_ms"].to_numpy(dtype=np.int64)
    local_ms = timestamps + 8 * 60 * 60 * 1_000
    local_hours = (local_ms % DAY_MS) / (DAY_MS / 24)
    local_days = local_ms // DAY_MS
    local_weekdays = (local_days + 3) % 7
    output["local_hour_sin"] = np.sin(2 * np.pi * local_hours / 24).astype(np.float32)
    output["local_hour_cos"] = np.cos(2 * np.pi * local_hours / 24).astype(np.float32)
    output["local_weekday_sin"] = np.sin(2 * np.pi * local_weekdays / 7).astype(np.float32)
    output["local_weekday_cos"] = np.cos(2 * np.pi * local_weekdays / 7).astype(np.float32)

    for name in ("author_bucket", "music_bucket", "upload_type", "video_type", "music_type"):
        output[name] = np.asarray(arrays[name][item_ids], dtype=np.int32)
    tags = np.asarray(arrays["tags"][item_ids], dtype=np.int32)
    for position in range(tags.shape[1]):
        output[f"tag_{position}"] = tags[:, position]
    output["duration_log_seconds"] = np.asarray(
        arrays["duration_log_seconds"][item_ids], dtype=np.float32
    )
    output["aspect_ratio"] = np.asarray(arrays["aspect_ratio"][item_ids], dtype=np.float32)
    upload_times = np.asarray(arrays["upload_time_ms"][item_ids], dtype=np.int64)
    present = np.asarray(arrays["present"][item_ids]) & (upload_times > 0)
    output["item_age_days"] = np.where(
        present, np.maximum(timestamps - upload_times, 0) / DAY_MS, 0
    ).astype(np.float32)
    output["item_metadata_missing"] = (~present).astype(np.float32)

    for raw_name, model_name in zip(RAW_HISTORY_FEATURES, LOG_HISTORY_FEATURES):
        values = frame[raw_name].to_numpy(dtype=np.float32)
        if model_name.startswith("log1p_"):
            values = np.log1p(np.maximum(values, 0))
        output[model_name] = values
    return output.loc[:, MODEL_FEATURES]


def _predict_split(
    model: Any,
    feature_path: Path,
    arrays: dict[str, np.ndarray],
    batch_rows: int,
    split: str,
) -> dict[str, Any]:
    if split not in {"calibration", "validation", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split}")
    columns = ", ".join(SOURCE_COLUMNS)
    connection = duckdb.connect()
    reader = connection.execute(
        f"""
        SELECT {columns}
        FROM read_parquet('{_escaped(feature_path)}')
        WHERE split = '{split}'
        """
    ).fetch_record_batch(rows_per_batch=batch_rows)
    labels_parts = []
    predictions_parts = []
    baselines_parts = []
    users_parts = []
    days_parts = []
    started = time.perf_counter()
    for batch in reader:
        frame = batch.to_pandas()
        matrix = _feature_frame(frame, arrays)
        best_iteration = model.best_iteration if model.best_iteration > 0 else -1
        predictions_parts.append(
            np.asarray(model.predict(matrix, num_iteration=best_iteration), dtype=np.float32)
        )
        labels_parts.append(frame["long_view"].to_numpy(dtype=np.int8))
        baselines_parts.append(
            frame["item_hist_long_view_rate"].to_numpy(dtype=np.float32)
        )
        users_parts.append(frame["user_id"].to_numpy(dtype=np.int32))
        days_parts.append(
            ((frame["time_ms"].to_numpy(dtype=np.int64) + 8 * 60 * 60 * 1_000) // DAY_MS)
            .astype(np.int32)
        )
    connection.close()
    labels = np.concatenate(labels_parts)
    predictions = np.concatenate(predictions_parts)
    baselines = np.concatenate(baselines_parts)
    users = np.concatenate(users_parts)
    days = np.concatenate(days_parts)
    return {
        "model": evaluate_scores(labels, predictions, users, days).to_dict(),
        "historical_item_rate_baseline": evaluate_scores(
            labels, baselines, users, days
        ).to_dict(),
        "prediction_seconds": time.perf_counter() - started,
    }


def _fit_lightgbm(
    lgb: Any,
    train_matrix: pd.DataFrame,
    train_labels: np.ndarray,
    validation_matrix: pd.DataFrame,
    validation_labels: np.ndarray,
    model_config: dict[str, Any],
    seed: int,
    train_weights: np.ndarray | None = None,
    validation_weights: np.ndarray | None = None,
) -> tuple[Any, float]:
    train_dataset = lgb.Dataset(
        train_matrix,
        label=train_labels,
        weight=train_weights,
        categorical_feature=list(CATEGORICAL_FEATURES),
        free_raw_data=True,
    )
    validation_dataset = lgb.Dataset(
        validation_matrix,
        label=validation_labels,
        weight=validation_weights,
        categorical_feature=list(CATEGORICAL_FEATURES),
        reference=train_dataset,
        free_raw_data=True,
    )
    parameters = dict(model_config)
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
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[validation_dataset],
        valid_names=["internal_validation"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=True),
            lgb.log_evaluation(log_period),
        ],
    )
    return model, time.perf_counter() - started


def train_ranker(config: dict[str, Any]) -> dict[str, Any]:
    """Train on a chronological train subset and report untouched validation metrics."""

    lgb = _lightgbm()

    paths = config["paths"]
    training = config["training"]
    feature_path = Path(paths["features"])
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    seed = int(config["seed"])
    holdout_fraction = float(training["internal_validation_fraction"])
    if not 0 < holdout_fraction < 1:
        raise ValueError("internal_validation_fraction must be between zero and one")

    connection = duckdb.connect()
    cutoff = int(
        connection.execute(
            f"""
            SELECT quantile_disc(time_ms, {1 - holdout_fraction})
            FROM read_parquet('{_escaped(feature_path)}')
            WHERE split = 'train'
            """
        ).fetchone()[0]
    )
    train_frame = _sample(
        connection,
        feature_path,
        f"split = 'train' AND time_ms <= {cutoff}",
        int(training["max_train_rows"]),
        seed,
    )
    internal_validation_frame = _sample(
        connection,
        feature_path,
        f"split = 'train' AND time_ms > {cutoff}",
        int(training["max_internal_validation_rows"]),
        seed + 1,
    )
    connection.close()
    arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
    train_matrix = _feature_frame(train_frame, arrays)
    internal_validation_matrix = _feature_frame(internal_validation_frame, arrays)
    train_labels = train_frame["long_view"].to_numpy(dtype=np.int8)
    internal_validation_labels = internal_validation_frame["long_view"].to_numpy(
        dtype=np.int8
    )

    model, training_seconds = _fit_lightgbm(
        lgb,
        train_matrix,
        train_labels,
        internal_validation_matrix,
        internal_validation_labels,
        config["model"],
        seed,
    )
    model_path = Path(paths["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path, num_iteration=model.best_iteration)

    evaluation = _predict_split(
        model,
        feature_path,
        arrays,
        int(config["evaluation"]["batch_rows"]),
        "validation",
    )
    importance = model.feature_importance(importance_type="gain")
    top_features = sorted(
        zip(model.feature_name(), importance.tolist()), key=lambda pair: pair[1], reverse=True
    )[:20]
    return {
        "objective": "long_view_binary",
        "seed": seed,
        "feature_count": len(MODEL_FEATURES),
        "features": list(MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "training": {
            "rows": len(train_frame),
            "positive_rate": float(train_labels.mean()),
            "internal_validation_rows": len(internal_validation_frame),
            "internal_validation_positive_rate": float(internal_validation_labels.mean()),
            "split_time_ms": cutoff,
            "best_iteration": model.best_iteration,
            "seconds": training_seconds,
        },
        "standard_validation": evaluation,
        "top_features_by_gain": [
            {"feature": name, "gain": float(gain)} for name, gain in top_features
        ],
        "artifact": str(model_path),
    }


def evaluate_saved_ranker(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the observationally trained ranker on held-out randomized traffic."""

    lgb = _lightgbm()
    paths = config["paths"]
    feature_path = Path(paths["random_features"])
    model_path = Path(paths["biased_model"])
    for path in (feature_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    arrays = load_item_feature_arrays(Path(paths["item_feature_dir"]))
    model = lgb.Booster(model_file=str(model_path))
    batch_rows = int(config["evaluation"]["batch_rows"])
    return {
        "model": "standard_training",
        "random_validation": _predict_split(
            model, feature_path, arrays, batch_rows, "validation"
        ),
        "random_test": _predict_split(model, feature_path, arrays, batch_rows, "test"),
    }
