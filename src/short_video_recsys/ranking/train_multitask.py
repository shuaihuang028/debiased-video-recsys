"""Train and compare single-task, shared-bottom, and MMoE neural rankers."""

from __future__ import annotations

import copy
import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from short_video_recsys.ranking.metrics import evaluate_scores
from short_video_recsys.ranking.multitask_models import (
    FeatureEncoder,
    MMoE,
    SharedBottom,
    SingleTaskMLP,
    parameter_count,
)
from short_video_recsys.ranking.train_ranker import (
    CATEGORICAL_FEATURES,
    DAY_MS,
    MODEL_FEATURES,
    RAW_HISTORY_FEATURES,
    _escaped,
    _feature_frame,
)
from short_video_recsys.training.item_features import load_item_feature_arrays

TASK_NAMES = ("is_click", "long_view", "is_like")
PRIMARY_TASK = "long_view"
NUMERIC_FEATURES = tuple(name for name in MODEL_FEATURES if name not in CATEGORICAL_FEATURES)
MULTITASK_SOURCE_COLUMNS = (
    "user_id",
    "video_id",
    "time_ms",
    "tab",
    *TASK_NAMES,
    *RAW_HISTORY_FEATURES,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample(
    connection: duckdb.DuckDBPyConnection,
    feature_path: Path,
    predicate: str,
    rows: int,
    seed: int,
) -> pd.DataFrame:
    columns = ", ".join(MULTITASK_SOURCE_COLUMNS)
    return connection.execute(
        f"""
        SELECT {columns}
        FROM (
            SELECT {columns}
            FROM read_parquet('{_escaped(feature_path)}')
            WHERE {predicate}
        )
        USING SAMPLE reservoir({rows} ROWS) REPEATABLE({seed})
        """
    ).fetch_df()


def _normalization(matrix: pd.DataFrame) -> dict[str, list[float]]:
    numeric = matrix.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    means = np.mean(numeric, axis=0, dtype=np.float64).astype(np.float32)
    standard_deviations = np.std(numeric, axis=0, dtype=np.float64).astype(np.float32)
    standard_deviations = np.where(standard_deviations < 1e-6, 1.0, standard_deviations)
    return {
        "mean": means.tolist(),
        "standard_deviation": standard_deviations.tolist(),
    }


def _tensorize_features(
    matrix: pd.DataFrame,
    normalization: Mapping[str, Sequence[float]],
    cardinalities: Sequence[int],
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    categorical_values = matrix.loc[:, CATEGORICAL_FEATURES].to_numpy(dtype=np.int64)
    for index, (name, cardinality) in enumerate(zip(CATEGORICAL_FEATURES, cardinalities)):
        values = categorical_values[:, index]
        if len(values) and (values.min() < 0 or values.max() >= cardinality):
            raise ValueError(f"Feature {name} is outside configured cardinality {cardinality}")
    numeric_values = matrix.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    means = np.asarray(normalization["mean"], dtype=np.float32)
    standard_deviations = np.asarray(normalization["standard_deviation"], dtype=np.float32)
    numeric_values = np.clip((numeric_values - means) / standard_deviations, -clip, clip)
    return (
        torch.from_numpy(np.ascontiguousarray(categorical_values)),
        torch.from_numpy(np.ascontiguousarray(numeric_values)),
    )


def _labels(frame: pd.DataFrame) -> torch.Tensor:
    values = frame.loc[:, TASK_NAMES].to_numpy(dtype=np.float32)
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Multi-task labels must be binary")
    return torch.from_numpy(np.ascontiguousarray(values))


def _groups(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    users = frame["user_id"].to_numpy(dtype=np.int32)
    days = (
        (frame["time_ms"].to_numpy(dtype=np.int64) + 8 * 60 * 60 * 1_000) // DAY_MS
    ).astype(np.int32)
    return users, days


def _encoder(config: Mapping[str, Any]) -> FeatureEncoder:
    cardinality_config = config["categorical_cardinalities"]
    embedding_config = config["embedding_dimensions"]
    cardinalities = [int(cardinality_config[name]) for name in CATEGORICAL_FEATURES]
    dimensions = [int(embedding_config[name]) for name in CATEGORICAL_FEATURES]
    return FeatureEncoder(cardinalities, dimensions, len(NUMERIC_FEATURES))


def _build_models(config: Mapping[str, Any]) -> dict[str, nn.Module]:
    architecture = config["architecture"]
    dropout = float(architecture["dropout"])
    return {
        "single_task_mlp": SingleTaskMLP(
            _encoder(config), architecture["single_task_hidden"], dropout
        ),
        "shared_bottom": SharedBottom(
            _encoder(config),
            TASK_NAMES,
            architecture["shared_bottom_hidden"],
            int(architecture["task_tower_dim"]),
            dropout,
        ),
        "mmoe": MMoE(
            _encoder(config),
            TASK_NAMES,
            int(architecture["num_experts"]),
            architecture["expert_hidden"],
            int(architecture["task_tower_dim"]),
            dropout,
        ),
    }


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _predict_tensor(
    model: nn.Module,
    categorical: torch.Tensor,
    numeric: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(categorical), batch_size):
            stop = start + batch_size
            logits = model(categorical[start:stop].to(device), numeric[start:stop].to(device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _task_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    model_tasks: Sequence[str],
    users: np.ndarray,
    days: np.ndarray,
) -> dict[str, dict[str, int | float]]:
    metrics = {}
    for prediction_index, task in enumerate(model_tasks):
        label_index = TASK_NAMES.index(task)
        metrics[task] = evaluate_scores(
            labels[:, label_index], predictions[:, prediction_index], users, days
        ).to_dict()
    return metrics


def _gate_diagnostics(
    model: MMoE,
    categorical: torch.Tensor,
    numeric: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    weight_sums = {
        task: np.zeros(model.num_experts, dtype=np.float64) for task in model.task_names
    }
    entropy_sums = {task: 0.0 for task in model.task_names}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(categorical), batch_size):
            stop = start + batch_size
            weights = model.gate_weights(
                categorical[start:stop].to(device), numeric[start:stop].to(device)
            )
            for task, task_weights in weights.items():
                values = task_weights.cpu().numpy().astype(np.float64)
                weight_sums[task] += values.sum(axis=0)
                entropy_sums[task] += float(
                    np.sum(-values * np.log(np.clip(values, 1e-12, 1.0)))
                )
    output = {}
    for task in model.task_names:
        mean_weights = weight_sums[task] / len(categorical)
        output[task] = {
            "mean_expert_weights": mean_weights.tolist(),
            "dominant_expert": int(np.argmax(mean_weights)),
            "mean_normalized_entropy": float(
                entropy_sums[task] / len(categorical) / np.log(model.num_experts)
            ),
        }
    return output


def _train_one(
    name: str,
    model: nn.Module,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation_groups: tuple[np.ndarray, np.ndarray],
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    _set_seed(seed)
    training = config["training"]
    batch_size = int(training["batch_size"])
    task_names = tuple(model.task_names)
    task_indices = [TASK_NAMES.index(task) for task in task_names]
    train_categorical, train_numeric, train_labels = train_tensors
    validation_categorical, validation_numeric, validation_labels = validation_tensors
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_categorical, train_numeric, train_labels[:, task_indices]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    positive_weights = torch.tensor(
        [float(training["positive_weights"][task]) for task in task_names],
        dtype=torch.float32,
        device=device,
    )
    loss_weights = torch.tensor(
        [float(training["task_loss_weights"][task]) for task in task_names],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights, reduction="none")
    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()
    users, days = validation_groups

    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for categorical_batch, numeric_batch, label_batch in loader:
            categorical_batch = categorical_batch.to(device)
            numeric_batch = numeric_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(categorical_batch, numeric_batch)
            per_task_loss = criterion(logits, label_batch).mean(dim=0)
            loss = torch.sum(per_task_loss * loss_weights) / torch.sum(loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(label_batch)
            row_count += len(label_batch)

        predictions = _predict_tensor(
            model,
            validation_categorical,
            validation_numeric,
            int(training["prediction_batch_size"]),
            device,
        )
        metrics = _task_metrics(
            validation_labels.numpy(), predictions, task_names, users, days
        )
        primary_auc = float(metrics[PRIMARY_TASK]["roc_auc"])
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / row_count,
                "internal_validation": metrics,
            }
        )
        print(
            f"{name}: epoch={epoch} loss={loss_sum / row_count:.6f} "
            f"long_view_auc={primary_auc:.6f}",
            flush=True,
        )
        if primary_auc > best_auc + float(training["minimum_auc_improvement"]):
            best_auc = primary_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(training["early_stopping_patience"]):
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    final_predictions = _predict_tensor(
        model,
        validation_categorical,
        validation_numeric,
        int(training["prediction_batch_size"]),
        device,
    )
    final_metrics = _task_metrics(
        validation_labels.numpy(), final_predictions, task_names, users, days
    )
    result = {
        "tasks": list(task_names),
        "parameter_count": parameter_count(model),
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "internal_validation": final_metrics,
        "history": history,
    }
    if isinstance(model, MMoE):
        result["gate_diagnostics"] = _gate_diagnostics(
            model,
            validation_categorical,
            validation_numeric,
            int(training["prediction_batch_size"]),
            device,
        )
    return model, result


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    normalization: Mapping[str, Sequence[float]],
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "tasks": list(model.task_names),
            "model_features": list(MODEL_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "numeric_features": list(NUMERIC_FEATURES),
            "normalization": dict(normalization),
            "model_config": dict(config["model"]),
        },
        path,
    )


def _evaluate_models(
    models: Mapping[str, nn.Module],
    feature_path: Path,
    arrays: Mapping[str, np.ndarray],
    normalization: Mapping[str, Sequence[float]],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    columns = ", ".join(MULTITASK_SOURCE_COLUMNS)
    connection = duckdb.connect()
    reader = connection.execute(
        f"""
        SELECT {columns}
        FROM read_parquet('{_escaped(feature_path)}')
        WHERE split = 'validation'
        """
    ).fetch_record_batch(rows_per_batch=int(config["evaluation"]["batch_rows"]))
    labels_parts: list[np.ndarray] = []
    users_parts: list[np.ndarray] = []
    days_parts: list[np.ndarray] = []
    baseline_parts = {task: [] for task in TASK_NAMES}
    prediction_parts: dict[str, list[np.ndarray]] = {name: [] for name in models}
    prediction_seconds = {name: 0.0 for name in models}
    cardinalities = [
        int(config["model"]["categorical_cardinalities"][name])
        for name in CATEGORICAL_FEATURES
    ]
    rate_columns = {
        "is_click": "item_hist_ctr",
        "long_view": "item_hist_long_view_rate",
        "is_like": "item_hist_like_rate",
    }
    for batch in reader:
        frame = batch.to_pandas()
        matrix = _feature_frame(frame, dict(arrays))
        categorical, numeric = _tensorize_features(
            matrix,
            normalization,
            cardinalities,
            float(config["training"]["numeric_clip"]),
        )
        labels_parts.append(frame.loc[:, TASK_NAMES].to_numpy(dtype=np.int8))
        users, days = _groups(frame)
        users_parts.append(users)
        days_parts.append(days)
        for task, column in rate_columns.items():
            baseline_parts[task].append(frame[column].to_numpy(dtype=np.float32))
        for name, model in models.items():
            started = time.perf_counter()
            prediction_parts[name].append(
                _predict_tensor(
                    model,
                    categorical,
                    numeric,
                    int(config["training"]["prediction_batch_size"]),
                    device,
                )
            )
            prediction_seconds[name] += time.perf_counter() - started
    connection.close()

    labels = np.concatenate(labels_parts)
    users = np.concatenate(users_parts)
    days = np.concatenate(days_parts)
    model_results = {}
    for name, model in models.items():
        predictions = np.concatenate(prediction_parts[name])
        model_results[name] = {
            "tasks": _task_metrics(labels, predictions, model.task_names, users, days),
            "prediction_seconds": prediction_seconds[name],
            "rows_per_second": len(labels) / prediction_seconds[name],
        }
    baselines = {
        task: evaluate_scores(
            labels[:, TASK_NAMES.index(task)],
            np.concatenate(baseline_parts[task]),
            users,
            days,
        ).to_dict()
        for task in TASK_NAMES
    }
    return model_results, baselines


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "standard_deviation": float(array.std(ddof=0))}


def _aggregate_runs(runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    first_run = next(iter(runs.values()))
    model_names = tuple(first_run["standard_validation"])
    aggregate = {}
    metric_names = ("roc_auc", "user_day_ndcg_at_10", "log_loss", "brier_score")
    for model_name in model_names:
        tasks = tuple(first_run["standard_validation"][model_name]["tasks"])
        aggregate[model_name] = {
            "parameter_count": first_run["training"][model_name]["parameter_count"],
            "best_epoch": _summary(
                [run["training"][model_name]["best_epoch"] for run in runs.values()]
            ),
            "training_seconds": _summary(
                [run["training"][model_name]["training_seconds"] for run in runs.values()]
            ),
            "standard_validation": {
                task: {
                    metric: _summary(
                        [
                            run["standard_validation"][model_name]["tasks"][task][metric]
                            for run in runs.values()
                        ]
                    )
                    for metric in metric_names
                }
                for task in tasks
            },
        }
    return aggregate


def train_multitask_comparison(config: dict[str, Any]) -> dict[str, Any]:
    """Run the controlled neural ranker comparison and return JSON-ready results."""

    seed = int(config["seed"])
    _set_seed(seed)
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    feature_path = Path(config["paths"]["features"])
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    device = _resolve_device(str(config["training"]["device"]))
    holdout_fraction = float(config["training"]["internal_validation_fraction"])
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
        int(config["training"]["max_train_rows"]),
        seed,
    )
    validation_frame = _sample(
        connection,
        feature_path,
        f"split = 'train' AND time_ms > {cutoff}",
        int(config["training"]["max_internal_validation_rows"]),
        seed + 1,
    )
    connection.close()
    arrays = load_item_feature_arrays(Path(config["paths"]["item_feature_dir"]))
    train_matrix = _feature_frame(train_frame, arrays)
    validation_matrix = _feature_frame(validation_frame, arrays)
    normalization = _normalization(train_matrix)
    cardinalities = [
        int(config["model"]["categorical_cardinalities"][name])
        for name in CATEGORICAL_FEATURES
    ]
    train_features = _tensorize_features(
        train_matrix,
        normalization,
        cardinalities,
        float(config["training"]["numeric_clip"]),
    )
    validation_features = _tensorize_features(
        validation_matrix,
        normalization,
        cardinalities,
        float(config["training"]["numeric_clip"]),
    )
    train_tensors = (*train_features, _labels(train_frame))
    validation_tensors = (*validation_features, _labels(validation_frame))
    validation_groups = _groups(validation_frame)

    experiment_seeds = tuple(int(value) for value in config.get("experiment_seeds", [seed]))
    if not experiment_seeds:
        raise ValueError("experiment_seeds cannot be empty")
    runs = {}
    baselines = None
    for run_seed in experiment_seeds:
        _set_seed(run_seed)
        models = _build_models(config["model"])
        training_results = {}
        trained_models = {}
        for model_index, (name, model) in enumerate(models.items()):
            trained, result = _train_one(
                name,
                model,
                train_tensors,
                validation_tensors,
                validation_groups,
                config,
                run_seed + model_index,
                device,
            )
            checkpoint = (
                Path(config["paths"]["artifact_dir"]) / f"{name}_seed{run_seed}.pt"
            )
            _save_checkpoint(trained, checkpoint, normalization, config)
            result["artifact"] = str(checkpoint)
            training_results[name] = result
            trained_models[name] = trained

        standard_validation, run_baselines = _evaluate_models(
            trained_models,
            feature_path,
            arrays,
            normalization,
            config,
            device,
        )
        if baselines is None:
            baselines = run_baselines
        primary_scores = {
            name: float(result["tasks"][PRIMARY_TASK]["user_day_ndcg_at_10"])
            for name, result in standard_validation.items()
        }
        runs[str(run_seed)] = {
            "training": training_results,
            "standard_validation": standard_validation,
            "comparison": {
                "winner": max(primary_scores, key=primary_scores.get),
                "long_view_ndcg_at_10": primary_scores,
                "absolute_change_vs_single_task": {
                    name: score - primary_scores["single_task_mlp"]
                    for name, score in primary_scores.items()
                },
            },
        }

    aggregate = _aggregate_runs(runs)
    primary_means = {
        name: result["standard_validation"][PRIMARY_TASK]["user_day_ndcg_at_10"]["mean"]
        for name, result in aggregate.items()
    }
    deltas_by_model = {
        name: _summary(
            [
                run["comparison"]["absolute_change_vs_single_task"][name]
                for run in runs.values()
            ]
        )
        for name in primary_means
    }
    return {
        "experiment": "controlled_neural_multitask_comparison",
        "data_sample_seed": seed,
        "experiment_seeds": list(experiment_seeds),
        "tasks": list(TASK_NAMES),
        "primary_task": PRIMARY_TASK,
        "selection_metric": "internal_validation_long_view_roc_auc",
        "final_metric": "standard_validation_long_view_user_day_ndcg_at_10",
        "device": str(device),
        "feature_contract": {
            "feature_count": len(MODEL_FEATURES),
            "categorical_feature_count": len(CATEGORICAL_FEATURES),
            "numeric_feature_count": len(NUMERIC_FEATURES),
            "features": list(MODEL_FEATURES),
        },
        "data": {
            "training_rows": len(train_frame),
            "internal_validation_rows": len(validation_frame),
            "split_time_ms": cutoff,
            "training_positive_rates": {
                task: float(train_frame[task].mean()) for task in TASK_NAMES
            },
            "internal_validation_positive_rates": {
                task: float(validation_frame[task].mean()) for task in TASK_NAMES
            },
        },
        "runs": runs,
        "aggregate": aggregate,
        "historical_item_rate_baselines": baselines,
        "comparison": {
            "winner_by_mean_long_view_ndcg_at_10": max(
                primary_means, key=primary_means.get
            ),
            "mean_long_view_ndcg_at_10": primary_means,
            "change_vs_single_task": deltas_by_model,
        },
        "config": json.loads(json.dumps(config)),
    }
