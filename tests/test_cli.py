from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from short_video_recsys.cli import build_features, split_data, validate_data


def test_cli_import_keeps_torch_and_faiss_out_of_the_same_process() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import short_video_recsys.cli; "
                "print('torch' in sys.modules, 'faiss' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "False False"


def _rows(dates: list[int], is_rand: int) -> pd.DataFrame:
    size = len(dates)
    return pd.DataFrame(
        {
            "user_id": [1] * size,
            "video_id": list(range(100, 100 + size)),
            "date": dates,
            "time_ms": [index * 1_000 for index in range(1, size + 1)],
            "is_click": [index % 2 for index in range(size)],
            "is_like": [0] * size,
            "long_view": [index % 2 for index in range(size)],
            "play_time_ms": [1_000] * size,
            "duration_ms": [2_000] * size,
            "is_rand": [is_rand] * size,
        }
    )


def _config(tmp_path: Path) -> dict:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _rows([20220421], is_rand=0).to_csv(raw_dir / "standard_early.csv", index=False)
    _rows([20220422, 20220423], is_rand=0).assign(time_ms=[2_000, 3_000]).to_csv(
        raw_dir / "standard_late.csv", index=False
    )
    _rows([20220421, 20220422, 20220423], is_rand=1).to_csv(
        raw_dir / "random.csv", index=False
    )
    return {
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "processed"),
            "results_dir": str(tmp_path / "results"),
            "standard_logs": ["standard_early.csv", "standard_late.csv"],
            "random_log": "random.csv",
        },
        "standard_split": {
            "train_end_ms": 1_000,
            "validation_end_ms": 2_000,
            "test_end_ms": 3_000,
        },
        "random_split": {
            "calibration_end_ms": 1_000,
            "validation_end_ms": 2_000,
            "test_end_ms": 3_000,
        },
        "features": {
            "engine": "duckdb",
            "output_file": "features_standard.parquet",
            "temp_dir": str(tmp_path / "duckdb_tmp"),
            "watch_ratio_cap": 5.0,
            "smoothing": {"alpha": 1.0, "beta": 9.0},
            "windows_ms": {"1h": 3_600_000},
        },
    }


def test_data_commands_run_end_to_end(tmp_path: Path) -> None:
    config = _config(tmp_path)

    validate_data(config)
    split_data(config)
    build_features(config)

    output_dir = Path(config["paths"]["processed_dir"])
    expected = {
        "standard_train.parquet",
        "standard_validation.parquet",
        "standard_test.parquet",
        "random_calibration.parquet",
        "random_validation.parquet",
            "random_test.parquet",
            "features_standard.parquet",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})

    featured = pd.read_parquet(output_dir / "features_standard.parquet")
    train = featured[featured.split == "train"].reset_index(drop=True)
    validation = featured[featured.split == "validation"].reset_index(drop=True)
    assert train.loc[0, "user_hist_impression_count"] == 0
    assert validation.loc[0, "user_hist_impression_count"] == 1
