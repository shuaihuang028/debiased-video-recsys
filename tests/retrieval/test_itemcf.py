from __future__ import annotations

from pathlib import Path

import pandas as pd

from short_video_recsys.retrieval.itemcf import fit_sequential_itemcf, recommend_itemcf


def test_sequential_itemcf_learns_forward_neighbors(tmp_path: Path) -> None:
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "video_id": [10, 11, 12, 10, 11],
            "time_ms": [1, 2, 3, 1, 2],
            "long_view": [1, 1, 1, 1, 1],
        }
    )
    input_path = tmp_path / "fit.parquet"
    neighbor_path = tmp_path / "neighbors.parquet"
    interactions.to_parquet(input_path, index=False)

    summary = fit_sequential_itemcf(
        [input_path],
        neighbor_path,
        context_window=2,
        neighbors_per_item=10,
        temp_directory=tmp_path / "duckdb_tmp",
    )
    recommendations = recommend_itemcf(neighbor_path, {0: [10], 1: [11, 10]}, k=2)

    assert summary.neighbors >= 3
    assert recommendations[0][0] == 11
    assert recommendations[1] == [12]
