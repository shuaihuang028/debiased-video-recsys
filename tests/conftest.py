from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 1],
            "video_id": [11, 10, 10, 10],
            "date": [20220422, 20220421, 20220423, 20220424],
            "time_ms": [2_000, 1_000, 3_000, 4_000],
            "is_click": [0, 1, 1, 0],
            "is_like": [0, 1, 0, 0],
            "long_view": [0, 1, 1, 0],
            "play_time_ms": [1_000, 10_000, 8_000, 2_000],
            "duration_ms": [10_000, 10_000, 10_000, 10_000],
            "is_rand": [0, 0, 0, 0],
        }
    )
