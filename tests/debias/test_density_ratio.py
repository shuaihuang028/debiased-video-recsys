from __future__ import annotations

import numpy as np
import pytest

from short_video_recsys.debias.density_ratio import density_ratio_weights


def test_density_ratio_weights_are_clipped_and_self_normalized() -> None:
    weights, diagnostics = density_ratio_weights(
        np.array([0.01, 0.5, 0.99]), lower=0.1, upper=10.0
    )

    assert weights.mean() == pytest.approx(1.0)
    assert diagnostics["fraction_clipped_low"] == pytest.approx(1 / 3)
    assert diagnostics["fraction_clipped_high"] == pytest.approx(1 / 3)
    assert diagnostics["effective_sample_fraction"] < 1.0
