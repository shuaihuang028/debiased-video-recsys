from __future__ import annotations

import pytest

from short_video_recsys.debias.randomized_adaptation import randomized_example_weight


def test_randomized_weight_produces_requested_total_mass() -> None:
    standard_rows = 1_000
    randomized_rows = 100
    mass = 0.25

    randomized_weight = randomized_example_weight(standard_rows, randomized_rows, mass)
    realized_mass = randomized_rows * randomized_weight / (
        standard_rows + randomized_rows * randomized_weight
    )

    assert realized_mass == pytest.approx(mass)
