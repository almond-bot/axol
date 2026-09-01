from __future__ import annotations

import numpy as np
import pytest

from almond_axol.teleop.filter import (
    AlphaSmoothFilter,
    OneEuroFilter,
    TrapezoidalFilter,
)


def test_alpha_filter_smooths_and_resets() -> None:
    filt = AlphaSmoothFilter(alpha=0.25)
    np.testing.assert_allclose(filt.update(np.array([0.0, 4.0])), [0.0, 4.0])
    np.testing.assert_allclose(filt.update(np.array([4.0, 0.0])), [1.0, 3.0])
    assert filt.update(None) is None
    filt.reset(np.array([2.0]))
    np.testing.assert_allclose(filt.update(np.array([6.0])), [3.0])


def test_trapezoidal_filter_respects_acceleration_and_velocity() -> None:
    filt = TrapezoidalFilter(max_vel=1.0, max_accel=2.0, dt=0.1)
    filt.update(np.array([0.0]))
    positions = [float(filt.update(np.array([10.0]))[0]) for _ in range(10)]

    deltas = np.diff([0.0, *positions])
    assert np.all(deltas <= 0.10001)
    assert deltas[0] == pytest.approx(0.02)
    assert positions == sorted(positions)


def test_one_euro_filter_is_stable_and_uses_real_timing() -> None:
    filt = OneEuroFilter(freq=100.0, min_cutoff=1.0, beta=2.0)
    first = filt.update(np.array([0.0]), t=1.0)
    second = filt.update(np.array([1.0]), t=1.01)
    third = filt.update(np.array([1.0]), t=1.03)

    np.testing.assert_allclose(first, [0.0])
    assert 0.0 < second[0] < 1.0
    assert second[0] < third[0] <= 1.0
    filt.reset()
    np.testing.assert_allclose(filt.update(np.array([3.0])), [3.0])
