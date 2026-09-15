from __future__ import annotations

import math

import pytest

from mostgen.numerics import (
    beer_lambert_transmittance,
    critical_wavelength,
    interval_score,
    lower_confidence_bound,
    trapezoid_auc,
    weighted_geometric_mean,
)


def test_trapezoid_auc_constant_curve():
    assert trapezoid_auc([290, 320, 400], [2, 2, 2], 290, 400) == pytest.approx(220.0)
    assert trapezoid_auc([290, 320, 400], [2, 2, 2], 290, 320) == pytest.approx(60.0)


def test_critical_wavelength_is_90_percent_cumulative_auc():
    assert critical_wavelength([290, 400], [1, 1]) == pytest.approx(389.0)


def test_beer_lambert_fixed_loading():
    assert beer_lambert_transmittance([1.0, 1.0]) == pytest.approx(0.1)


def test_uncertainty_lower_bound():
    assert lower_confidence_bound(10.0, 2.0, 1.5) == pytest.approx(7.0)


def test_dense_geometric_reward_has_nonzero_floor():
    reward = weighted_geometric_mean({"a": 0.0, "b": 0.8}, {"a": 1, "b": 1}, floor=0.001)
    assert 0.0 < reward < 0.8


def test_interval_transform_prefers_window_centre():
    assert interval_score(5, 4, 6, 0.2) > interval_score(2, 4, 6, 0.2)


def test_bad_axes_are_rejected():
    with pytest.raises(ValueError):
        trapezoid_auc([300, 290], [1, 1], 290, 300)

