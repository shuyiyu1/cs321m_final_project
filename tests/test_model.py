"""Tests for the metric-aging model."""

from __future__ import annotations

import numpy as np
import pytest

from metric_aging.model import (
    LearningParams,
    hacking_probability,
    is_single_peaked,
    optimal_replacement_age,
    payoff_sequence,
    productive_probability,
    productive_probability_by_convolution,
    public_signal_probability,
)


def test_probabilities_are_valid() -> None:
    """Latent-state probabilities should be nonnegative and sum to at most one."""

    params = LearningParams(p=0.08, q=0.20, delta=0.95, gamma=0.90)
    ages = np.arange(1, 100)
    p_k = productive_probability(ages, params.p, params.q)
    h_k = hacking_probability(ages, params.p, params.q)
    unlearned = np.power(1 - params.p, ages)
    assert np.all(p_k >= -1e-12)
    assert np.all(h_k >= -1e-12)
    assert np.allclose(p_k + h_k + unlearned, 1.0, atol=1e-10)


def test_closed_form_matches_convolution() -> None:
    """Closed form P_k should match the direct finite-sum expression."""

    for p, q in [(0.08, 0.08), (0.08, 0.20), (0.20, 0.04)]:
        for age in [1, 2, 5, 20, 50]:
            assert productive_probability(age, p, q) == pytest.approx(
                productive_probability_by_convolution(age, p, q), abs=1e-12
            )


def test_public_signal_can_rise_while_true_value_falls() -> None:
    """In the balanced regime, public signal increases after true value peaks."""

    params = LearningParams(p=0.08, q=0.08, delta=0.95, gamma=0.90)
    ages = np.arange(1, 60)
    true_value = payoff_sequence(params, len(ages))
    public_signal = public_signal_probability(ages, params)
    peak = int(np.argmax(true_value))
    assert is_single_peaked(true_value)
    assert true_value[-1] < true_value[peak]
    assert public_signal[-1] > public_signal[peak]


def test_balanced_regime_optimal_age_matches_paper() -> None:
    """The main figure's balanced regime should have K*=22."""

    params = LearningParams(p=0.08, q=0.08, delta=0.95, gamma=0.90)
    result = optimal_replacement_age(params, k_max=1000)
    assert result.k_star == 22


def test_invalid_parameters_raise_clear_errors() -> None:
    """Parameter validation should catch invalid probabilities."""

    with pytest.raises(ValueError, match="p must be"):
        LearningParams(p=0.0, q=0.1, delta=0.95, gamma=0.9)
    with pytest.raises(ValueError, match="delta must be"):
        LearningParams(p=0.1, q=0.1, delta=1.0, gamma=0.9)
