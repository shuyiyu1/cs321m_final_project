"""Core model equations for metric aging and reward hacking.

The functions in this module implement the closed-form expressions from the
paper. They are deterministic and have no external data dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LearningParams:
    """Parameters of the metric-aging model.

    Args:
        p: Per-period probability of learning the productive action before it is learned.
        q: Per-period probability of learning the hacking action after productivity is learned.
        delta: Discount factor in (0, 1).
        gamma: Probability that productive effort or hacking creates the public success signal.
    """

    p: float
    q: float
    delta: float = 0.95
    gamma: float = 0.90

    def __post_init__(self) -> None:
        _check_probability("p", self.p, allow_zero=False)
        _check_probability("q", self.q, allow_zero=False)
        _check_probability("delta", self.delta, allow_zero=False, allow_one=False)
        _check_probability("gamma", self.gamma, allow_zero=False)


@dataclass(frozen=True)
class ReplacementResult:
    """Output of exhaustive cutoff-policy enumeration."""

    k_star: int
    value: float
    values: np.ndarray


def _check_probability(
    name: str,
    value: float,
    *,
    allow_zero: bool = True,
    allow_one: bool = True,
) -> None:
    """Validate probability-like inputs and raise a helpful error if invalid."""

    lower_ok = value >= 0 if allow_zero else value > 0
    upper_ok = value <= 1 if allow_one else value < 1
    if not (lower_ok and upper_ok):
        lower = "0" if allow_zero else "0 (exclusive)"
        upper = "1" if allow_one else "1 (exclusive)"
        raise ValueError(f"{name} must be between {lower} and {upper}; got {value!r}.")


def _as_age_array(k: int | Iterable[int] | np.ndarray) -> np.ndarray:
    """Convert scalar or vector ages to a positive integer NumPy array."""

    ages = np.asarray(k, dtype=int)
    if np.any(ages < 0):
        raise ValueError("metric age k must be nonnegative.")
    return ages


def productive_probability(k: int | Iterable[int] | np.ndarray, p: float, q: float) -> np.ndarray | float:
    """Probability that the agent is productive but not hacked after age ``k``.

    This is the paper's closed-form expression for P_k. The return type mirrors
    the input: a scalar input returns a float, and a vector input returns an array.
    """

    _check_probability("p", p, allow_zero=False)
    _check_probability("q", q, allow_zero=False)
    ages = _as_age_array(k)
    scalar_input = ages.ndim == 0
    ages = np.atleast_1d(ages)

    output = np.zeros_like(ages, dtype=float)
    positive = ages > 0
    kp = ages[positive].astype(float)

    if abs(p - q) < 1e-12:
        output[positive] = kp * p * np.power(1 - p, kp - 1)
    else:
        output[positive] = p * (np.power(1 - q, kp) - np.power(1 - p, kp)) / (p - q)

    if scalar_input:
        return float(output[0])
    return output


def productive_probability_by_convolution(k: int, p: float, q: float) -> float:
    """Direct finite-sum expression for P_k, useful for validation tests."""

    if k < 0:
        raise ValueError("metric age k must be nonnegative.")
    if k == 0:
        return 0.0
    terms = [(1 - p) ** (m - 1) * p * (1 - q) ** (k - m) for m in range(1, k + 1)]
    return float(sum(terms))


def hacking_probability(k: int | Iterable[int] | np.ndarray, p: float, q: float) -> np.ndarray | float:
    """Probability that the hacking action has been learned after age ``k``."""

    ages = _as_age_array(k)
    scalar_input = ages.ndim == 0
    p_k = productive_probability(ages, p, q)
    h_k = 1.0 - np.power(1 - p, ages) - p_k
    h_k = np.clip(h_k, 0.0, 1.0)  # Protect against tiny floating-point roundoff.
    if scalar_input:
        return float(np.asarray(h_k)[()])
    return h_k


def public_signal_probability(
    k: int | Iterable[int] | np.ndarray,
    params: LearningParams,
) -> np.ndarray | float:
    """Probability that the public metric reports success at age ``k``."""

    ages = _as_age_array(k)
    signal = params.gamma * (1.0 - np.power(1 - params.p, ages))
    if ages.ndim == 0:
        return float(np.asarray(signal)[()])
    return signal


def payoff_sequence(params: LearningParams, k_max: int) -> np.ndarray:
    """Return true one-period values P_1, ..., P_kmax."""

    if k_max <= 0:
        raise ValueError(f"k_max must be positive; got {k_max}.")
    ages = np.arange(1, k_max + 1)
    return np.asarray(productive_probability(ages, params.p, params.q), dtype=float)


def cycle_value(k_cycle: int, params: LearningParams) -> float:
    """Normalized discounted value of replacing a metric every ``k_cycle`` periods."""

    if k_cycle <= 0:
        raise ValueError(f"k_cycle must be positive; got {k_cycle}.")
    payoffs = payoff_sequence(params, k_cycle)
    discounts = np.power(params.delta, np.arange(k_cycle))
    numerator = (1.0 - params.delta) * float(np.dot(discounts, payoffs))
    denominator = 1.0 - params.delta**k_cycle
    return numerator / denominator


def optimal_replacement_age(params: LearningParams, k_max: int = 1000) -> ReplacementResult:
    """Enumerate cutoff policies and return the optimal replacement age.

    Args:
        params: Learning parameters for the model.
        k_max: Largest replacement cycle length to consider.

    Returns:
        ReplacementResult with 1-indexed optimal age, optimal value, and the full
        vector of values for cycle lengths 1 through k_max.
    """

    if k_max <= 0:
        raise ValueError(f"k_max must be positive; got {k_max}.")
    values = np.array([cycle_value(k, params) for k in range(1, k_max + 1)], dtype=float)
    best_idx = int(np.argmax(values))
    return ReplacementResult(k_star=best_idx + 1, value=float(values[best_idx]), values=values)


def peak_age(params: LearningParams, k_max: int = 1000) -> int:
    """Return the age that maximizes one-period true value P_k."""

    payoffs = payoff_sequence(params, k_max)
    return int(np.argmax(payoffs) + 1)


def never_replace_value(params: LearningParams, horizon: int = 5000) -> float:
    """Approximate the value of never replacing the metric over a long horizon."""

    if horizon <= 0:
        raise ValueError(f"horizon must be positive; got {horizon}.")
    payoffs = payoff_sequence(params, horizon)
    discounts = np.power(params.delta, np.arange(horizon))
    return float((1.0 - params.delta) * np.dot(discounts, payoffs))


def is_single_peaked(values: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check whether a numerical sequence is weakly single-peaked."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    peak = int(np.argmax(values))
    left_diffs = np.diff(values[: peak + 1])
    right_diffs = np.diff(values[peak:])
    return bool(np.all(left_diffs >= -tolerance) and np.all(right_diffs <= tolerance))
