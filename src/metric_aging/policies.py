"""Replacement policies and baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from .model import LearningParams, cycle_value, never_replace_value, optimal_replacement_age, peak_age


@dataclass(frozen=True)
class BaselineComparison:
    """Comparison of optimal cutoff policy against baseline policies."""

    optimal_age: int
    peak_age: int
    optimal_value: float
    peak_value: float
    never_value: float
    peak_regret_percent: float
    never_regret_percent: float


def regret_percent(optimal_value: float, baseline_value: float) -> float:
    """Compute percentage regret relative to an optimal value."""

    if optimal_value <= 0:
        raise ValueError("optimal_value must be positive for regret computation.")
    return 100.0 * (optimal_value - baseline_value) / optimal_value


def compare_baselines(
    params: LearningParams,
    *,
    k_max: int = 1000,
    never_horizon: int = 5000,
) -> BaselineComparison:
    """Evaluate optimal, myopic-peak, and no-replacement policies."""

    optimal = optimal_replacement_age(params, k_max=k_max)
    peak = peak_age(params, k_max=k_max)
    peak_value = cycle_value(peak, params)
    no_replace_value = never_replace_value(params, horizon=never_horizon)
    return BaselineComparison(
        optimal_age=optimal.k_star,
        peak_age=peak,
        optimal_value=optimal.value,
        peak_value=peak_value,
        never_value=no_replace_value,
        peak_regret_percent=regret_percent(optimal.value, peak_value),
        never_regret_percent=regret_percent(optimal.value, no_replace_value),
    )
