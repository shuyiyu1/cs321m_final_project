"""Experiment definitions for reproducing the paper's figures and tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .model import LearningParams, hacking_probability, optimal_replacement_age, productive_probability, public_signal_probability
from .policies import compare_baselines


@dataclass(frozen=True)
class Regime:
    """Named parameter regime used in the manuscript table."""

    name: str
    p: float
    q: float


def default_regimes() -> list[Regime]:
    """Return the four parameter regimes reported in the manuscript."""

    return [
        Regime("Slow hacking", p=0.08, q=0.02),
        Regime("Balanced", p=0.08, q=0.08),
        Regime("Fast hacking", p=0.08, q=0.20),
        Regime("Fast learning", p=0.20, q=0.04),
    ]


def make_regime_table(
    regimes: Iterable[Regime] | None = None,
    *,
    delta: float = 0.95,
    gamma: float = 0.90,
    k_max: int = 1000,
    never_horizon: int = 5000,
) -> pd.DataFrame:
    """Compute the regime-comparison table from the paper."""

    rows: list[dict[str, float | int | str]] = []
    for regime in regimes or default_regimes():
        params = LearningParams(p=regime.p, q=regime.q, delta=delta, gamma=gamma)
        comparison = compare_baselines(params, k_max=k_max, never_horizon=never_horizon)
        rows.append(
            {
                "Regime": regime.name,
                "p": regime.p,
                "q": regime.q,
                "K_star": comparison.optimal_age,
                "Peak_age": comparison.peak_age,
                "J_K_star": comparison.optimal_value,
                "Peak_regret_percent": comparison.peak_regret_percent,
                "Never_regret_percent": comparison.never_regret_percent,
            }
        )
    return pd.DataFrame(rows)


def make_aging_curve(
    params: LearningParams,
    *,
    max_age: int = 60,
    k_max_for_optimum: int = 400,
) -> pd.DataFrame:
    """Return the data behind the metric-aging dynamics figure."""

    if max_age <= 0:
        raise ValueError("max_age must be positive.")
    ages = np.arange(1, max_age + 1)
    result = optimal_replacement_age(params, k_max=k_max_for_optimum)
    return pd.DataFrame(
        {
            "age": ages,
            "productive_value": productive_probability(ages, params.p, params.q),
            "hacking_probability": hacking_probability(ages, params.p, params.q),
            "public_signal": public_signal_probability(ages, params),
            "optimal_age": result.k_star,
        }
    )


def make_heatmap_grid(
    *,
    p_values: np.ndarray | None = None,
    q_values: np.ndarray | None = None,
    delta: float = 0.95,
    gamma: float = 0.90,
    k_max: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute K* over a grid of productive and hacking learning rates."""

    if p_values is None:
        p_values = np.linspace(0.02, 0.30, 24)
    if q_values is None:
        q_values = np.linspace(0.01, 0.30, 24)
    k_star = np.zeros((len(q_values), len(p_values)), dtype=int)
    for i, q in enumerate(q_values):
        for j, p in enumerate(p_values):
            params = LearningParams(p=float(p), q=float(q), delta=delta, gamma=gamma)
            k_star[i, j] = optimal_replacement_age(params, k_max=k_max).k_star
    return p_values, q_values, k_star
