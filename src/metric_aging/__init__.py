"""Metric aging and reward-hacking reproduction package."""

from .model import (
    LearningParams,
    ReplacementResult,
    cycle_value,
    hacking_probability,
    is_single_peaked,
    optimal_replacement_age,
    payoff_sequence,
    productive_probability,
    public_signal_probability,
)

__all__ = [
    "LearningParams",
    "ReplacementResult",
    "cycle_value",
    "hacking_probability",
    "is_single_peaked",
    "optimal_replacement_age",
    "payoff_sequence",
    "productive_probability",
    "public_signal_probability",
]
