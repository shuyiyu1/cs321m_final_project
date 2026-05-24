#!/usr/bin/env python
"""Regenerate only the figures used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from metric_aging.visualization import plot_aging_dynamics, plot_optimal_age_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate paper figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"), help="Directory for figure files.")
    parser.add_argument("--seed", type=int, default=231, help="Random seed; retained for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(plot_aging_dynamics(args.output_dir))
    print(plot_optimal_age_heatmap(args.output_dir))


if __name__ == "__main__":
    main()
