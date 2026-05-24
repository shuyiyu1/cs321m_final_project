#!/usr/bin/env python
"""End-to-end reproduction script for all paper figures and tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metric_aging.experiments import make_regime_table
from metric_aging.visualization import plot_aging_dynamics, plot_optimal_age_heatmap


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Reproduce all figures and tables for the metric-aging paper.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated outputs.")
    parser.add_argument("--seed", type=int, default=231, help="Random seed; retained for reproducibility.")
    return parser.parse_args()


def main() -> None:
    """Generate all artifacts used in the manuscript."""

    args = parse_args()
    np.random.seed(args.seed)

    figures_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    figure_outputs = {
        "aging_dynamics": plot_aging_dynamics(figures_dir),
        "optimal_age_heatmap": plot_optimal_age_heatmap(figures_dir),
    }

    table = make_regime_table()
    csv_path = tables_dir / "regime_comparison.csv"
    tex_path = tables_dir / "regime_comparison.tex"
    table.to_csv(csv_path, index=False)
    table.to_latex(tex_path, index=False, float_format="%.3f")

    summary = {
        "seed": args.seed,
        "figures": figure_outputs,
        "tables": {"regime_comparison_csv": str(csv_path), "regime_comparison_tex": str(tex_path)},
        "status": "success",
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Reproduction complete. Outputs written to: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
