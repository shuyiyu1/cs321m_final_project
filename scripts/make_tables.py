#!/usr/bin/env python
"""Regenerate only the tables used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from metric_aging.experiments import make_regime_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate paper tables.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"), help="Directory for table files.")
    parser.add_argument("--seed", type=int, default=231, help="Random seed; retained for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = make_regime_table()
    csv_path = args.output_dir / "regime_comparison.csv"
    tex_path = args.output_dir / "regime_comparison.tex"
    table.to_csv(csv_path, index=False)
    table.to_latex(tex_path, index=False, float_format="%.3f")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
