"""Plotting utilities for paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiments import make_aging_curve, make_heatmap_grid
from .model import LearningParams


def _ensure_directory(path: Path) -> None:
    """Create an output directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def plot_aging_dynamics(output_dir: str | Path, *, params: LearningParams | None = None) -> dict[str, str]:
    """Create Figure 1: metric-aging dynamics."""

    output_path = Path(output_dir)
    _ensure_directory(output_path)
    params = params or LearningParams(p=0.08, q=0.08, delta=0.95, gamma=0.90)
    data = make_aging_curve(params, max_age=60, k_max_for_optimum=400)
    k_star = int(data["optimal_age"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(data["age"], data["productive_value"], linewidth=2, label="True productive value $P_k$")
    ax.plot(data["age"], data["hacking_probability"], linewidth=2, label="Hacking probability $H_k$")
    ax.plot(data["age"], data["public_signal"], linewidth=2, linestyle="--", label="Public success signal")
    ax.axvline(k_star, linestyle=":", linewidth=1.8, label=f"Optimal $K^*={k_star}$")
    ax.set_xlabel("Metric age $k$")
    ax.set_ylabel("Probability / normalized payoff")
    ax.set_title("Metric aging: measured success can rise as true value decays")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()

    pdf_path = output_path / "aging_dynamics.pdf"
    png_path = output_path / "aging_dynamics.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def plot_optimal_age_heatmap(output_dir: str | Path) -> dict[str, str]:
    """Create Figure 2: optimal replacement age over learning-rate grid."""

    output_path = Path(output_dir)
    _ensure_directory(output_path)
    p_grid, q_grid, k_star = make_heatmap_grid(delta=0.95, gamma=0.90, k_max=180)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.imshow(
        k_star,
        origin="lower",
        aspect="auto",
        extent=[float(p_grid.min()), float(p_grid.max()), float(q_grid.min()), float(q_grid.max())],
    )
    fig.colorbar(im, ax=ax, label="Optimal replacement age $K^*$")
    contours = ax.contour(p_grid, q_grid, k_star, levels=[5, 10, 15, 25, 40, 70], linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=8)
    ax.set_xlabel("Productive learning rate $p$")
    ax.set_ylabel("Hacking learning rate $q$")
    ax.set_title("Optimal replacement age across learning environments")
    fig.tight_layout()

    pdf_path = output_path / "optimal_age_heatmap.pdf"
    png_path = output_path / "optimal_age_heatmap.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}
