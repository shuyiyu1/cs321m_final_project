# Metric Aging Code Submission

This repository contains **only the reproducible code** used to generate the figures and tables for the CS231M final project on metric aging, reward hacking, and optimal metric replacement. It intentionally excludes the manuscript PDF, manuscript source, and LaTeX files.

The project studies a dynamic AI measurement-science setting in which a public evaluation metric can first encourage productive learning and later become increasingly vulnerable to reward hacking. The code regenerates the numerical outputs used in the paper: the metric-aging dynamics figure, the optimal replacement heatmap, and the regime-comparison table.

## Repository structure

```text
.
├── README.md                         # Setup and reproduction instructions
├── requirements.txt                  # Exact Python package versions
├── pyproject.toml                    # Editable install metadata
├── Makefile                          # Convenience commands: reproduce/test/clean
├── src/metric_aging/
│   ├── __init__.py
│   ├── model.py                      # Core closed-form model equations
│   ├── policies.py                   # Replacement policies and baseline comparisons
│   ├── experiments.py                # Experiment definitions and result generation
│   └── visualization.py              # Plotting functions for figures
├── scripts/
│   ├── run_all.py                    # End-to-end reproduction script
│   ├── make_figures.py               # Regenerate figures only
│   └── make_tables.py                # Regenerate tables only
└── tests/
    └── test_model.py                 # Unit tests for probabilities and policies
```

Generated files are written to `outputs/` when the scripts are run. The `outputs/` directory is ignored by git so that the repository stays code-only.

## Environment setup

The code was tested with Python 3.11. Package versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Step-by-step reproduction

From the repository root, run:

```bash
python scripts/run_all.py --output-dir outputs --seed 231
```

This command creates the following outputs:

| Result item | Generated file(s) | Producing code |
|---|---|---|
| Metric-aging dynamics figure | `outputs/figures/aging_dynamics.pdf`, `outputs/figures/aging_dynamics.png` | `plot_aging_dynamics` in `src/metric_aging/visualization.py` |
| Optimal replacement heatmap | `outputs/figures/optimal_age_heatmap.pdf`, `outputs/figures/optimal_age_heatmap.png` | `plot_optimal_age_heatmap` in `src/metric_aging/visualization.py` |
| Regime-comparison table | `outputs/tables/regime_comparison.csv`, `outputs/tables/regime_comparison.tex` | `make_regime_table` in `src/metric_aging/experiments.py` |
| Reproduction metadata | `outputs/summary.json` | `scripts/run_all.py` |

You can also use the Makefile:

```bash
make reproduce
make test
```

## Running selected parts

Regenerate figures only:

```bash
python scripts/make_figures.py --output-dir outputs/figures --seed 231
```

Regenerate tables only:

```bash
python scripts/make_tables.py --output-dir outputs/tables --seed 231
```

## Tests

Run the test suite with:

```bash
pytest -q
```

The tests check that the latent-state probabilities are valid, that the equal-rate formula agrees with the general convolution expression, that true value is single-peaked under the default regime, and that the balanced-regime replacement age is stable.

## Expected runtime and computational requirements

The full reproduction script should run in under 10 seconds on a standard laptop CPU. The tests should run in under 5 seconds. No GPU is required. Memory usage is small because the largest numerical grid is only 24 by 24 parameter settings.

## Data and random seeds

No external dataset is required. All figures and tables are generated from the closed-form model in `src/metric_aging/model.py`. A `--seed` argument is accepted by all scripts and set for reproducibility, although the current experiments are deterministic and do not use random sampling.

## Main model parameters

The core parameters are:

- `p`: probability per period that the agent learns the productive action before it has been learned.
- `q`: probability per period that the agent learns the hacking action after productive learning.
- `delta`: evaluator discount factor.
- `gamma`: probability that productive effort or hacking produces a positive public metric signal.
- `Kmax`: maximum replacement age considered by enumeration.

Default experiment settings are defined in `src/metric_aging/experiments.py`.

## Example usage from Python

```python
from metric_aging import LearningParams, optimal_replacement_age

params = LearningParams(p=0.08, q=0.08, delta=0.95, gamma=0.90)
result = optimal_replacement_age(params, k_max=1000)
print(result.k_star, result.value)
```
