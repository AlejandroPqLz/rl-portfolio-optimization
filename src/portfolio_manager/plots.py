"""Figures for the memoir, thesis section 4.4.

Outside the code freeze by D73, and the reason is a constraint rather than a convenience: this
module **consumes** what the rest of the pipeline has already written and recomputes none of it. No
figure here can change a reported number, because none of them is computed here. Everything comes
from ``results/``: the tables that ``evaluate.py`` wrote, the solver output, the training metrics.

The one exception is declared rather than hidden. The variance-ratio curve is computed here from
``data/returns_matrix.csv``, because the deferred-work table of the build brief puts it in this
module; it is a property of the input data and of no result, so computing it cannot move a figure
of the comparison either.

What is read, and where it comes from
-------------------------------------
``results/dp/<config>.npz``            value, policy and the count of optimal actions per state
``results/runs/<config>/seed_*/``      the training metrics and the saved policies
``results/*.csv``                      the tables of ``evaluate.py``, quantiles included
``data/returns_matrix.csv``            the historical matrix, for the variance ratio alone

Labels are Spanish, because the memoir is; the keys are English, because the code is. The
dictionary is the only place the two meet.

Command line
------------
    uv run python -m portfolio_manager.plots --all
    uv run python -m portfolio_manager.plots --policy-maps --config A_gbm
"""

# Imports
# =====================================================================
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # figures are written to files, never shown; there is no display in a batch
import matplotlib.pyplot as plt  # noqa: E402

from portfolio_manager.generator import load_matrix  # noqa: E402

# Constants
# =====================================================================
RESULTS_DIRECTORY = Path("results")
FIGURES_DIRECTORY = RESULTS_DIRECTORY / "figures"
DP_DIRECTORY = RESULTS_DIRECTORY / "dp"
RUNS_DIRECTORY = RESULTS_DIRECTORY / "runs"

VARIANCE_RATIO_YEARS = (1, 2, 3, 5, 7, 10)
"""Horizons of the variance-ratio curve, in years, as the data document fixes them."""

MONTHS_PER_YEAR = 12
FIGURE_DPI = 150

LABELS: dict[str, str] = {
    # Axes
    "relative_wealth": "Riqueza relativa $W_t/G$",
    "time_elapsed": "Fracción de horizonte $t/T$",
    "months": "Meses",
    "timesteps": "Pasos de entrenamiento",
    "probability_of_reaching_goal": "$P(W_T \\geq G)$",
    "terminal_wealth": "Riqueza terminal (USD)",
    "variance_ratio": "Ratio de varianza $VR(k)$",
    "horizon_years": "Horizonte $k$ (años)",
    "equity_weight": "Peso en renta variable",
    "policy": "Estrategia",
    # Series
    "dp_table": "Óptimo tabulado",
    "agent": "Agente",
    "random_policy": "Política aleatoria",
    "historical": "Serie histórica",
    "independence": "Independencia",
    "goal": "Meta",
    "degenerate": "Óptimo degenerado",
    # Titles
    "policy_map_dp": "Mapa de política: óptimo tabulado",
    "policy_map_agent": "Mapa de política: agente",
    "policy_map_difference": "Discrepancia entre ambos mapas",
    "training_curves": "Curvas de aprendizaje sobre el conjunto de validación",
    "terminal_wealth_distribution": "Distribución de la riqueza terminal",
    "variance_ratio_curve": "Ratio de varianza de la serie histórica",
    # Notes
    "excluded_note": "Zonas en blanco: estados con más de una cartera óptima",
    "quantile_note": "Cajas: cuantiles 25-75; bigotes: 5-95; línea: mediana",
    "lo_mackinlay": "Corrección de Lo y MacKinlay, observaciones solapadas",
}
"""English keys, Spanish values. The figures are read in Spanish and the code is written in
English, so the translation happens once and in one place rather than inline at every call."""


# ======================================================================================
# Reading what the pipeline wrote
# ======================================================================================
def read_table(path: Path) -> dict[str, dict[str, float]]:
    """Read a results table written by ``evaluate.py``."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        name = row.pop("policy")
        table[name] = {key: float(value) if value else float("nan") for key, value in row.items()}
    return table


def read_training_metrics(
    configuration: str, runs: Path = RUNS_DIRECTORY, *, train_cost_bp: float | None = None
) -> dict[str, dict[str, Any]]:
    """Read every seed's ``metrics.json`` for one configuration, in seed order.

    ``train_cost_bp`` selects the run directory, since an agent trained at one cost is a different
    agent from one trained at another; the base case is used when it is omitted.
    """
    from portfolio_manager.env import load_problem
    from portfolio_manager.train import run_directory_name

    cost = load_problem().cost_bp if train_cost_bp is None else float(train_cost_bp)
    root = runs / run_directory_name(configuration, cost)
    found = sorted(root.glob("seed_*/metrics.json"), key=lambda path: int(path.parent.name.split("_")[1]))
    return {path.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in found}


def _save(figure: plt.Figure, path: Path) -> Path:
    """Write a figure and close it, so a long run does not accumulate open figures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


# ======================================================================================
# Policy maps
# ======================================================================================
def equity_weight_map(policy: np.ndarray, column: int = 0) -> np.ndarray:
    """The weight of one column across the table, as a map over wealth and time."""
    return np.asarray(policy, dtype=np.float64)[:, :, column]


def plot_policy_maps(
    configuration: str,
    *,
    seed: str = "seed_0",
    train_cost_bp: float,
    directory: Path = FIGURES_DIRECTORY,
    dp_directory: Path = DP_DIRECTORY,
    runs: Path = RUNS_DIRECTORY,
) -> Path:
    """The tabulated optimum against the agent, over the wealth and time-remaining plane.

    ``train_cost_bp`` names the cost the plotted agent was trained under and is required for the
    same reason it is in ``evaluate.py``: reading it from ``config/problem.yml`` would quietly
    plot a different agent than the caller asked for, which is the defect that produced a table
    of ten-basis-point agents under a file named for zero.

    Degenerate states are left blank. Most of the plane has several optimal portfolios, and there
    the table shows whichever one the tie-break selected; colouring them would put a confident
    pattern on the part of the figure that carries no information, and the eye would read it as the
    optimal policy. The white region is the honest statement that the comparison of the first
    hypothesis has nothing to say there.
    """
    table = np.load(dp_directory / f"{configuration}.npz")
    tabulated = equity_weight_map(table["policy"])
    degenerate = np.asarray(table["n_optimal_actions"]) > 1
    grid = np.asarray(table["wealth_grid"], dtype=np.float64)
    horizon = tabulated.shape[1]
    extent = (0.0, 1.0, float(grid[0]), float(grid[-1]))

    from portfolio_manager.evaluate import agent_run_directory

    archive = agent_run_directory(configuration, train_cost_bp, runs) / seed / "policy.zip"
    learned = None
    if archive.is_file():
        from portfolio_manager.evaluate import load_trained_policy

        policy = load_trained_policy(archive)
        months = np.arange(horizon)
        learned = np.empty_like(tabulated)
        for t in months:
            observations = np.stack(
                [grid, np.full(grid.size, t / horizon), np.zeros(grid.size)], axis=-1
            ).astype(np.float32)
            learned[:, t] = policy(observations, int(t))[:, 0]

    panels = 3 if learned is not None else 1
    figure, axes = plt.subplots(
        1, panels, figsize=(5.6 * panels, 4.4), squeeze=False, layout="constrained"
    )
    masked = np.ma.masked_where(degenerate, tabulated)

    image = axes[0][0].imshow(
        masked, origin="lower", aspect="auto", extent=extent, vmin=0.0, vmax=1.0, cmap="viridis"
    )
    axes[0][0].set_title(LABELS["policy_map_dp"])
    figure.colorbar(image, ax=axes[0][0], label=LABELS["equity_weight"])

    if learned is not None:
        second = axes[0][1].imshow(
            np.ma.masked_where(degenerate, learned),
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axes[0][1].set_title(f"{LABELS['policy_map_agent']} ({seed})")
        figure.colorbar(second, ax=axes[0][1], label=LABELS["equity_weight"])

        difference = np.ma.masked_where(degenerate, learned - tabulated)
        third = axes[0][2].imshow(
            difference, origin="lower", aspect="auto", extent=extent, vmin=-1.0, vmax=1.0, cmap="coolwarm"
        )
        axes[0][2].set_title(LABELS["policy_map_difference"])
        figure.colorbar(third, ax=axes[0][2], label=LABELS["equity_weight"])

    for axis in axes[0]:
        axis.set_xlabel(LABELS["time_elapsed"])
        axis.set_ylabel(LABELS["relative_wealth"])
        axis.set_ylim(0.0, 2.0)
    figure.suptitle(f"{configuration} · {LABELS['excluded_note']}", fontsize=9)
    return _save(figure, directory / f"policy_maps_{configuration}.png")


# ======================================================================================
# Training curves
# ======================================================================================
def plot_training_curves(
    configuration: str, *, directory: Path = FIGURES_DIRECTORY, runs: Path = RUNS_DIRECTORY
) -> Path:
    """Held-out success probability against training steps, one line per seed.

    Five seeds are plotted individually rather than averaged. The spread between them is part of
    the result: a mean curve would hide whether the method is reliable or whether one seed carried
    the configuration.
    """
    metrics = read_training_metrics(configuration, runs)
    if not metrics:
        raise FileNotFoundError(f"no runs under {runs / configuration}")

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    for name, document in metrics.items():
        history = document["held_out"]["history"]
        axis.plot(
            [point["timesteps"] for point in history],
            [point["probability_of_reaching_goal"] for point in history],
            linewidth=1.1,
            label=name.replace("seed_", "semilla "),
        )
    random_value = float(next(iter(metrics.values()))["random_policy_probability"])
    axis.axhline(random_value, color="black", linestyle="--", linewidth=1.0, label=LABELS["random_policy"])

    axis.set_xlabel(LABELS["timesteps"])
    axis.set_ylabel(LABELS["probability_of_reaching_goal"])
    axis.set_title(f"{LABELS['training_curves']} · {configuration}")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.3)
    return _save(figure, directory / f"training_curves_{configuration}.png")


# ======================================================================================
# Terminal wealth
# ======================================================================================
def plot_terminal_wealth(
    table_path: Path, *, goal: float | None = None, directory: Path = FIGURES_DIRECTORY
) -> Path:
    """The distribution of terminal wealth per policy, from the quantiles already recorded.

    Drawn from the five quantiles the evaluation wrote and not from the raw outcomes, which is the
    point of D73: the figure can show only what was computed, so it cannot disagree with the table
    beside it in the memoir.
    """
    table = read_table(table_path)
    names = list(table)
    quantiles = np.array(
        [[table[name][f"terminal_wealth_p{q}_usd"] for q in (5, 25, 50, 75, 95)] for name in names]
    )

    figure, axis = plt.subplots(figsize=(1.35 * len(names) + 3.0, 4.6))
    positions = np.arange(len(names))
    axis.bxp(
        [
            {
                "label": name,
                "whislo": row[0],
                "q1": row[1],
                "med": row[2],
                "q3": row[3],
                "whishi": row[4],
                "fliers": [],
            }
            for name, row in zip(names, quantiles, strict=True)
        ],
        positions=positions,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#cfe3f3", "edgecolor": "#30506b"},
        medianprops={"color": "#b03030", "linewidth": 1.6},
    )
    if goal is not None:
        axis.axhline(goal, color="black", linestyle="--", linewidth=1.0, label=LABELS["goal"])
        axis.legend(fontsize=8)

    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axis.set_xlabel(LABELS["policy"])
    axis.set_ylabel(LABELS["terminal_wealth"])
    axis.set_title(f"{LABELS['terminal_wealth_distribution']} · {LABELS['quantile_note']}", fontsize=10)
    axis.grid(alpha=0.3, axis="y")
    return _save(figure, directory / f"terminal_wealth_{table_path.stem}.png")


# ======================================================================================
# The variance ratio
# ======================================================================================
def variance_ratio(log_returns: np.ndarray, q: int) -> dict[str, float]:
    """Variance ratio at horizon ``q`` with the heteroskedasticity-robust statistic.

    Equation ``eq:vr`` estimated on overlapping observations, which is what Lo and MacKinlay (1988)
    provide the sampling theory for. The overlapping estimator uses the sample far more efficiently
    than non-overlapping blocks, at the cost of a correlation between the observations that the
    naive standard error ignores; their correction is what makes the statistic readable.

    The robust form is used rather than the homoskedastic one because monthly equity returns are
    not homoskedastic, and under heteroskedasticity alone the homoskedastic statistic rejects
    independence far too often. Rejecting the random walk for the wrong reason is exactly the
    mistake the curve exists to avoid.

    Args:
        log_returns: ``(n,)`` log returns of one asset.
        q: Horizon in periods.

    Returns:
        The ratio, the robust statistic and its two-sided p-value.
    """
    from scipy.stats import norm

    returns = np.asarray(log_returns, dtype=np.float64)
    n = returns.size
    if q < 1 or q > n // 2:
        raise ValueError(f"the horizon must lie between 1 and {n // 2}, got {q}")

    mean = returns.mean()
    centred = returns - mean
    # One-period variance, and the q-period variance from overlapping sums, each with the
    # small-sample corrections of the paper.
    variance_one = (centred**2).sum() / (n - 1)
    rolling = np.convolve(returns, np.ones(q), mode="valid") - q * mean
    denominator = q * (n - q + 1) * (1.0 - q / n)
    variance_q = (rolling**2).sum() / denominator
    ratio = variance_q / variance_one

    # Heteroskedasticity-consistent variance of the ratio.
    squared = centred**2
    total = squared.sum()
    theta = 0.0
    for lag in range(1, q):
        delta = (squared[lag:] * squared[:-lag]).sum() * n / (total**2)
        theta += (2.0 * (q - lag) / q) ** 2 * delta
    statistic = (ratio - 1.0) / np.sqrt(theta) if theta > 0.0 else float("nan")
    return {
        "q": int(q),
        "variance_ratio": float(ratio),
        "statistic": float(statistic),
        "p_value": float(2.0 * (1.0 - norm.cdf(abs(statistic)))) if np.isfinite(statistic) else float("nan"),
    }


def plot_variance_ratio(*, directory: Path = FIGURES_DIRECTORY) -> tuple[Path, dict[str, Any]]:
    """The variance-ratio curve of the historical matrix, one line per asset.

    This is the measurement that bounds the central limitation of the generator. The block
    bootstrap destroys dependence beyond the block length by construction, so the synthetic curve
    is flat at one by design; what the historical curve does at three to ten years is the size of
    what was destroyed, and it is reported with its interval because at ten years the estimate
    rests on about three and a half independent windows.
    """
    columns, matrix = load_matrix()
    log_returns = np.log1p(matrix)

    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    measured: dict[str, Any] = {}
    for index, name in enumerate(columns):
        points = [variance_ratio(log_returns[:, index], year * MONTHS_PER_YEAR) for year in VARIANCE_RATIO_YEARS]
        measured[name] = points
        axis.plot(
            VARIANCE_RATIO_YEARS,
            [point["variance_ratio"] for point in points],
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=name,
        )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label=LABELS["independence"])

    # Logarithmic, because the two columns whose return is dominated by a persistent accrual term
    # reach a ratio of fourteen while equity sits at one; on a linear axis the series the argument
    # is actually about would be a flat line at the bottom of the frame.
    axis.set_yscale("log")
    axis.set_xlabel(LABELS["horizon_years"])
    axis.set_ylabel(LABELS["variance_ratio"])
    axis.set_title(f"{LABELS['variance_ratio_curve']} · {LABELS['lo_mackinlay']}", fontsize=10)
    axis.set_xticks(VARIANCE_RATIO_YEARS)
    axis.set_yticks([0.5, 1, 2, 5, 10])
    axis.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, which="both")
    return _save(figure, directory / "variance_ratio.png"), measured


# ======================================================================================
# Command line
# ======================================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Figures for section 4.4. Reads results/, computes no result.")
    parser.add_argument("--all", action="store_true", help="every figure for which the inputs exist")
    parser.add_argument("--policy-maps", action="store_true", help="the tabulated optimum against the agent")
    parser.add_argument("--training", action="store_true", help="held-out probability against training steps")
    parser.add_argument("--wealth", action="store_true", help="terminal wealth per policy, from the quantiles")
    parser.add_argument("--variance-ratio", action="store_true", help="the curve of the historical matrix")
    parser.add_argument("--config", default="A_gbm", help="configuration whose maps and curves are drawn")
    parser.add_argument("--seed", default="seed_0", help="seed whose policy map is drawn")
    parser.add_argument(
        "--train-cost-bp",
        type=float,
        default=None,
        help="cost the plotted agent was trained under, selecting the run directory; the base case when omitted",
    )
    parser.add_argument("--table", type=Path, default=None, help="results table for the wealth figure")
    args = parser.parse_args(argv)

    requested = args.all or any((args.policy_maps, args.training, args.wealth, args.variance_ratio))
    if not requested:
        parser.print_help()
        return 0

    written: list[Path] = []
    skipped: list[str] = []
    try:
        if args.all or args.policy_maps:
            if (DP_DIRECTORY / f"{args.config}.npz").is_file():
                from portfolio_manager.env import load_problem as _load_problem

                train_cost = (
                    _load_problem().cost_bp if args.train_cost_bp is None else args.train_cost_bp
                )
                written.append(
                    plot_policy_maps(args.config, seed=args.seed, train_cost_bp=train_cost)
                )
            else:
                skipped.append(f"policy maps: no table at {DP_DIRECTORY / f'{args.config}.npz'}")
        if args.all or args.training:
            if read_training_metrics(args.config):
                written.append(plot_training_curves(args.config))
            else:
                skipped.append(f"training curves: no runs under {RUNS_DIRECTORY / args.config}")
        if args.all or args.wealth:
            table = args.table or next(iter(sorted(RESULTS_DIRECTORY.glob("*.csv"))), None)
            if table is not None:
                from portfolio_manager.env import load_problem

                written.append(plot_terminal_wealth(table, goal=load_problem().goal_usd))
            else:
                skipped.append("terminal wealth: no results table")
        if args.all or args.variance_ratio:
            path, _ = plot_variance_ratio()
            written.append(path)
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    for path in written:
        print(f"wrote {path}")
    for note in skipped:
        print(f"skipped {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
