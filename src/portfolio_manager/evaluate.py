"""Evaluation protocol: scoring any policy on the frozen test set, thesis subsections 4.2.4 and 4.3.5.

Contract
--------
This module scores **any** callable ``policy(observations, t) -> weights``. It never asks which
strategy it is holding, and no branch anywhere below mentions a comparator by name. That is what
makes the comparison of H2 fair in the only sense that matters operationally: the agent, the
tabulated optimum and the five comparator portfolios go through one code path, over one tensor, in
one order.

Metrics, from thesis 4.2.4
--------------------------
Primary, ``P(W_T >= G)``: the quantity an investor can act on, and the one the literature reports.
Reported beside it, and not instead of it, the mean terminal reward, because the system optimises
the lower partial moment of order 2 and the two need not order policies alike; that divergence is
a finding rather than an inconsistency. Secondary: the conditional expected shortfall
``E[G - W_T | W_T < G]``, which says how badly the failures fail; the quantiles of terminal wealth,
which are the distribution the single probability summarises; and the annualised turnover, which is
what the transaction cost is charged on.

Confidence intervals come from a bootstrap over the test scenarios, drawn with **common random
numbers across policies**: every policy is resampled on the same scenario indices, so the intervals
are comparable and the paired difference of proportions of H2 can later be built from the same
draws rather than from a second, independent resampling.

Command line
------------
    uv run python -m portfolio_manager.evaluate --policies baselines --test-set B_L12 --cost-bp 10
"""

# Imports
# =====================================================================
import argparse
import csv
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from portfolio_manager.baselines import Policy, build_baselines, describe_baselines, load_baselines
from portfolio_manager.env import ProblemConfig, load_problem, rollout
from portfolio_manager.generator import (
    config_path,
    frozen_tensor,
    load_matrix,
    load_matrix_metadata,
    load_test_set,
    sha256_of_file,
    write_sidecar,
)

# Constants
# =====================================================================
RESULTS_DIRECTORY = Path("results")

QUANTILES = (5, 25, 50, 75, 95)
"""Quantiles of terminal wealth reported as the secondary distributional metric (thesis 4.2.4)."""

N_BOOTSTRAP = 10_000
"""Resamples behind each confidence interval. A protocol parameter rather than a property of the
problem, so it is overridable from the command line."""

BOOTSTRAP_CHUNK = 250
"""Resamples drawn at a time. Pure memory bound: the index block is ``(chunk, n_scenarios)``."""

CONFIDENCE = 0.95
"""Nominal coverage of the reported interval."""

MONTHS_PER_YEAR = 12


# ======================================================================================
# Metrics
# ======================================================================================
def bootstrap_blocks(rng: np.random.Generator, n_scenarios: int, n_bootstrap: int) -> Iterator[np.ndarray]:
    """Yield blocks of resampled scenario indices.

    Drawn once and reused across policies, so that every strategy is scored on the same resampled
    worlds. Independent draws per policy would leave the intervals incomparable and would throw
    away the pairing the frozen test set exists to provide.
    """
    for start in range(0, n_bootstrap, BOOTSTRAP_CHUNK):
        size = min(BOOTSTRAP_CHUNK, n_bootstrap - start)
        yield rng.integers(0, n_scenarios, size=(size, n_scenarios))


def metrics(
    terminal_wealth: np.ndarray,
    reward: np.ndarray,
    total_turnover: np.ndarray,
    *,
    goal: float,
    horizon: int,
) -> dict[str, float]:
    """The point estimates of thesis 4.2.4 for one policy.

    Args:
        terminal_wealth: ``(n,)`` terminal wealth per scenario.
        reward: ``(n,)`` terminal reward per scenario.
        total_turnover: ``(n,)`` turnover summed over the horizon.
        goal: ``G``.
        horizon: ``T`` in months.

    Returns:
        Mapping of metric name to value. The conditional shortfall is ``nan`` when no scenario
        fails, which is the honest reading of an expectation conditioned on an empty set.
    """
    shortfall = goal - terminal_wealth
    failed = shortfall > 0.0
    return {
        "n_scenarios": int(terminal_wealth.size),
        "probability_of_reaching_goal": float((~failed).mean()),
        "conditional_expected_shortfall_usd": float(shortfall[failed].mean()) if failed.any() else float("nan"),
        "probability_of_shortfall": float(failed.mean()),
        "mean_terminal_reward": float(reward.mean()),
        "mean_terminal_wealth_usd": float(terminal_wealth.mean()),
        **{f"terminal_wealth_p{q}_usd": float(np.quantile(terminal_wealth, q / 100.0)) for q in QUANTILES},
        # Turnover sums purchases and sales, so a complete switch between two assets is 2.0.
        "annualised_turnover": float(total_turnover.mean() * MONTHS_PER_YEAR / horizon),
    }


def evaluate_policies(
    policies: dict[str, Policy],
    scenarios: np.ndarray,
    *,
    problem: ProblemConfig | None = None,
    cost_rate: float | None = None,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
) -> dict[str, dict[str, float]]:
    """Score every policy over the same tensor, in the same order, with the same cost.

    Args:
        policies: Mapping from name to callable.
        scenarios: ``(n, T, N)`` frozen tensor.
        problem: Base case; read from configuration when omitted.
        cost_rate: Overrides ``lambda``, for the robustness sweep.
        seed: Seed of the bootstrap stream.
        n_bootstrap: Resamples behind each interval.
        confidence: Nominal coverage.

    Returns:
        Mapping from policy name to its metrics, including the interval bounds.
    """
    problem = problem or load_problem()
    horizon = int(scenarios.shape[1])

    results = {
        name: rollout(scenarios, policy, problem=problem, cost_rate=cost_rate)
        for name, policy in policies.items()
    }
    table = {
        name: metrics(
            result["terminal_wealth"],
            result["reward"],
            result["total_turnover"],
            goal=problem.goal_usd,
            horizon=horizon,
        )
        for name, result in results.items()
    }

    # Common random numbers: one stream of resampled indices, applied to every policy.
    success = {name: (result["terminal_wealth"] >= problem.goal_usd) for name, result in results.items()}
    estimates: dict[str, list[np.ndarray]] = {name: [] for name in policies}
    rng = np.random.default_rng(seed)
    for block in bootstrap_blocks(rng, int(scenarios.shape[0]), n_bootstrap):
        for name, flags in success.items():
            estimates[name].append(flags[block].mean(axis=1))

    alpha = (1.0 - confidence) / 2.0
    for name in policies:
        draws = np.concatenate(estimates[name])
        table[name]["probability_ci_low"] = float(np.quantile(draws, alpha))
        table[name]["probability_ci_high"] = float(np.quantile(draws, 1.0 - alpha))
        table[name]["bootstrap_resamples"] = int(draws.size)
        table[name]["bootstrap_confidence"] = confidence
    return table


# ======================================================================================
# Output
# ======================================================================================
COLUMNS = (
    "policy",
    "n_scenarios",
    "probability_of_reaching_goal",
    "probability_ci_low",
    "probability_ci_high",
    "conditional_expected_shortfall_usd",
    "mean_terminal_reward",
    "terminal_wealth_p5_usd",
    "terminal_wealth_p25_usd",
    "terminal_wealth_p50_usd",
    "terminal_wealth_p75_usd",
    "terminal_wealth_p95_usd",
    "annualised_turnover",
)
"""Columns of the results table, in the order thesis 4.2.4 introduces the metrics."""


def write_table(table: dict[str, dict[str, float]], path: Path) -> Path:
    """Write the results table as CSV, one row per policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for name, row in table.items():
            writer.writerow({"policy": name, **row})
    return path


def _print_parameters(described: dict[str, Any], cost_bp: float) -> None:
    """Print what the comparators were built from, above the figures they produced.

    The two values the author has to be able to read off the output: the fixed-income split, which
    must be the sample-based one that A28 solved and not the nominal 0.621, and the unclipped
    Merton fractions, which show the corner at ``gamma = 2`` as a consequence rather than a choice.
    """
    shares = ", ".join(f"{name} {share:.6f}" for name, share in described["fixed_income_split"].items())
    print(f"fixed-income split   {shares}")
    print(
        f"glide path           {described['glide_path']['equity_weight_start']:.3f} -> "
        f"{described['glide_path']['equity_weight_end']:.3f} equity, {described['glide_path']['shape']}"
    )
    for name, fraction in described["merton"].items():
        corner = "  <- corner, no-leverage constraint binds (D53)" if fraction["unclipped"] > 1.0 else ""
        print(
            f"merton {name:<13} w* = {fraction['unclipped']:.4f} -> held {fraction['weight']:.4f}{corner}"
        )
    print(f"transaction cost     {cost_bp:g} bp per unit of turnover\n")


def _print_table(table: dict[str, dict[str, float]], goal: float) -> None:
    """Print the table the author reads on the evening of the build."""
    header = (
        f"{'policy':18} {'P(W_T>=G)':>10} {'95% CI':>18} {'E[G-W_T|fail]':>14} "
        f"{'median W_T':>12} {'mean LPM2':>11} {'turn/yr':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, row in table.items():
        interval = f"[{row['probability_ci_low']:.4f}, {row['probability_ci_high']:.4f}]"
        shortfall = row["conditional_expected_shortfall_usd"]
        shortfall_text = "n/a" if np.isnan(shortfall) else f"{shortfall:,.0f}"
        print(
            f"{name:18} {row['probability_of_reaching_goal']:10.4f} {interval:>18} "
            f"{shortfall_text:>14} {row['terminal_wealth_p50_usd']:12,.0f} "
            f"{row['mean_terminal_reward']:11.5f} {row['annualised_turnover']:8.3f}"
        )
    print(f"\ngoal {goal:,.0f} USD. Turnover sums purchases and sales, so a full switch is 2.0 per month.")


# ======================================================================================
# Command line
# ======================================================================================
def run(
    test_set: str,
    cost_bp: float | None,
    *,
    seed: int,
    n_bootstrap: int = N_BOOTSTRAP,
    argv: list[str] | None = None,
    results_directory: Path = RESULTS_DIRECTORY,
) -> tuple[dict[str, dict[str, float]], Path, dict[str, Any]]:
    """Evaluate the comparator set on one frozen configuration and write the table.

    Args:
        test_set: Configuration key of the frozen recipe, for example ``B_L12``.
        cost_bp: Transaction cost in basis points; the base case when omitted.
        seed: Seed of the bootstrap stream.
        n_bootstrap: Resamples behind each interval.
        argv: Command line recorded in the side-car.
        results_directory: Where the table is written.

    Returns:
        Tuple ``(table, path, described)``, the last being the comparator parameters.
    """
    problem = load_problem()
    recipe = load_test_set()
    columns, matrix = load_matrix()
    scenarios, provenance = frozen_tensor(test_set, recipe, matrix=matrix)

    cost_rate = problem.cost_rate if cost_bp is None else float(cost_bp) * 1e-4
    effective_bp = problem.cost_bp if cost_bp is None else float(cost_bp)

    config = load_baselines(columns=columns)
    statistics = load_matrix_metadata()["statistics"]
    horizon = int(scenarios.shape[1])
    policies = build_baselines(config, statistics=statistics, horizon=horizon)

    table = evaluate_policies(
        policies, scenarios, problem=problem, cost_rate=cost_rate, seed=seed, n_bootstrap=n_bootstrap
    )

    described = describe_baselines(config, statistics=statistics, horizon=horizon)
    path = results_directory / f"heuristics_{test_set}_{effective_bp:g}bp.csv"
    write_table(table, path)
    write_sidecar(
        path,
        {
            "policies": "baselines",
            "cost_bp": effective_bp,
            "cost_rate": cost_rate,
            "seed": seed,
            "bootstrap_resamples": n_bootstrap,
            "bootstrap_confidence": CONFIDENCE,
            "problem": {
                "goal_usd": problem.goal_usd,
                "horizon_months": problem.horizon_months,
                "monthly_contribution_usd": problem.monthly_contribution_usd,
                "initial_wealth_usd": problem.initial_wealth_usd,
                "objective": problem.objective,
            },
            "baselines": described,
            "source_matrix_sha256": load_matrix_metadata()["sha256"],
            "config_sha256": {
                "baselines": sha256_of_file(config_path("baselines")),
                "problem": sha256_of_file(config_path("problem")),
                "test_set": sha256_of_file(config_path("test_set")),
            },
            "status": "preliminary until the figures freeze",
            **provenance,
        },
        argv=argv,
    )
    return table, path, described


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluation protocol: score policies on the frozen test set.")
    parser.add_argument("--policies", choices=("baselines",), required=True, help="which set of policies to score")
    parser.add_argument("--test-set", required=True, help="configuration of the frozen recipe, for example B_L12")
    parser.add_argument("--cost-bp", type=float, default=None, help="transaction cost; the base case when omitted")
    parser.add_argument("--seed", type=int, default=None, help="bootstrap seed; the validation band when omitted")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="resamples behind each interval")
    args = parser.parse_args(argv)

    try:
        problem = load_problem()
        seed = args.seed if args.seed is not None else int(load_test_set()["seeds"]["validation"]["min"])
        table, path, described = run(
            args.test_set,
            args.cost_bp,
            seed=seed,
            n_bootstrap=args.n_bootstrap,
            argv=argv if argv is not None else sys.argv,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"test set {args.test_set}, {table[next(iter(table))]['n_scenarios']:,} scenarios, seed {seed}\n")
    _print_parameters(described, problem.cost_bp if args.cost_bp is None else args.cost_bp)
    _print_table(table, problem.goal_usd)
    print(f"\nwrote {path}")
    print("PRELIMINARY: these figures are not frozen until the code freeze of 10 October.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
