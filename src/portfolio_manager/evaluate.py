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

Beyond the comparators
----------------------
A trained agent and the tabulated optimum enter the same table through the same interface. When
agents are scored, each seed is also tested against two references on the same scenarios: against
``glide_path``, the best heuristic by probability in every configuration, the paired difference of
proportions of the second hypothesis; and against ``dp_table`` where one exists, the same test
plus the discrepancy index between the two policy maps, for the first. A reference run records the
digest of the comparator table so that a change meant to be non-functional can be shown to have
been one.

The two costs
-------------
A table involves two transaction costs and they are not the same number. The **training** cost
selects which agents are read, and nothing else; the **evaluation** cost enters the transition,
and nothing else. The sweep of thesis 4.2.3 scores one trained agent at 0, 10 and 25 basis
points, so collapsing them is not an available simplification: ``--train-cost-bp`` defaults to
``--cost-bp`` because an agent is normally evaluated on the problem it was trained for, and both
reach the file name and the side-car whenever they differ. Reading the training cost from
``config/problem.yml`` instead, as this module once did, wrote a table of agents trained at ten
basis points to a file named and documented for zero.

Command line
------------
    uv run python -m portfolio_manager.evaluate --policies baselines --test-set B_L12 --cost-bp 10
    uv run python -m portfolio_manager.evaluate --policies all --test-set A_gbm --cost-bp 0
    uv run python -m portfolio_manager.evaluate --policies all --test-set B_L12 --train-cost-bp 10 --cost-bp 25
    uv run python -m portfolio_manager.evaluate --reference-run
"""

# Imports
# =====================================================================
import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from portfolio_manager.baselines import (
    Policy,
    build_baselines,
    describe_baselines,
    dp_table_policy,
    load_baselines,
)
from portfolio_manager.env import ProblemConfig, load_problem, project_to_simplex, rollout
from portfolio_manager.generator import (
    config_path,
    frozen_tensor,
    git_commit,
    load_matrix,
    load_matrix_metadata,
    load_test_set,
    sha256_of_file,
    write_sidecar,
)

# Constants
# =====================================================================
RESULTS_DIRECTORY = Path("results")
RUNS_DIRECTORY = RESULTS_DIRECTORY / "runs"
DP_DIRECTORY = RESULTS_DIRECTORY / "dp"

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

FREEZE_NOTICE = "PRELIMINARY: these figures are not frozen until the freeze of 11 October."


@dataclass(frozen=True)
class Costs:
    """The two transaction costs a table involves, which are not the same number.

    An agent is trained against one cost and may be scored against another: that is the whole of
    the sweep in thesis 4.2.3, where one trained policy meets 0, 10 and 25 basis points. Carrying
    them as one value is what let a table of agents trained at ten basis points be written to a
    file named for zero. They are separate fields here, both reach every file name and every
    side-car, and nothing downstream may collapse them.

    Attributes:
        train_bp: The cost the agents were trained under. Selects the run directory, nothing else.
        evaluate_bp: The cost charged during this evaluation. Enters the transition, nothing else.
    """

    train_bp: float
    evaluate_bp: float

    @property
    def rate(self) -> float:
        """``lambda`` of the transition, from the evaluation cost alone.

        Returns:
            The evaluation cost as a rate per unit of turnover (basis points times ``1e-4``).
        """
        return self.evaluate_bp * 1e-4

    @property
    def differ(self) -> bool:
        """Whether the training and the evaluation cost are different numbers.

        Returns:
            ``True`` when ``train_bp != evaluate_bp``; it decides the file-name suffix.
        """
        return self.train_bp != self.evaluate_bp

    @property
    def suffix(self) -> str:
        """File-name suffix: both costs whenever they differ, the single one when they agree.

        Returns:
            ``"train10bp_eval25bp"`` when the costs differ, ``"10bp"`` when they agree.
        """
        if self.differ:
            return f"train{self.train_bp:g}bp_eval{self.evaluate_bp:g}bp"
        return f"{self.evaluate_bp:g}bp"

    def describe(self) -> str:
        """Human-readable statement of both costs, for the console output.

        Returns:
            One line naming the training and the evaluation cost in basis points.
        """
        if self.differ:
            return f"trained at {self.train_bp:g} bp, evaluated at {self.evaluate_bp:g} bp"
        return f"{self.evaluate_bp:g} bp, training and evaluation"


def resolve_costs(cost_bp: float | None, train_cost_bp: float | None, problem: ProblemConfig) -> Costs:
    """Fix both costs from what the command line said, with the base case behind each.

    ``--train-cost-bp`` defaults to ``--cost-bp`` rather than to the configured base case: an agent
    is evaluated on the problem it was trained for unless the caller says otherwise. The sweep is
    the exception and states the training cost explicitly.

    Args:
        cost_bp: ``--cost-bp``, in basis points; ``cost_bp`` of ``config/problem.yml`` when ``None``.
        train_cost_bp: ``--train-cost-bp``, in basis points; the evaluation cost when ``None``.
        problem: Base case read from ``config/problem.yml``.

    Returns:
        Both costs, in basis points.
    """
    evaluate_bp = problem.cost_bp if cost_bp is None else float(cost_bp)
    train_bp = evaluate_bp if train_cost_bp is None else float(train_cost_bp)
    return Costs(train_bp=train_bp, evaluate_bp=evaluate_bp)


# ======================================================================================
# Metrics
# ======================================================================================
def bootstrap_blocks(rng: np.random.Generator, n_scenarios: int, n_bootstrap: int) -> Iterator[np.ndarray]:
    """Yield blocks of resampled scenario indices.

    Drawn once and reused across policies, so that every strategy is scored on the same resampled
    worlds. Independent draws per policy would leave the intervals incomparable and would throw
    away the pairing the frozen test set exists to provide.

    Args:
        rng: Bootstrap stream, seeded once by the caller.
        n_scenarios: Number of test scenarios, which is also the size of each resample.
        n_bootstrap: Total number of resamples to draw.

    Yields:
        Integer arrays of shape ``(chunk, n_scenarios)``, at most ``BOOTSTRAP_CHUNK`` rows each.
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
        terminal_wealth: ``(n,)`` terminal wealth per scenario, in USD.
        reward: ``(n,)`` terminal reward per scenario, in goal units.
        total_turnover: ``(n,)`` turnover summed over the horizon (purchases plus sales).
        goal: ``G``, in USD.
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
    success_out: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Score every policy over the same tensor, in the same order, with the same cost.

    Args:
        policies: Mapping from name to callable.
        scenarios: ``(n, T, N)`` frozen tensor of monthly simple returns.
        problem: Base case; read from ``config/problem.yml`` when omitted.
        cost_rate: Overrides ``lambda`` (a rate, not basis points), for the robustness sweep.
        seed: Seed of the bootstrap stream.
        n_bootstrap: Resamples behind each interval.
        confidence: Nominal coverage.
        success_out: Filled, when given, with the per-scenario success flags of each policy. The
            paired tests need them and the rollouts are the expensive part of this function, so
            they are handed out rather than recomputed. Scoring is unaffected.

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
    if success_out is not None:
        success_out.update(success)
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
# The policies that are not comparators
# ======================================================================================
def load_trained_policy(run_directory: Path, device: str = "cpu") -> Policy:
    """Load a saved agent and expose it through the comparator interface.

    The deterministic action of the Dirichlet policy is its mean, so this is the same policy the
    held-out evaluation of a training run reports, now scored by the same code as the five
    comparators and the table (section 2b).

    Args:
        run_directory: Directory of a run, holding ``policy.zip``, or the archive itself.
        device: Compute device; CPU for networks this size.

    Returns:
        A callable ``policy(observations, t) -> weights``.
    """
    # Imported here and not at module scope: loading the agent pulls in Torch and
    # Stable-Baselines3, and scoring the comparators must not pay for that.
    from stable_baselines3 import PPO

    from portfolio_manager.train import DirichletActorCriticPolicy, policy_from_model

    archive = run_directory if run_directory.suffix == ".zip" else run_directory / "policy.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"{archive} not found: no trained policy to score")
    model = PPO.load(
        archive, device=device, custom_objects={"policy_class": DirichletActorCriticPolicy}
    )
    return policy_from_model(model)


def agent_run_directory(
    configuration: str, train_cost_bp: float, runs_directory: Path = RUNS_DIRECTORY
) -> Path:
    """Where the agents trained on this configuration at this cost live.

    The only place the run directory is derived. It takes the training cost as a required argument
    and has no default to fall back on, which is the point: the previous version of this module
    read the cost from ``config/problem.yml`` and so scored agents trained at ten basis points
    into a table named and documented for zero. A cost that is not passed is now a signature
    error rather than a silently wrong directory.

    Args:
        configuration: Name of the frozen recipe in ``config/test_set.yml``, e.g. ``"B_L12"``.
        train_cost_bp: Cost the agents were trained under, in basis points.
        runs_directory: Root of the training runs.

    Returns:
        ``runs_directory / "<configuration>_<cost>bp"``, which may not exist.
    """
    from portfolio_manager.train import run_directory_name

    return runs_directory / run_directory_name(configuration, float(train_cost_bp))


def trained_policies(
    configuration: str, runs_directory: Path = RUNS_DIRECTORY, *, train_cost_bp: float
) -> dict[str, Policy]:
    """Every seed trained on one configuration at one training cost, keyed by seed.

    The protocol trains five independent seeds per configuration (D70), and they are scored
    individually rather than averaged: the spread across seeds is part of the result.

    ``train_cost_bp`` is the cost the agents were TRAINED under, which is not the cost they are
    being evaluated at. The sweep of thesis 4.2.3 scores one trained agent at several costs, so the
    two are independent and the training cost selects which run directory to read. It is
    **keyword-only and required**: there is no correct default, and the one that used to be here
    was the defect.

    Args:
        configuration: Name of the frozen recipe in ``config/test_set.yml``.
        runs_directory: Root of the training runs.
        train_cost_bp: Cost the agents were trained under, in basis points.

    Returns:
        Mapping from ``"agent_seed_<n>"`` to the deterministic policy of that seed, in seed order.

    Raises:
        FileNotFoundError: If no run directory exists for this cost, or it holds no seed. Silence
            here would produce a table whose agent rows are simply missing, or worse, a table of
            the wrong agents under the right name.
    """
    root = agent_run_directory(configuration, train_cost_bp, runs_directory)
    if not root.is_dir():
        raise FileNotFoundError(
            f"no agents trained at {train_cost_bp:g} bp on {configuration}: {root} does not exist. "
            f"Train them, or name the training cost with --train-cost-bp"
        )
    found = sorted(root.glob("seed_*/policy.zip"), key=lambda path: int(path.parent.name.split("_")[1]))
    if not found:
        raise FileNotFoundError(f"{root} holds no seed_*/policy.zip")
    return {f"agent_{archive.parent.name}": load_trained_policy(archive) for archive in found}


def table_policy(configuration: str, directory: Path = DP_DIRECTORY) -> dict[str, Policy]:
    """The tabulated optimum, if the solver has produced one for this configuration.

    Args:
        configuration: Name of the frozen recipe, which names ``<directory>/<configuration>.npz``.
        directory: Where the solver writes its tables.

    Returns:
        ``{"dp_table": policy}`` when the table exists, an empty mapping otherwise.
    """
    path = directory / f"{configuration}.npz"
    if not path.is_file():
        return {}
    return {"dp_table": dp_table_policy(np.load(path))}


# ======================================================================================
# Comparing two policy maps, and two sets of outcomes
# ======================================================================================
def discrepancy_index(
    table: Any, policy: Policy, scenarios: np.ndarray, problem: ProblemConfig
) -> dict[str, float]:
    """How far a learned policy sits from the tabulated optimum, over the states it actually visits.

    The first hypothesis is not settled by the success probability alone: two policies can reach
    the goal equally often and allocate quite differently, and the claim being tested is that the
    agent recovers the optimum rather than merely performing like it. So the two are compared as
    maps over the wealth and time-remaining plane, at the states the agent's own trajectories
    reach, because agreement where nothing goes says nothing.

    **States where the table has several optimal portfolios are excluded.** Under this problem most
    of the plane is degenerate, and there the table names one of many equally optimal portfolios by
    a tie-break rule; counting those disagreements would measure the tie-break. The share excluded
    is reported so the reader can see how much of the plane the index speaks for.

    Args:
        table: Loaded solver output, with ``wealth_grid``, ``policy`` and ``n_optimal_actions``.
        policy: The policy to compare against it.
        scenarios: Tensor whose trajectories define the states visited.
        problem: Base case.

    Returns:
        Mean and median distance, the share of states that disagree by more than one action-grid
        step, and the share of visited states the index had to discard.
    """
    grid = np.asarray(table["wealth_grid"], dtype=np.float64)
    tabulated = np.asarray(table["policy"], dtype=np.float64)
    n_optimal = np.asarray(table["n_optimal_actions"])
    # The granularity of the table's own action set, read from the grid it was solved on rather
    # than inferred from the weights that happen to appear in the policy.
    action_grid = np.asarray(table["action_grid"], dtype=np.float64)
    step = float(np.min(action_grid[action_grid > 0.0])) if (action_grid > 0.0).any() else 0.0

    path = rollout(scenarios, policy, problem=problem, cost_rate=0.0)["wealth_path"]
    horizon = min(path.shape[1] - 1, tabulated.shape[1])

    distances: list[np.ndarray] = []
    degenerate = 0
    total = 0
    for t in range(horizon):
        relative = path[:, t] / problem.goal_usd
        node = np.clip(
            np.rint((relative - grid[0]) / (grid[1] - grid[0])).astype(np.intp), 0, grid.size - 1
        )
        remaining = problem.monthly_contribution_usd * (problem.horizon_months - t)
        observations = np.stack(
            [
                relative,
                np.full(relative.size, t / problem.horizon_months),
                np.full(relative.size, remaining / problem.goal_usd),
            ],
            axis=-1,
        ).astype(np.float32)

        unique = n_optimal[node, t] == 1
        total += relative.size
        degenerate += int((~unique).sum())
        if not unique.any():
            continue
        chosen = project_to_simplex(policy(observations, t))
        # Half the L1 distance: zero when the portfolios agree, one when they share nothing.
        distances.append(
            0.5 * np.abs(chosen[unique] - tabulated[node[unique], t]).sum(axis=-1)
        )

    if not distances:
        return {
            "comparable_states": 0,
            "degenerate_share": 1.0,
            "mean_distance": float("nan"),
            "median_distance": float("nan"),
            "share_beyond_one_action_step": float("nan"),
        }
    pooled = np.concatenate(distances)
    return {
        "comparable_states": int(pooled.size),
        "degenerate_share": float(degenerate / max(total, 1)),
        "mean_distance": float(pooled.mean()),
        "median_distance": float(np.median(pooled)),
        "share_beyond_one_action_step": float((pooled > step).mean()) if step > 0.0 else float("nan"),
    }


def paired_proportions(success_a: np.ndarray, success_b: np.ndarray) -> dict[str, float]:
    """Test whether two policies reach the goal equally often, on the same scenarios.

    The frozen test set exists so that every policy meets the same hundred and twenty months in the
    same order, which makes the comparison paired. Treating the two success rates as independent
    samples would throw that away and widen the interval for no reason: most of the variation is
    the scenario, not the policy, and it cancels.

    What remains after it cancels are the scenarios where exactly one of the two succeeds. Under
    the null that the policies are equally good, each such scenario is a fair coin, which is
    McNemar's test; the p-value here is the exact binomial one rather than its chi-squared
    approximation, because the discordant counts can be small in the tail configurations.

    Args:
        success_a: ``(n,)`` boolean, whether policy A reached the goal in each scenario.
        success_b: ``(n,)`` boolean, likewise for policy B.

    Returns:
        The two discordant counts, the paired difference in proportions with its standard error,
        and the exact two-sided p-value.
    """
    success_a = np.asarray(success_a, dtype=bool)
    success_b = np.asarray(success_b, dtype=bool)
    if success_a.shape != success_b.shape:
        raise ValueError(
            "the two policies must be scored on the same scenarios for the comparison to be paired, "
            f"got {success_a.shape} and {success_b.shape}"
        )

    n = success_a.size
    only_a = int(np.sum(success_a & ~success_b))
    only_b = int(np.sum(~success_a & success_b))
    discordant = only_a + only_b
    difference = (only_a - only_b) / n
    # Variance of the paired difference: the concordant scenarios contribute nothing.
    standard_error = float(np.sqrt(discordant) / n) if discordant else 0.0
    p_value = float(binomtest(only_a, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "n_scenarios": int(n),
        "a_only": only_a,
        "b_only": only_b,
        "discordant": discordant,
        "difference": float(difference),
        "standard_error": standard_error,
        "p_value": p_value,
    }


AGENT_PREFIX = "agent_"
TABLE_POLICY = "dp_table"

H2_REFERENCE = "glide_path"
"""Comparator the second hypothesis is contested against.

The best heuristic by probability of reaching the goal in every configuration evaluated, which is
what makes it the honest opponent: testing against a comparator the agent already dominates would
measure the comparator. It is also the strategy the question is practically about, since a glide
path is what a target-date fund sells."""


def agent_comparisons(
    policies: dict[str, Policy],
    success: dict[str, np.ndarray],
    scenarios: np.ndarray,
    *,
    problem: ProblemConfig,
    directory: Path = DP_DIRECTORY,
    configuration: str,
) -> list[dict[str, Any]]:
    """Per seed: whether the agent does better, and how far its policy sits from the table.

    What the metrics table cannot show. That table reports each policy on its own, and two
    proportions read off it are not a test: the frozen test set exists so that every policy meets
    the same scenarios in the same order, and throwing the pairing away would widen the comparison
    for no reason. Each agent is therefore tested against two references on the same scenarios.

    Against ``glide_path``, the paired difference of proportions of H2 (thesis 4.2.4). Against
    ``dp_table``, the same test plus the discrepancy index over the states the agent actually
    visits, which is H1. The second is available only where the solver has produced a table; the
    first is available wherever the comparators were scored, so a configuration without a table
    still gets the hypothesis it was run for.

    Args:
        policies: Every policy scored, keyed by name; only names starting with ``agent_`` are
            compared.
        success: Per-scenario success flags of each policy, from ``evaluate_policies``.
        scenarios: ``(n, T, N)`` frozen tensor of monthly simple returns.
        problem: Base case, needed by the discrepancy index.
        directory: Where the solver writes its tables.
        configuration: Name of the frozen recipe, which names the table file.

    Returns:
        One row per agent that has at least one reference to be compared against, and an empty
        list otherwise: ``--policies agent`` scores neither reference, and a row carrying no
        comparison is not a result.
    """
    path = directory / f"{configuration}.npz"
    table = np.load(path) if path.is_file() and TABLE_POLICY in success else None

    rows: list[dict[str, Any]] = []
    for name in policies:
        if not name.startswith(AGENT_PREFIX):
            continue
        paired: dict[str, Any] = {}
        for reference in (H2_REFERENCE, TABLE_POLICY):
            if reference not in success:
                continue
            paired[reference] = {
                "reference_probability": float(success[reference].mean()),
                **paired_proportions(success[name], success[reference]),
            }
        if not paired:
            continue
        row: dict[str, Any] = {
            "policy": name,
            "probability_of_reaching_goal": float(success[name].mean()),
            "paired": paired,
        }
        if table is not None:
            row["discrepancy"] = discrepancy_index(table, policies[name], scenarios, problem)
        rows.append(row)
    return rows


def _print_agent_comparisons(rows: list[dict[str, Any]], configuration: str) -> None:
    """Print the paired tests and the distance between policy maps, per seed."""
    if not rows:
        return

    if any(H2_REFERENCE in row["paired"] for row in rows):
        print(f"\nH2, each seed against {H2_REFERENCE}, paired on the same scenarios")
        header = (
            f"{'policy':18} {'P(agent)':>9} {'P(ref)':>9} {'diff':>8} {'a_only':>7} {'b_only':>7} "
            f"{'p':>11}"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            entry = row["paired"].get(H2_REFERENCE)
            if entry is None:
                continue
            print(
                f"{row['policy']:18} {row['probability_of_reaching_goal']:9.4f} "
                f"{entry['reference_probability']:9.4f} {entry['difference']:+8.4f} "
                f"{entry['a_only']:7,} {entry['b_only']:7,} {entry['p_value']:11.3e}"
            )

    if not any("discrepancy" in row for row in rows):
        print(
            f"\nno table at {DP_DIRECTORY / f'{configuration}.npz'}: the discrepancy index and the "
            f"paired test against {TABLE_POLICY} are not available for this configuration."
        )
        return

    print(f"\nagainst {TABLE_POLICY}, per seed: paired outcomes and distance between policy maps")
    header = (
        f"{'policy':18} {'P(agent)':>9} {'P(table)':>9} {'diff':>8} {'a_only':>7} {'b_only':>7} "
        f"{'p':>11} {'mean dist':>10} {'median':>8} {'>1 step':>8} {'states':>11} {'degenerate':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        entry = row["paired"][TABLE_POLICY]
        distance = row["discrepancy"]
        print(
            f"{row['policy']:18} {row['probability_of_reaching_goal']:9.4f} "
            f"{entry['reference_probability']:9.4f} {entry['difference']:+8.4f} "
            f"{entry['a_only']:7,} {entry['b_only']:7,} {entry['p_value']:11.3e} "
            f"{distance['mean_distance']:10.4f} {distance['median_distance']:8.4f} "
            f"{distance['share_beyond_one_action_step']:8.4f} "
            f"{distance['comparable_states']:11,} {distance['degenerate_share']:11.4f}"
        )
    print(
        "Distance is half the L1 distance between portfolios, so 0 is agreement and 1 shares "
        "nothing. McNemar is exact and two-sided; a_only counts scenarios the agent reaches the "
        "goal in and the reference does not."
    )


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
    """Write the results table as CSV, one row per policy.

    Args:
        table: Mapping from policy name to its metrics; keys outside ``COLUMNS`` are dropped.
        path: Destination CSV file; parent directories are created.

    Returns:
        ``path``.
    """
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
# The third hypothesis: where the advantage sits
# ======================================================================================
N_TERCILES = 3
MIDPOINT_FRACTION = 0.5
"""Terciles are formed on relative wealth at the middle of the horizon (thesis 4.2.4)."""


def terciles(
    policies: dict[str, Policy],
    scenarios: np.ndarray,
    *,
    problem: ProblemConfig | None = None,
    cost_rate: float | None = None,
    reference: str = "glide_path",
    agent: str | None = None,
) -> dict[str, Any]:
    """Split the test set by progress at mid-horizon and report success within each third.

    The third hypothesis is not about whether the agent wins but about where. If an advantage
    exists it should sit with the trajectories that are behind the pace at the midpoint, because
    that is the only situation in which looking at accumulated capital can tell you something the
    calendar cannot: a glide path descends on the same schedule whether the investor is ahead or
    behind.

    **Each policy is split by its own trajectory**, not by a common one. The tercile is a statement
    about the state that policy reached, and sorting every policy by, say, the glide path's wealth
    would be asking how each does in situations some other strategy created.

    The consequence is that the paired test inside a tercile is paired on scenarios, not on a
    shared bucket: the two policies being compared may have placed a given scenario in different
    thirds. The comparison is therefore run on the scenarios the two agree to put in the same
    third, and the number dropped is reported, because a tercile comparison that quietly conditions
    on different samples is not paired at all.

    Args:
        policies: Mapping from name to callable.
        scenarios: ``(n, T, N)`` frozen tensor.
        problem: Base case; read from configuration when omitted.
        cost_rate: Overrides ``lambda``.
        reference: Comparator the agents are tested against inside each third.
        agent: Restrict the paired test to this policy; every name starting with ``agent`` when
            omitted. The protocol trains five seeds and the spread across them is part of the
            result, so all five are tested by default rather than whichever came first.

    Returns:
        A JSON-serialisable document with the per-tercile success rates and the paired tests.
    """
    problem = problem or load_problem()
    midpoint = int(round(problem.horizon_months * MIDPOINT_FRACTION))

    results = {
        name: rollout(scenarios, policy, problem=problem, cost_rate=cost_rate)
        for name, policy in policies.items()
    }
    # Relative wealth at the midpoint, and the third of the distribution it falls in. Edges are
    # taken from the policy's own distribution, so each third holds a third of the scenarios.
    buckets: dict[str, np.ndarray] = {}
    edges: dict[str, list[float]] = {}
    for name, result in results.items():
        middle = result["wealth_path"][:, midpoint] / problem.goal_usd
        cut = np.quantile(middle, [1.0 / N_TERCILES, 2.0 / N_TERCILES])
        buckets[name] = np.digitize(middle, cut)
        edges[name] = [float(value) for value in cut]

    success = {name: result["terminal_wealth"] >= problem.goal_usd for name, result in results.items()}
    names = [
        "behind",  # lowest third of relative wealth at the midpoint
        "on_pace",
        "ahead",
    ]

    document: dict[str, Any] = {
        "midpoint_month": midpoint,
        "n_scenarios": int(scenarios.shape[0]),
        "tercile_names": names,
        "policies": {},
        "paired": {},
    }
    for name in policies:
        document["policies"][name] = {
            "midpoint_edges_over_goal": edges[name],
            "terciles": {
                names[index]: {
                    "n_scenarios": int((buckets[name] == index).sum()),
                    "probability_of_reaching_goal": float(success[name][buckets[name] == index].mean()),
                    "median_wealth_at_midpoint_over_goal": float(
                        np.median(results[name]["wealth_path"][buckets[name] == index, midpoint])
                        / problem.goal_usd
                    ),
                }
                for index in range(N_TERCILES)
            },
        }

    tested = [agent] if agent else [name for name in policies if name.startswith(AGENT_PREFIX)]
    if not tested or reference not in policies:
        document["paired"] = {
            "skipped": f"needs both an agent and {reference!r} among the policies scored"
        }
        return document

    document["paired"] = {"reference": reference, "agents": {}}
    for name in tested:
        per_tercile: dict[str, Any] = {}
        for index, label in enumerate(names):
            shared = (buckets[name] == index) & (buckets[reference] == index)
            only_one = ((buckets[name] == index) | (buckets[reference] == index)).sum() - shared.sum()
            test = paired_proportions(success[name][shared], success[reference][shared])
            per_tercile[label] = {
                **test,
                "scenarios_in_exactly_one_tercile": int(only_one),
                "agent_probability": float(success[name][shared].mean()) if shared.any() else float("nan"),
                "reference_probability": (
                    float(success[reference][shared].mean()) if shared.any() else float("nan")
                ),
            }
        document["paired"]["agents"][name] = {"terciles": per_tercile}
    return document


# ======================================================================================
# The post-freeze guard
# ======================================================================================
REFERENCE_TEST_SET = "B_L12"
"""Configuration the reference run is measured on: the main engine, where H2 is contested."""

REFERENCE_PATH = RESULTS_DIRECTORY / "reference_run.json"


def reference_digest(table: dict[str, dict[str, float]]) -> str:
    """A digest of a results table, insensitive to anything but the numbers.

    Serialised with sorted keys and full float precision, so the digest changes when and only when
    a figure changes. Timestamps, paths and the order policies happen to be scored in are excluded
    deliberately: they move on every run and would make the guard useless.

    Args:
        table: Mapping from policy name to its metrics.

    Returns:
        Hex SHA-256 of the canonical JSON serialisation.
    """
    canonical = {name: {key: repr(value) for key, value in sorted(row.items())} for name, row in sorted(table.items())}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


def reference_run(
    *, seed: int, n_bootstrap: int = N_BOOTSTRAP, path: Path = REFERENCE_PATH, argv: list[str] | None = None
) -> dict[str, Any]:
    """Re-score the comparators on the main configuration and compare against the recorded digest.

    This is the guard for the period after the freeze, when changes are supposed to be
    non-functional: a rename, a docstring, a faster loop. Any of those can change a number by
    accident, and nothing else in the suite would notice, because the unit tests check properties
    rather than the specific figures the memoir quotes. Running this before and after such a change
    turns "nothing should have moved" into a statement that can be checked in one command.

    It is not a test of correctness. It only says whether today's figures are yesterday's.

    Args:
        seed: Bootstrap seed, so the intervals are reproducible too.
        n_bootstrap: Resamples behind each interval.
        path: Where the reference is recorded.
        argv: Command line recorded in the document.

    Returns:
        A document with the digest, the verdict and the table.
    """
    problem = load_problem()
    recipe = load_test_set()
    columns, matrix = load_matrix()
    scenarios, provenance = frozen_tensor(REFERENCE_TEST_SET, recipe, matrix=matrix)
    config = load_baselines(columns=columns)
    policies = build_baselines(
        config, statistics=load_matrix_metadata()["statistics"], horizon=int(scenarios.shape[1])
    )
    table = evaluate_policies(policies, scenarios, problem=problem, seed=seed, n_bootstrap=n_bootstrap)
    digest = reference_digest(table)

    recorded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if recorded is None:
        verdict = "recorded"
    elif recorded.get("digest") == digest:
        verdict = "match"
    else:
        verdict = "MISMATCH"

    document = {
        "test_set": REFERENCE_TEST_SET,
        "cost_bp": problem.cost_bp,
        "seed": seed,
        "bootstrap_resamples": n_bootstrap,
        "digest": digest,
        "verdict": verdict,
        "previous_digest": None if recorded is None else recorded.get("digest"),
        "table": table,
        "tensor_sha256": provenance["tensor_sha256"],
        "config_sha256": {
            "baselines": sha256_of_file(config_path("baselines")),
            "problem": sha256_of_file(config_path("problem")),
            "test_set": sha256_of_file(config_path("test_set")),
        },
        "command_line": argv if argv is not None else sys.argv,
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
    }
    if verdict != "MISMATCH":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


# ======================================================================================
# Command line
# ======================================================================================
def run(
    test_set: str,
    cost_bp: float | None,
    *,
    seed: int,
    which: str = "baselines",
    train_cost_bp: float | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    argv: list[str] | None = None,
    results_directory: Path = RESULTS_DIRECTORY,
    note: str | None = None,
) -> tuple[dict[str, dict[str, float]], Path, dict[str, Any], list[dict[str, Any]], Costs]:
    """Evaluate a set of policies on one frozen configuration and write the table.

    Args:
        test_set: Configuration key of the frozen recipe, for example ``B_L12``.
        cost_bp: Transaction cost charged in this evaluation; the base case when omitted.
        which: ``baselines``, ``agent``, ``dp`` or ``all``.
        train_cost_bp: Cost the agents were trained under, selecting the run directory; the
            evaluation cost when omitted, because an agent is scored on the problem it was trained
            for unless the caller says otherwise.
        seed: Seed of the bootstrap stream.
        n_bootstrap: Resamples behind each interval.
        argv: Command line recorded in the side-car.
        results_directory: Where the table is written.
        note: Recorded verbatim in the side-car, for a file whose history needs one.

    Returns:
        Tuple ``(table, path, described, comparisons, costs)``.
    """
    problem = load_problem()
    recipe = load_test_set()
    columns, matrix = load_matrix()
    scenarios, provenance = frozen_tensor(test_set, recipe, matrix=matrix)

    costs = resolve_costs(cost_bp, train_cost_bp, problem)

    config = load_baselines(columns=columns)
    statistics = load_matrix_metadata()["statistics"]
    horizon = int(scenarios.shape[1])

    policies: dict[str, Policy] = {}
    if which in ("baselines", "all"):
        policies.update(build_baselines(config, statistics=statistics, horizon=horizon))
    if which in ("dp", "all"):
        policies.update(table_policy(test_set))
    if which in ("agent", "all"):
        policies.update(trained_policies(test_set, train_cost_bp=costs.train_bp))
    if not policies:
        raise ValueError(
            f"no policies to score for --policies {which} on {test_set}: no table at "
            f"{DP_DIRECTORY / f'{test_set}.npz'}"
        )

    success: dict[str, np.ndarray] = {}
    table = evaluate_policies(
        policies,
        scenarios,
        problem=problem,
        cost_rate=costs.rate,
        seed=seed,
        n_bootstrap=n_bootstrap,
        success_out=success,
    )
    comparisons = (
        agent_comparisons(policies, success, scenarios, problem=problem, configuration=test_set)
        if which in ("agent", "all")
        else []
    )

    described = describe_baselines(config, statistics=statistics, horizon=horizon)
    name = "heuristics" if which == "baselines" else which
    path = results_directory / f"{name}_{test_set}_{costs.suffix}.csv"
    write_table(table, path)
    write_sidecar(
        path,
        {
            "policies": which,
            "policy_names": list(policies),
            "cost_bp": costs.evaluate_bp,
            "evaluation_cost_bp": costs.evaluate_bp,
            "training_cost_bp": costs.train_bp,
            "costs_differ": costs.differ,
            "agent_run_directory": (
                str(agent_run_directory(test_set, costs.train_bp)) if which in ("agent", "all") else None
            ),
            "agent_comparisons": comparisons,
            "cost_rate": costs.rate,
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
            **({"note": note} if note else {}),
            **provenance,
        },
        argv=argv,
    )
    return table, path, described, comparisons, costs


def run_terciles(
    test_set: str,
    cost_bp: float | None,
    *,
    which: str = "all",
    train_cost_bp: float | None = None,
    argv: list[str] | None = None,
    results_directory: Path = RESULTS_DIRECTORY,
) -> tuple[dict[str, Any], Path, Costs]:
    """Build the tercile split for one configuration and write it with its side-car.

    Takes both costs for the same reason the metrics table does: the third hypothesis is about
    where an agent's advantage sits, and which agent that is depends on the training cost.

    Args:
        test_set: Name of the frozen recipe in ``config/test_set.yml``.
        cost_bp: Evaluation cost in basis points; ``cost_bp`` of ``config/problem.yml`` when ``None``.
        which: ``"baselines"``, ``"dp"``, ``"agent"`` or ``"all"``.
        train_cost_bp: Training cost in basis points; the evaluation cost when ``None``.
        argv: Command line recorded in the side-car.
        results_directory: Where ``terciles_<test_set>_<suffix>.json`` is written.

    Returns:
        The tercile document, the path written and the two costs.

    Raises:
        ValueError: If ``which`` selects no policy.
    """
    problem = load_problem()
    recipe = load_test_set()
    columns, matrix = load_matrix()
    scenarios, provenance = frozen_tensor(test_set, recipe, matrix=matrix)

    costs = resolve_costs(cost_bp, train_cost_bp, problem)

    config = load_baselines(columns=columns)
    horizon = int(scenarios.shape[1])
    policies: dict[str, Policy] = {}
    if which in ("baselines", "all"):
        policies.update(
            build_baselines(config, statistics=load_matrix_metadata()["statistics"], horizon=horizon)
        )
    if which in ("dp", "all"):
        policies.update(table_policy(test_set))
    if which in ("agent", "all"):
        policies.update(trained_policies(test_set, train_cost_bp=costs.train_bp))
    if not policies:
        raise ValueError(f"no policies to score for --policies {which} on {test_set}")

    document = terciles(policies, scenarios, problem=problem, cost_rate=costs.rate)
    document["test_set"] = test_set
    document["cost_bp"] = costs.evaluate_bp
    document["evaluation_cost_bp"] = costs.evaluate_bp
    document["training_cost_bp"] = costs.train_bp

    path = results_directory / f"terciles_{test_set}_{costs.suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    write_sidecar(
        path,
        {
            "policies": which,
            "policy_names": list(policies),
            "cost_bp": costs.evaluate_bp,
            "evaluation_cost_bp": costs.evaluate_bp,
            "training_cost_bp": costs.train_bp,
            "costs_differ": costs.differ,
            "agent_run_directory": (
                str(agent_run_directory(test_set, costs.train_bp)) if which in ("agent", "all") else None
            ),
            "midpoint_month": document["midpoint_month"],
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
    return document, path, costs


def _print_terciles(result: tuple[dict[str, Any], Path, Costs]) -> int:
    """Print the per-tercile success rates and the paired tests inside each third."""
    document, path, costs = result
    names = document["tercile_names"]
    print(
        f"terciles on {document['test_set']}, {costs.describe()}, "
        f"split by W/G at month {document['midpoint_month']} of each policy's own trajectory\n"
    )
    header = f"{'policy':18} " + " ".join(f"{name:>12}" for name in names)
    print(header)
    print("-" * len(header))
    for name, entry in document["policies"].items():
        row = f"{name:18} "
        row += " ".join(
            f"{entry['terciles'][tercile]['probability_of_reaching_goal']:>12.4f}" for tercile in names
        )
        print(row)

    paired = document["paired"]
    if "skipped" in paired:
        print(f"\npaired test skipped: {paired['skipped']}")
    else:
        print(f"\npaired against {paired['reference']} inside each third, every seed:")
        header = (
            f"  {'policy':18} {'tercile':10} {'n':>7} {'agent':>8} {'ref':>8} {'diff':>9} "
            f"{'p':>10} {'dropped':>8}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, entry in paired["agents"].items():
            for tercile in names:
                row = entry["terciles"][tercile]
                print(
                    f"  {name:18} {tercile:10} {row['n_scenarios']:>7} {row['agent_probability']:>8.4f} "
                    f"{row['reference_probability']:>8.4f} {row['difference']:>+9.4f} "
                    f"{row['p_value']:>10.3g} {row['scenarios_in_exactly_one_tercile']:>8}"
                )
        print(
            "\n  dropped: scenarios the two policies place in different thirds. They are excluded so\n"
            "  the comparison stays paired on the same scenarios rather than on two different samples."
        )
    print(f"\nwrote {path}")
    print(FREEZE_NOTICE)
    return 0


def _print_reference_run(document: dict[str, Any]) -> int:
    """Print the verdict of the post-freeze guard."""
    print(f"reference run on {document['test_set']} at {document['cost_bp']:g} bp, seed {document['seed']}")
    print(f"digest {document['digest']}")
    if document["verdict"] == "recorded":
        print(f"\nrecorded for the first time in {REFERENCE_PATH}. A later run must reproduce it.")
        return 0
    if document["verdict"] == "match":
        print("\nMATCH: every figure of the heuristics table is unchanged.")
        return 0
    print(f"previous {document['previous_digest']}")
    print(
        "\nMISMATCH: a figure of the heuristics table moved. If the change that caused it was meant\n"
        "to be non-functional, it was not. The recorded reference has been left untouched so the\n"
        "comparison can be repeated after the cause is found.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: ``python -m portfolio_manager.evaluate``.

    Args:
        argv: Arguments to parse; ``sys.argv`` when ``None``.

    Returns:
        Process exit code: 0 on success or MATCH, 1 on a failure or a reference-run MISMATCH.
    """
    parser = argparse.ArgumentParser(description="Evaluation protocol: score policies on the frozen test set.")
    parser.add_argument(
        "--policies",
        choices=("baselines", "agent", "dp", "all"),
        default="baselines",
        help="comparators, the trained seeds, the tabulated optimum, or every one of them",
    )
    parser.add_argument("--test-set", default="B_L12", help="configuration of the frozen recipe, e.g. B_L12")
    parser.add_argument(
        "--reference-run",
        action="store_true",
        help="re-score the comparators on the main configuration and compare against the recorded digest",
    )
    parser.add_argument(
        "--terciles",
        action="store_true",
        help="split the test set by progress at mid-horizon and report success within each third (H3)",
    )
    parser.add_argument(
        "--cost-bp", type=float, default=None, help="transaction cost charged here; the base case when omitted"
    )
    parser.add_argument(
        "--train-cost-bp",
        type=float,
        default=None,
        help=(
            "cost the agents were trained under, selecting the run directory; --cost-bp when "
            "omitted, because an agent is evaluated on the problem it was trained for. The sweep "
            "passes it explicitly, e.g. --train-cost-bp 10 --cost-bp 25"
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="bootstrap seed; the validation band when omitted")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="resamples behind each interval")
    parser.add_argument("--note", default=None, help="recorded verbatim in the side-car")
    args = parser.parse_args(argv)

    try:
        problem = load_problem()
        seed = args.seed if args.seed is not None else int(load_test_set()["seeds"]["validation"]["min"])
        if args.reference_run:
            return _print_reference_run(
                reference_run(
                    seed=seed,
                    n_bootstrap=args.n_bootstrap,
                    argv=argv if argv is not None else sys.argv,
                )
            )
        if args.terciles:
            return _print_terciles(
                run_terciles(
                    args.test_set,
                    args.cost_bp,
                    which=args.policies,
                    train_cost_bp=args.train_cost_bp,
                    argv=argv if argv is not None else sys.argv,
                )
            )
        table, path, described, comparisons, costs = run(
            args.test_set,
            args.cost_bp,
            seed=seed,
            which=args.policies,
            train_cost_bp=args.train_cost_bp,
            n_bootstrap=args.n_bootstrap,
            argv=argv if argv is not None else sys.argv,
            note=args.note,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"test set {args.test_set}, {table[next(iter(table))]['n_scenarios']:,} scenarios, seed {seed}")
    print(f"cost                 {costs.describe()}")
    if args.policies in ("agent", "all"):
        print(f"agents from          {agent_run_directory(args.test_set, costs.train_bp)}")
    print()
    _print_parameters(described, costs.evaluate_bp)
    _print_table(table, problem.goal_usd)
    if args.policies in ("agent", "all"):
        _print_agent_comparisons(comparisons, args.test_set)
    print(f"\nwrote {path}")
    print(FREEZE_NOTICE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
