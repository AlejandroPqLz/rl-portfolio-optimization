"""Dynamic programming solver: the optimal policy by backward induction, thesis subsection 4.3.6.

Contract
--------
``solve(...)`` returns the pair of tables that solve Equations ``eq:bellman`` and
``eq:bellman-politica``: a value table of shape ``(n_wealth, T + 1)`` and a policy table of shape
``(n_wealth, T, N)``. Both are saved together with the grids they are defined on, because the
discrepancy index of the first hypothesis has to read the tabulated policy at the states the
learned one visits.

What the table is, and is not
-----------------------------
It is the exact optimum of the problem it was built for, up to the discretisation declared in
``config/dp_solver.yml`` and bounded by the stability checks. That is a stronger claim than
anything a learned policy can make, and it is why the first hypothesis can be stated as a
comparison against a reference rather than as a comparison between two approximations.

The claim holds exactly where the state of Equation ``eq:estado`` is Markovian: under the
parametric generator, whose months are independent, and at zero transaction cost. With costs the
transition depends on the arrival weights, which are not in the state (D56). Applied to resampled
scenarios or with costs, the table is a strategy computed for a neighbouring problem and is
reported under that condition.

The transition is imported, not reimplemented: :func:`portfolio_manager.env.step_wealth` is called
on the whole grid at once (D74). There is one piece of transition arithmetic in the project, so
the validation against Browne debugs one piece.

Decisions involved
------------------
D44 (the validation runs in a separate configuration with the indicator objective), D56 (the table
is computed at zero cost), D74 (the shared transition).

Command line
------------
    uv run python -m portfolio_manager.dp_solver --config A_gbm
    uv run python -m portfolio_manager.dp_solver --config A_gbm --stability
"""

# Imports
# =====================================================================
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from portfolio_manager.baselines import dp_table_policy
from portfolio_manager.env import (
    OBJECTIVES,
    ProblemConfig,
    load_problem,
    rollout,
    step_wealth,
    terminal_reward,
)
from portfolio_manager.environment_fingerprint import collect_fingerprint
from portfolio_manager.generator import (
    CHECKS_DIRECTORY,
    GbmParams,
    config_path,
    frozen_tensor,
    git_commit,
    load_gbm_params,
    load_matrix,
    load_test_set,
    write_sidecar,
)

# Constants
# =====================================================================
DP_DIRECTORY = Path("results/dp")

BASIS_POINT = 1e-4

STABILITY_SCENARIOS = 2000
"""Test-set scenarios rolled out to find the nodes the stability check is measured on. The frozen
tensor holds twenty thousand; the visited set of (node, month) pairs saturates long before that,
since there are only ``n_wealth * T`` of them."""

MAX_CHUNK_ELEMENTS = 32_000_000
"""Ceiling on the intermediate array of one batch, ``n_wealth * chunk * n_quad ** k``, per worker.
The batch size in the configuration is capped by this, because the stability resolution multiplies
that product by nearly eight and every worker holds a batch of its own.

Deliberately **not** a function of the worker count. The batch size changes how the contraction
over the quadrature axis is blocked, and therefore its last bits: solving with a batch of 64
against one of 31 moves the value function by about 2e-15. That is immaterial next to any
threshold here, but a table whose digest depended on how many cores the machine had would not be
reproducible at all, so the batching is fixed by the configuration and the machine only decides
how many batches run at once."""

TIE_BREAKS = ("min_variance", "first")
"""How a tie between equally optimal portfolios is resolved. See ``config/dp_solver.yml``."""

GOAL_UNITS = 1.0
"""The grid is normalised by the goal, so the goal is one and the terminal reward reads directly."""


# ======================================================================================
# Configuration
# ======================================================================================
@dataclass(frozen=True)
class SolverConfig:
    """The three discretisations and what is being solved.

    Attributes:
        n_wealth: Nodes on the normalised wealth axis.
        wealth_max_over_goal: Upper bound of that axis, in goal units.
        action_step: Granularity of the candidate portfolios.
        n_quad: Gauss-Hermite nodes per non-degenerate factor.
        interpolation: Only ``linear`` is implemented.
        objective: ``lpm2`` or ``indicator``.
        cost_bp: Transaction cost; zero, and not negotiable (D56).
        tie_break: How a tie between optimal portfolios is resolved.
        tie_tolerance: Values within this of the maximum count as tied.
        stability: Resolution and thresholds of the two internal checks.
        action_chunk: Candidate portfolios evaluated per batch, a memory bound only.
        n_workers: Threads over the batches of a month; ``None`` means the performance cores.
    """

    n_wealth: int
    wealth_max_over_goal: float
    action_step: float
    n_quad: int
    interpolation: str
    objective: str
    cost_bp: float
    tie_break: str
    tie_tolerance: float
    stability: dict[str, Any]
    action_chunk: int
    n_workers: int | None = None

    @property
    def workers(self) -> int:
        """Threads to run the batches of a month on."""
        return self.n_workers if self.n_workers is not None else performance_cores()

    @property
    def cost_rate(self) -> float:
        """``lambda`` as a rate per unit of turnover."""
        return self.cost_bp * BASIS_POINT

    def at_resolution(self, n_wealth: int, n_quad: int) -> "SolverConfig":
        """The same problem at a different resolution, for the stability checks."""
        return SolverConfig(**{**self.__dict__, "n_wealth": int(n_wealth), "n_quad": int(n_quad)})


def load_solver_config(path: Path | None = None) -> SolverConfig:
    """Read ``config/dp_solver.yml``."""
    path = path or config_path("dp_solver")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: no discretisation has a default in code")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    objective = str(raw["objective"])
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; this module implements {OBJECTIVES}")
    if str(raw["interpolation"]) != "linear":
        raise ValueError(f"only linear interpolation is implemented, the file declares {raw['interpolation']!r}")
    if float(raw["cost_bp"]) != 0.0:
        raise ValueError(
            "the table must be computed at zero transaction cost (D56): with costs the transition "
            "depends on the arrival weights, which are not part of the state, so the table would no "
            "longer be the optimum of any problem defined on this grid"
        )
    tie_break = str(raw["tie_break"])
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"unknown tie_break {tie_break!r}; this module implements {TIE_BREAKS}")
    return SolverConfig(
        n_wealth=int(raw["n_wealth"]),
        wealth_max_over_goal=float(raw["wealth_max_over_goal"]),
        action_step=float(raw["action_step"]),
        n_quad=int(raw["n_quad"]),
        interpolation=str(raw["interpolation"]),
        objective=objective,
        cost_bp=float(raw["cost_bp"]),
        tie_break=tie_break,
        tie_tolerance=float(raw["tie_tolerance"]),
        stability=dict(raw["stability"]),
        action_chunk=int(raw["action_chunk"]),
        n_workers=None if raw.get("n_workers") is None else int(raw["n_workers"]),
    )


def performance_cores() -> int:
    """Number of performance cores, or the best available estimate.

    Above this the induction stops gaining: the efficiency cores of a heterogeneous processor are
    much slower at this arithmetic and contend for the same memory bandwidth, so adding them moves
    the wall clock very little and sometimes backwards.
    """
    if sys.platform == "darwin":
        try:
            measured = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return max(1, int(measured.stdout.strip()))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return max(1, os.cpu_count() or 1)


# ======================================================================================
# The three discretisations
# ======================================================================================
def simplex_grid(action_step: float, n_assets: int) -> np.ndarray:
    """Every simplex point whose weights are multiples of ``action_step``.

    Enumerated by stars and bars: the portfolios are the ways of splitting ``1 / action_step``
    indivisible units among the assets, so there are ``C(units + N - 1, N - 1)`` of them, which is
    1771 for twenty units and four assets.

    Args:
        action_step: Granularity, which must divide one exactly.
        n_assets: Number of columns.

    Returns:
        ``(A, N)`` array of candidate portfolios, each row on the simplex.
    """
    units = int(round(1.0 / action_step))
    if abs(units * action_step - 1.0) > 1e-12:
        raise ValueError(f"the action step must divide one exactly, got {action_step}")
    if n_assets < 1:
        raise ValueError(f"at least one asset is needed, got {n_assets}")

    points = np.empty((_n_compositions(units, n_assets), n_assets), dtype=np.float64)
    for row, cuts in enumerate(combinations(range(units + n_assets - 1), n_assets - 1)):
        previous = -1
        for column, cut in enumerate(cuts):
            points[row, column] = cut - previous - 1
            previous = cut
        points[row, -1] = units + n_assets - 2 - previous
    return points * action_step


def _n_compositions(units: int, n_assets: int) -> int:
    """Number of ways of splitting ``units`` indivisible units among ``n_assets`` columns."""
    total = 1
    for i in range(n_assets - 1):
        total = total * (units + n_assets - 1 - i) // (i + 1)
    return total


def quadrature(params: GbmParams, n_quad: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights for the monthly simple returns.

    The log gross return is ``mu + F z`` with ``z`` standard normal and ``F`` the factor of the
    covariance that the generator uses, so the expectation over ``z`` is a product of
    one-dimensional Gaussian integrals and Gauss-Hermite applies factor by factor. A factor with
    zero variance has a zero column in ``F`` and therefore no influence on the return, so it
    receives no nodes: that is what keeps the Browne configuration tractable, where three riskless
    assets would otherwise multiply the tensor product by ``n_quad`` cubed for nothing.

    Args:
        params: Mean and covariance of the monthly log gross returns.
        n_quad: Nodes per non-degenerate factor.

    Returns:
        Tuple ``(returns, weights)`` with returns ``(Q, N)`` simple and weights ``(Q,)`` summing to
        one, where ``Q = n_quad ** k`` and ``k`` is the number of non-degenerate factors.
    """
    if n_quad < 1:
        raise ValueError(f"at least one quadrature node is needed, got {n_quad}")
    factor = params.factor()
    active = np.flatnonzero(np.abs(factor).sum(axis=0) > 0.0)

    # Gauss-Hermite integrates against exp(-x^2); the standard normal needs z = sqrt(2) x and the
    # weights divided by sqrt(pi), after which they sum to one.
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)

    if active.size == 0:
        grids, weight = np.zeros((1, 0)), np.ones(1)
    else:
        mesh = np.meshgrid(*([nodes] * active.size), indexing="ij")
        grids = np.stack([axis.ravel() for axis in mesh], axis=-1)
        weight_mesh = np.meshgrid(*([weights] * active.size), indexing="ij")
        weight = np.prod(np.stack([axis.ravel() for axis in weight_mesh], axis=-1), axis=-1)

    standard = np.zeros((grids.shape[0], params.n_assets), dtype=np.float64)
    standard[:, active] = grids
    # Normalised so the weights sum to one exactly. A Gauss-Hermite rule is exact for constants by
    # construction, but the tensor product of the tabulated weights misses by a few parts in 1e16,
    # and the induction multiplies that error in a hundred and twenty times. Without this the value
    # drifts outside the [-1, 0] the reward is bounded to, which is a property worth keeping exact.
    return np.expm1(params.mu + standard @ factor.T), weight / weight.sum()


def wealth_grid(config: SolverConfig) -> np.ndarray:
    """Equispaced nodes of normalised wealth, from zero to the declared bound."""
    return np.linspace(0.0, config.wealth_max_over_goal, config.n_wealth, dtype=np.float64)


# ======================================================================================
# Backward induction
# ======================================================================================
@dataclass
class Solution:
    """The tables and what is needed to read them.

    Attributes:
        value: ``(n_wealth, T + 1)`` value function, in reward units.
        policy: ``(n_wealth, T, N)`` optimal portfolio per node and month.
        wealth_grid: ``(n_wealth,)`` nodes, in goal units.
        action_grid: ``(A, N)`` candidate portfolios.
        quadrature_returns: ``(Q, N)`` simple returns of the quadrature nodes.
        quadrature_weights: ``(Q,)`` weights summing to one.
        n_optimal_actions: ``(n_wealth, T)`` count of portfolios attaining the optimum. Greater
            than one marks a state where the policy table carries no information and a consumer
            comparing policy maps must exclude it.
        max_wealth_reached: Largest normalised wealth any quadrature node produced.
        bound_exceeded: Whether that exceeded the upper node of the grid.
        seconds_elapsed: Wall-clock time of the induction.
    """

    value: np.ndarray
    policy: np.ndarray
    wealth_grid: np.ndarray
    action_grid: np.ndarray
    quadrature_returns: np.ndarray
    quadrature_weights: np.ndarray
    n_optimal_actions: np.ndarray
    max_wealth_reached: float
    bound_exceeded: bool
    seconds_elapsed: float

    def value_at(self, relative_wealth: float, t: int = 0) -> float:
        """The value at a state, interpolated exactly as the induction interpolates."""
        return float(np.interp(relative_wealth, self.wealth_grid, self.value[:, t]))

    @property
    def degenerate_fraction(self) -> float:
        """Share of states where more than one portfolio attains the optimum."""
        return float((self.n_optimal_actions > 1).mean())


def _interpolate(values: np.ndarray, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``values`` on an equispaced ``grid``, held flat beyond the bounds.

    Written out rather than delegated to ``numpy.interp`` because the grid is equispaced, so the
    node index is arithmetic rather than a search, and because this is the inner loop: it runs
    ``n_wealth * A * Q`` times per month.
    """
    step = grid[1] - grid[0]
    position = np.clip((points - grid[0]) / step, 0.0, grid.size - 1.0)
    lower = np.minimum(position.astype(np.intp), grid.size - 2)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[lower + 1] * fraction


def solve(
    config: SolverConfig | None = None,
    *,
    params: GbmParams | None = None,
    problem: ProblemConfig | None = None,
    horizon: int | None = None,
    asset_mask: np.ndarray | None = None,
    objective: str | None = None,
    contribution: float | None = None,
    progress: bool = False,
) -> Solution:
    """Solve the Bellman equation by backward induction on the grid.

    The recursion is the one of Equations ``eq:bellman`` and ``eq:bellman-politica``. At the
    horizon the value is the terminal reward of Equation ``eq:recompensa``. Stepping back one
    month, the value of a node is the best over the candidate portfolios of the expected value of
    next month's node, the expectation taken by quadrature and the value between nodes interpolated
    linearly.

    Everything is in goal units, which is legitimate because the dynamics of ``W/G`` depend only on
    ``c/G`` and ``W_0/G``: the table solves the problem for any goal, not only for sixty thousand.

    Args:
        config: Discretisation; read from ``config/dp_solver.yml`` when omitted.
        params: Generator parameters; read from ``config/gbm_params.yml`` when omitted.
        problem: Base case; read from configuration when omitted.
        horizon: Overrides ``T``.
        asset_mask: ``(N,)`` boolean selecting the columns the solver may hold. The Browne
            configuration restricts the universe to two of them.
        objective: Overrides the terminal objective.
        contribution: Overrides ``c``; the Browne configuration sets it to zero.
        progress: Whether to print the month being solved.

    Returns:
        The tables and their grids.
    """
    config = config or load_solver_config()
    problem = problem or load_problem()
    params = params or load_gbm_params()
    horizon = int(horizon if horizon is not None else problem.horizon_months)
    objective = str(objective or config.objective)
    contribution = float(
        contribution if contribution is not None else problem.monthly_contribution_usd
    ) / problem.goal_usd

    n_assets = params.n_assets
    if asset_mask is None:
        asset_mask = np.ones(n_assets, dtype=bool)
    asset_mask = np.asarray(asset_mask, dtype=bool)
    if asset_mask.shape != (n_assets,) or not asset_mask.any():
        raise ValueError(f"the asset mask must be a boolean vector of length {n_assets} with at least one asset")

    nodes = wealth_grid(config)
    # The action grid lives on the unmasked columns and is scattered into the full width, so a
    # masked asset is unreachable rather than merely unattractive.
    active_actions = simplex_grid(config.action_step, int(asset_mask.sum()))
    actions = np.zeros((active_actions.shape[0], n_assets), dtype=np.float64)
    actions[:, asset_mask] = active_actions

    returns, weights = quadrature(params, config.n_quad)

    value = np.empty((config.n_wealth, horizon + 1), dtype=np.float64)
    policy_index = np.empty((config.n_wealth, horizon), dtype=np.int32)
    n_optimal = np.empty((config.n_wealth, horizon), dtype=np.int32)
    value[:, horizon] = terminal_reward(nodes, GOAL_UNITS, objective)

    # Variance of the portfolio return under the quadrature, used only to resolve exact ties.
    portfolio_returns = actions @ returns.T
    action_variance = (portfolio_returns**2) @ weights - (portfolio_returns @ weights) ** 2
    del portfolio_returns

    # The transition is imported and called on the whole grid (D74). The wealth axis is kept out of
    # the weight and return axes so that the arrival weights, which the induction does not need,
    # stay a small array instead of growing by a factor of n_wealth.
    wealth_axis = nodes[:, None, None]
    chunk = max(1, min(config.action_chunk, MAX_CHUNK_ELEMENTS // (config.n_wealth * returns.shape[0])))
    batches = [(low, min(low + chunk, actions.shape[0])) for low in range(0, actions.shape[0], chunk)]
    workers = max(1, min(config.workers, len(batches)))

    def evaluate_batch(bounds: tuple[int, int], month: int) -> tuple[int, int, float, np.ndarray]:
        """Expected continuation value of one block of candidate portfolios.

        Independent of every other block: it reads the value function of the following month and
        writes a disjoint set of columns, which is what makes the parallel result bit-identical to
        the serial one rather than merely close to it.
        """
        low, high = bounds
        batch = actions[low:high][None, :, None, :]
        next_wealth, _, _ = step_wealth(
            wealth_axis, batch, batch, returns[None, None, :, :], contribution, config.cost_rate
        )
        return low, high, float(next_wealth.max()), _interpolate(value[:, month + 1], nodes, next_wealth) @ weights

    max_reached = 0.0
    started = time.perf_counter()
    # One pool for the whole induction. The months stay sequential because month t needs the value
    # function of month t + 1; only the batches within a month are run together.
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for t in range(horizon - 1, -1, -1):
            expected = np.empty((config.n_wealth, actions.shape[0]), dtype=np.float64)
            computed = (
                pool.map(evaluate_batch, batches, [t] * len(batches))
                if pool is not None
                else (evaluate_batch(bounds, t) for bounds in batches)
            )
            for low, high, reached, block in computed:
                expected[:, low:high] = block
                # A maximum does not depend on the order it is taken in, so this stays exact.
                max_reached = max(max_reached, reached)
            best = expected.max(axis=1)
            # Where the optimum is degenerate it is exactly degenerate, so the tolerance only
            # guards against floating-point dust. Both regions that produce it are real: at zero
            # wealth the action cannot affect the transition, and above the goal by enough that no
            # quadrature return sequence can fall short, every portfolio attains the same value.
            tied = expected >= best[:, None] - config.tie_tolerance
            n_optimal[:, t] = tied.sum(axis=1)
            if config.tie_break == "min_variance":
                policy_index[:, t] = np.where(tied, action_variance[None, :], np.inf).argmin(axis=1)
            else:
                policy_index[:, t] = expected.argmax(axis=1)
            value[:, t] = best
            if progress:
                print(f"  month {t:4d}  V(W_0) = {value[0, t]:+.6f}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    return Solution(
        value=value,
        policy=actions[policy_index],
        wealth_grid=nodes,
        action_grid=actions,
        quadrature_returns=returns,
        quadrature_weights=weights,
        n_optimal_actions=n_optimal,
        max_wealth_reached=max_reached,
        bound_exceeded=bool(max_reached > nodes[-1]),
        seconds_elapsed=time.perf_counter() - started,
    )


# ======================================================================================
# Output
# ======================================================================================
def save(
    solution: Solution,
    name: str,
    config: SolverConfig,
    problem: ProblemConfig,
    *,
    directory: Path = DP_DIRECTORY,
    argv: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the tables and the side-car that says what problem they solve.

    The grids are saved beside the tables, which section 2b requires: the discrepancy index of the
    first hypothesis reads the tabulated policy at the states the learned policy visits, and
    without the grids it could not locate them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.npz"
    np.savez_compressed(
        path,
        value=solution.value,
        policy=solution.policy,
        wealth_grid=solution.wealth_grid,
        action_grid=solution.action_grid,
        quadrature_returns=solution.quadrature_returns,
        quadrature_weights=solution.quadrature_weights,
        n_optimal_actions=solution.n_optimal_actions,
    )
    write_sidecar(
        path,
        {
            "configuration": name,
            "solver": {
                "n_wealth": config.n_wealth,
                "wealth_max_over_goal": config.wealth_max_over_goal,
                "action_step": config.action_step,
                "n_actions": int(solution.action_grid.shape[0]),
                "n_quad": config.n_quad,
                "n_quadrature_nodes": int(solution.quadrature_returns.shape[0]),
                "interpolation": config.interpolation,
                "objective": config.objective,
                "cost_bp": config.cost_bp,
                "tie_break": config.tie_break,
            },
            "problem": {
                "goal_usd": problem.goal_usd,
                "horizon_months": problem.horizon_months,
                "monthly_contribution_usd": problem.monthly_contribution_usd,
                "initial_wealth_usd": problem.initial_wealth_usd,
            },
            "value_at_initial_state": solution.value_at(problem.initial_wealth_usd / problem.goal_usd),
            "max_wealth_reached_over_goal": solution.max_wealth_reached,
            "upper_bound_exceeded": solution.bound_exceeded,
            # Quadrature nodes crossing the upper bound is expected and harmless exactly when the
            # value is already constant up there, since holding it flat is then exact rather than
            # an approximation. Recorded instead of asserted, and to a tolerance rather than to
            # the bit, because the value decays to the order of 1e-261 rather than to zero.
            "value_at_upper_bound": float(solution.value[-1, 0]),
            "value_flat_near_upper_bound": bool(
                np.ptp(solution.value[int(0.75 * solution.wealth_grid.size) :, :]) < 1e-12
            ),
            "degenerate_state_fraction": solution.degenerate_fraction,
            "seconds_elapsed": solution.seconds_elapsed,
            "status": "preliminary until the figures freeze",
            **(extra or {}),
        },
        argv=argv,
    )
    return path


# ======================================================================================
# The two internal stability checks of thesis 4.3.6
# ======================================================================================
def visited_nodes(solution: Solution, scenarios: np.ndarray, problem: ProblemConfig) -> np.ndarray:
    """The ``(node, month)`` pairs the test-set trajectories actually reach.

    Agreement between two tables in a region no trajectory visits says nothing about the policy
    being compared, so the second stability check is measured here and not over the whole grid.
    The trajectories are those of the tabulated policy itself, driven by the frozen scenarios.
    """
    path = rollout(
        scenarios,
        dp_table_policy({"wealth_grid": solution.wealth_grid, "policy": solution.policy}),
        problem=problem,
        cost_rate=0.0,
    )["wealth_path"]
    horizon = path.shape[1] - 1
    relative = path[:, :horizon] / problem.goal_usd
    grid = solution.wealth_grid
    step = grid[1] - grid[0]
    node = np.clip(np.rint((relative - grid[0]) / step).astype(np.intp), 0, grid.size - 1)
    month = np.broadcast_to(np.arange(horizon), node.shape)
    return np.unique(np.stack([node.ravel(), month.ravel()], axis=-1), axis=0)


def _expected_value(
    values: np.ndarray,
    grid: np.ndarray,
    wealth: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    weights: np.ndarray,
    contribution: float,
) -> np.ndarray:
    """Expected continuation value of holding ``actions[k]`` at wealth ``wealth[k]``."""
    next_wealth = wealth[:, None] * (1.0 + actions @ returns.T) + contribution
    return _interpolate(values, grid, next_wealth) @ weights


def policy_value_loss(
    reference: Solution,
    refined: Solution,
    node: np.ndarray,
    month: np.ndarray,
    refined_node: np.ndarray,
    contribution: float,
) -> np.ndarray:
    """What the refined table's portfolio gives up, measured in the reference problem.

    Two portfolios can differ by a long way on the simplex and be worth the same, which is exactly
    what happens when the objective is flat in the action near its optimum. Counting how often the
    weights differ then measures the flatness rather than an error, so the disagreement is also
    priced: this is the value the reference problem loses by playing the other table's choice, and
    it is zero when the disagreement is immaterial.

    Returns:
        ``(K,)`` non-negative losses, in the units of the value function.
    """
    losses = np.zeros(node.size, dtype=np.float64)
    wealth = reference.wealth_grid[node]
    for t in np.unique(month):
        selected = month == t

        def expected(actions: np.ndarray, t: int = int(t), selected: np.ndarray = selected) -> np.ndarray:
            return _expected_value(
                reference.value[:, t + 1],
                reference.wealth_grid,
                wealth[selected],
                actions,
                reference.quadrature_returns,
                reference.quadrature_weights,
                contribution,
            )

        losses[selected] = expected(reference.policy[node[selected], t]) - expected(
            refined.policy[refined_node[selected], t]
        )
    return losses


def stability_checks(
    config: SolverConfig | None = None,
    *,
    params: GbmParams | None = None,
    problem: ProblemConfig | None = None,
    scenarios: np.ndarray | None = None,
    reference: Solution | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Repeat the solve at higher resolution and compare, per thesis 4.3.6.

    Neither check proves the implementation correct. An error in the recursion would reproduce
    itself at any resolution and both checks would pass on a table that is wrong. What they bound
    is the discretisation error, which is a different and smaller claim; correctness is what the
    validation against the closed form addresses.

    Returns:
        A JSON-serialisable document with both checks and an overall verdict.
    """
    config = config or load_solver_config()
    problem = problem or load_problem()
    params = params or load_gbm_params()
    thresholds = config.stability

    if reference is None:
        reference = solve(config, params=params, problem=problem, progress=progress)
    refined = solve(
        config.at_resolution(thresholds["n_wealth"], thresholds["n_quad"]),
        params=params,
        problem=problem,
        progress=progress,
    )

    initial = problem.initial_wealth_usd / problem.goal_usd
    difference = abs(refined.value_at(initial) - reference.value_at(initial))

    if scenarios is None:
        scenarios, _ = frozen_tensor("A_gbm")
    pairs = visited_nodes(reference, scenarios[:STABILITY_SCENARIOS], problem)
    node, month = pairs[:, 0], pairs[:, 1]
    # The refined grid is finer, so the node of the reference grid is located on it by wealth.
    refined_node = np.clip(
        np.rint(
            (reference.wealth_grid[node] - refined.wealth_grid[0])
            / (refined.wealth_grid[1] - refined.wealth_grid[0])
        ).astype(np.intp),
        0,
        refined.wealth_grid.size - 1,
    )
    deviation = np.abs(reference.policy[node, month] - refined.policy[refined_node, month]).max(axis=-1)
    within_one_step = deviation <= config.action_step + 1e-9
    agreement = float(within_one_step.mean())

    # Reported separately, because the headline figure is weaker than it looks: most visited states
    # have several optimal portfolios, and there the two tables agree only because the tie-break is
    # deterministic, not because the optimum is the same. Agreement restricted to states with a
    # unique optimum is the number that says whether the discretisation moved the policy.
    unique = reference.n_optimal_actions[node, month] == 1
    unique_agreement = float(within_one_step[unique].mean()) if unique.any() else float("nan")

    loss = policy_value_loss(
        reference,
        refined,
        node,
        month,
        refined_node,
        problem.monthly_contribution_usd / problem.goal_usd,
    )

    return {
        "reference": {"n_wealth": config.n_wealth, "n_quad": config.n_quad},
        "refined": {"n_wealth": int(thresholds["n_wealth"]), "n_quad": int(thresholds["n_quad"])},
        "value": {
            "initial_state_over_goal": initial,
            "reference": reference.value_at(initial),
            "refined": refined.value_at(initial),
            "absolute_difference": difference,
            "tolerance": float(thresholds["value_tol"]),
            "passed": bool(difference < float(thresholds["value_tol"])),
            # The tolerance is absolute, in reward units. Reported beside it is what the change is
            # relative to the quantity compared, because at the initial state of the base case the
            # value is a few parts in ten thousand and an absolute tolerance of 0.005 is an order
            # of magnitude larger than the number it bounds.
            "relative_difference": float(difference / abs(reference.value_at(initial)))
            if reference.value_at(initial) != 0.0
            else float("nan"),
            "tolerance_over_value": float(thresholds["value_tol"] / abs(reference.value_at(initial)))
            if reference.value_at(initial) != 0.0
            else float("inf"),
        },
        "policy": {
            "visited_nodes": int(pairs.shape[0]),
            "scenarios": int(min(len(scenarios), STABILITY_SCENARIOS)),
            "agreement_within_one_step": agreement,
            "visited_nodes_with_unique_optimum": int(unique.sum()),
            "agreement_on_unique_optima": unique_agreement,
            "required": float(thresholds["policy_agreement"]),
            "max_deviation": float(deviation.max()),
            "passed": bool(agreement >= float(thresholds["policy_agreement"])),
            # What the disagreement costs, which is what says whether it matters. A large
            # deviation on the simplex at no cost in value means the objective is flat in the
            # action, not that the discretisation is too coarse.
            "value_loss": {
                "max": float(loss.max()),
                "mean": float(loss.mean()),
                "mean_where_disagreeing": float(loss[~within_one_step].mean())
                if (~within_one_step).any()
                else 0.0,
                "relative_to_value_tolerance": float(loss.max() / float(thresholds["value_tol"])),
            },
        },
        "upper_bound": {
            "max_wealth_reached_over_goal": reference.max_wealth_reached,
            "bound": reference.wealth_grid[-1],
            "exceeded": reference.bound_exceeded,
        },
        "seconds_elapsed": reference.seconds_elapsed + refined.seconds_elapsed,
    }


def write_stability(document: dict[str, Any], directory: Path = CHECKS_DIRECTORY) -> Path:
    """Write the stability report where subsection 4.4 reads it from."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "dp_stability.json"
    payload = {
        **document,
        "all_passed": bool(document["value"]["passed"] and document["policy"]["passed"]),
        "command_line": sys.argv,
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment_fingerprint": collect_fingerprint(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    return path


# ======================================================================================
# Command line
# ======================================================================================
def _require_parametric(name: str) -> None:
    """The table is a reference only where the state is Markovian, which is mode A."""
    spec = load_test_set()["configurations"][name]
    if str(spec["mode"]) != "A":
        raise ValueError(
            f"{name} uses generator mode {spec['mode']!r}. The tabulated optimum exists only under the "
            "parametric generator, where months are independent and the state of eq:estado is exactly "
            "Markovian. Under the block bootstrap the table is a comparator, not a reference, and is "
            "evaluated through evaluate.py rather than recomputed here."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backward induction on the wealth grid.")
    parser.add_argument("--config", required=True, help="configuration of the frozen recipe, e.g. A_gbm")
    parser.add_argument("--stability", action="store_true", help="also run the two internal checks")
    parser.add_argument("--seed", type=int, default=None, help="unused: the induction is deterministic")
    parser.add_argument("--progress", action="store_true", help="print the month being solved")
    args = parser.parse_args(argv)

    try:
        if args.seed is not None:
            raise ValueError("--seed is not accepted: backward induction draws no random numbers")
        _require_parametric(args.config)
        config = load_solver_config()
        problem = load_problem()
        params = load_gbm_params()
        columns, _ = load_matrix()

        solution = solve(config, params=params, problem=problem, progress=args.progress)
        path = save(
            solution, args.config, config, problem, argv=argv if argv is not None else sys.argv
        )

        document, stability_path = None, None
        if args.stability:
            document = stability_checks(
                config, params=params, problem=problem, reference=solution, progress=args.progress
            )
            stability_path = write_stability(document)
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    initial = problem.initial_wealth_usd / problem.goal_usd
    print(f"{args.config}: {config.n_wealth} wealth nodes, {solution.action_grid.shape[0]:,} portfolios, ")
    print(f"{solution.quadrature_returns.shape[0]} quadrature nodes, {problem.horizon_months} months")
    print(f"solved in {solution.seconds_elapsed:.1f} s\n")
    print(f"V(W_0 = {initial:.3f}) = {solution.value_at(initial):+.6f}")
    print(f"max wealth reached     {solution.max_wealth_reached:.3f} of the goal, bound {config.wealth_max_over_goal}")
    print(f"upper bound exceeded   {solution.bound_exceeded} (value there {solution.value[-1, 0]:+.1e})")
    print(
        f"degenerate states      {solution.degenerate_fraction:.1%} of (node, month) pairs have more "
        f"than one optimal portfolio;\n                       resolved by {config.tie_break}, "
        "and counted in n_optimal_actions so they can be excluded"
    )
    print("\nportfolio at W/G = 0.0 and the first month: " + _format_weights(solution.policy[0, 0], columns))
    print("portfolio at W/G = 1.0 and the last month:  " + _format_weights(
        solution.policy[int(round(1.0 / (solution.wealth_grid[1] - solution.wealth_grid[0]))), -1], columns
    ))
    print(f"\nwrote {path}")

    if document is not None:
        value_check = document["value"]
        print(
            f"\nstability  value {'PASS' if value_check['passed'] else 'FAIL'} "
            f"(|dV| = {value_check['absolute_difference']:.6f} < {value_check['tolerance']}), "
            f"but that is {value_check['relative_difference']:.0%} of the value compared,\n"
            f"                  and the tolerance is {value_check['tolerance_over_value']:.0f} times it"
        )
        policy_check = document["policy"]
        print(
            f"           policy {'PASS' if policy_check['passed'] else 'FAIL'} "
            f"({policy_check['agreement_within_one_step']:.4f} of "
            f"{policy_check['visited_nodes']:,} visited nodes agree, required "
            f"{policy_check['required']})"
        )
        print(
            f"                  {policy_check['agreement_on_unique_optima']:.4f} of the "
            f"{policy_check['visited_nodes_with_unique_optimum']:,} with a unique optimum, which is "
            "the figure that says whether the discretisation moved the policy"
        )
        print(
            f"                  the disagreement costs at most {policy_check['value_loss']['max']:.2e} "
            f"in value, {policy_check['value_loss']['relative_to_value_tolerance']:.1%} of the "
            "tolerance the value check uses"
        )
        print(f"wrote {stability_path}")
        if not (document["value"]["passed"] and document["policy"]["passed"]):
            return 1

    print("PRELIMINARY: not frozen until the code freeze of 10 October.")
    return 0


def _format_weights(weights: np.ndarray, columns: tuple[str, ...]) -> str:
    """Render a portfolio with its column names."""
    return "  ".join(f"{name} {weight:.2f}" for name, weight in zip(columns, weights, strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
