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
    uv run python -m portfolio_manager.dp_solver --browne
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
from portfolio_manager.browne import (
    BrowneParameters,
    borrowing_boundary,
    optimal_fraction,
    parameters_from_gbm,
    relative_wealth,
)
from portfolio_manager.browne import value as browne_value
from portfolio_manager.env import (
    OBJECTIVE_INDICATOR,
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

SETTLED_TOLERANCE = 1e-4
"""Below this, successive values of a refinement sequence count as settled whether or not they
keep shrinking. The remaining movement is floating-point noise, and calling it a divergence would
be reading the last digits of a converged answer."""


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
# The validation configuration (D44)
# ======================================================================================
def scale_params(params: GbmParams, steps_per_month: float) -> GbmParams:
    """Rescale the monthly log-return parameters to a step of ``1 / steps_per_month`` months.

    Log returns add over time and their covariance adds with it, so a step of ``h`` months has
    mean ``h * mu`` and covariance ``h * Sigma``. That is what makes a sequence of step sizes a
    refinement of one problem rather than three unrelated ones, which is how the gap between the
    continuous-time closed form and a discrete solver is meant to be handled.

    Args:
        params: Parameters of the monthly log gross returns.
        steps_per_month: Steps the month is divided into.

    Returns:
        Parameters of one step.
    """
    if steps_per_month <= 0:
        raise ValueError(f"steps_per_month must be positive, got {steps_per_month}")
    scale = 1.0 / float(steps_per_month)
    return GbmParams(assets=params.assets, mu=params.mu * scale, sigma=params.sigma * scale)


def browne_setup(
    browne: dict[str, Any],
    columns: tuple[str, ...],
    gbm: GbmParams,
    *,
    mean_shift: float = 0.0,
) -> tuple[GbmParams, np.ndarray, BrowneParameters]:
    """The generator override, the asset mask and the coefficients of the closed form.

    Section 5 of ``docs/browne_reference.md``. The coefficients come from ``config/gbm_params.yml``
    and not from the recorded statistics of the matrix, so that the closed form and the solver
    describe the same market by construction: the generator samples the very log parameters the
    mapping converts.

    The two sets of numbers must not be confused, and the types keep them apart. The returned
    :class:`GbmParams` carries **log** parameters, because that is what the generator samples. The
    returned :class:`BrowneParameters` carries the paper's, whose ``mu`` is the drift of ``dS/S``
    and therefore ``m + s^2/2``.

    Args:
        browne: Parsed validation configuration.
        columns: Normative column order.
        gbm: Parameters of the generator, as fitted.
        mean_shift: Added to the mean log return of the risky column. The third check of section 6
            perturbs the drift this way and expects the optimal control not to follow.

    Returns:
        Tuple ``(generator_params, asset_mask, closed_form_params)``.
    """
    risky, riskless = str(browne["assets"]["risky"]), str(browne["assets"]["riskless"])
    for name in (risky, riskless):
        if name not in columns:
            raise ValueError(f"{name!r} is not a column of the matrix; the header is {list(columns)}")

    risky_index, riskless_index = columns.index(risky), columns.index(riskless)
    log_mean = float(gbm.mu[risky_index]) + float(mean_shift)
    log_variance = float(gbm.sigma[risky_index, risky_index])
    riskless_rate = float(gbm.mu[riskless_index])

    mu = np.zeros(len(columns), dtype=np.float64)
    sigma = np.zeros((len(columns), len(columns)), dtype=np.float64)
    mu[risky_index] = log_mean
    sigma[risky_index, risky_index] = log_variance
    mu[riskless_index] = riskless_rate

    mask = np.zeros(len(columns), dtype=bool)
    mask[[risky_index, riskless_index]] = True
    return (
        GbmParams(assets=columns, mu=mu, sigma=sigma),
        mask,
        parameters_from_gbm(float(gbm.mu[risky_index]), log_variance, riskless_rate, mean_shift=mean_shift),
    )


def load_browne_config(path: Path | None = None) -> dict[str, Any]:
    """Read ``config/validation_browne.yml``."""
    path = path or config_path("validation_browne")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    browne = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if str(browne["objective"]) != OBJECTIVE_INDICATOR:
        raise ValueError(
            f"the validation configuration must use the indicator objective (D44), not "
            f"{browne['objective']!r}: the closed form of Browne maximises the probability of "
            "reaching the goal, and comparing it against a table of order 2 would measure the "
            "difference between objectives rather than the error of the solver"
        )
    if float(browne["contribution_usd"]) != 0.0 or float(browne["cost_bp"]) != 0.0:
        raise ValueError("the validation configuration has neither contributions nor costs")
    return browne


def solve_browne(
    browne: dict[str, Any] | None = None,
    *,
    config: SolverConfig | None = None,
    problem: ProblemConfig | None = None,
    steps_per_month: int = 1,
    mean_shift: float = 0.0,
    progress: bool = False,
) -> Solution:
    """Solve the validation problem at one time step.

    **This produces a table and makes no claim about it.** The closed-form value and policy of the
    paper are transcribed by the author and compared against these tables afterwards; writing them
    from memory would defeat the purpose of the exercise, which is to check the solver against an
    independent source rather than against a second thing written here.

    Args:
        browne: Parsed validation configuration; read from disk when omitted.
        config: Base discretisation, overridden by the validation one.
        problem: Base case, used only for the goal.
        steps_per_month: Steps the month is divided into.
        mean_shift: Shift applied to the mean log return of the risky column, for the third check.
        progress: Whether to print the step being solved.

    Returns:
        The tables, on a grid of normalised wealth as usual.
    """
    browne = browne if browne is not None else load_browne_config()
    config = config or load_solver_config()
    problem = problem or load_problem()
    columns, _ = load_matrix()
    params, mask, _ = browne_setup(browne, columns, load_gbm_params(), mean_shift=mean_shift)

    refined = SolverConfig(
        **{
            **config.__dict__,
            "n_wealth": int(browne["n_wealth"]),
            "wealth_max_over_goal": float(browne["wealth_max_over_goal"]),
            "action_step": float(browne["action_step"]),
            "n_quad": int(browne["n_quad"]),
            "objective": OBJECTIVE_INDICATOR,
        }
    )
    return solve(
        refined,
        params=scale_params(params, steps_per_month),
        problem=problem,
        horizon=int(browne["horizon_months"]) * int(steps_per_month),
        asset_mask=mask,
        objective=OBJECTIVE_INDICATOR,
        contribution=0.0,
        progress=progress,
    )


def comparison_set(
    solution: Solution, closed_form: BrowneParameters, steps_per_month: int, margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The nodes where the paper's problem and this one are the same problem.

    Section 6. At each step the state of the paper is ``z = x e^{r (T - t)}``, and the set is
    ``z*(tau) + margin <= z < 1`` with ``tau = sigma^2 (T - t)``.

    Both bounds matter, for different reasons. Below the boundary the optimal control asks for
    leverage and the simplex forbids it, so a disagreement there is a disagreement about the
    feasible set rather than about the solution. At or above one the bond alone reaches the goal
    and both answers are trivially one and zero, which would inflate the agreement without testing
    anything. The margin keeps the comparison off the boundary, where the monthly investor may
    re-enter the borrowing region before the horizon.

    Args:
        solution: The solved tables.
        closed_form: Coefficients of the paper.
        steps_per_month: Steps the month is divided into.
        margin: Distance from the boundary required of a node.

    Returns:
        Tuple ``(node_index, step_index, months_remaining)``, one entry per comparable state.
    """
    grid = solution.wealth_grid
    total_steps = solution.policy.shape[1]
    months = (total_steps - np.arange(total_steps)) / float(steps_per_month)
    boundary = np.array([borrowing_boundary(closed_form.sigma**2 * month)[1] for month in months])

    z = relative_wealth(grid[:, None], months[None, :], closed_form.r)
    inside = (z >= boundary[None, :] + margin) & (z < 1.0)
    node, step = np.nonzero(inside)
    return node, step, months[step]


def score_browne_checks(
    solution: Solution,
    closed_form: BrowneParameters,
    steps_per_month: int,
    margin: float,
    equity_column: int,
) -> dict[str, Any]:
    """The value and policy checks of section 6 at one refinement pair.

    Returns:
        The two maxima, the size of the comparison set and the figures behind them.
    """
    node, step, months = comparison_set(solution, closed_form, steps_per_month, margin)
    if node.size == 0:
        return {"n_nodes": 0, "value_max_error": float("nan"), "policy_max_error": float("nan")}

    wealth = solution.wealth_grid[node]
    reference_value = browne_value(wealth, months, closed_form)
    value_error = np.abs(solution.value[node, step] - reference_value)

    reference_fraction = optimal_fraction(wealth, months, closed_form)
    policy_error = np.abs(solution.policy[node, step, equity_column] - reference_fraction)

    return {
        "n_nodes": int(node.size),
        "value_max_error": float(value_error.max()),
        "value_mean_error": float(value_error.mean()),
        "policy_max_error": float(policy_error.max()),
        "policy_mean_error": float(policy_error.mean()),
        "max_optimal_fraction": float(reference_fraction.max()),
        "months_remaining_range": [float(months.min()), float(months.max())],
    }


DELTA_SENSITIVITY = (0.1, 0.25, 0.5, 1.0)
"""Drift shifts, in monthly standard deviations, at which the agreement is also reported.

The declared shift is the one in the configuration. This is reported beside it because the paper's
independence result holds for the unconstrained problem, and the solver's holding is confined to
[0, 1]: a shift large enough to push the optimal unconstrained holding through that bound changes
the constrained problem even though it leaves the unconstrained one alone. Reporting the agreement
as a function of the shift separates "the solver is not drift-independent" from "this shift is
large enough that the constraint, and not the drift, is what moved."""

MARGIN_SENSITIVITY = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
"""Margins at which the two maxima are also reported, as a diagnostic and not as a threshold.

The declared margin is the one in the configuration and nothing here changes it. What this adds is
the shape of the disagreement: whether it is spread over the comparison set, which would point at
the solver, or concentrated against the boundary, which is what section 6 predicts and would point
at the constraint. A verdict without that shape says only that two numbers differ."""


def margin_sensitivity(
    solution: Solution,
    closed_form: BrowneParameters,
    steps_per_month: int,
    equity_column: int,
    margins: tuple[float, ...] = MARGIN_SENSITIVITY,
) -> list[dict[str, Any]]:
    """The two maxima as a function of how far the comparison set is kept from the boundary."""
    return [
        {"margin": float(margin), **score_browne_checks(solution, closed_form, steps_per_month, margin, equity_column)}
        for margin in margins
    ]


def score_mu_independence(
    browne: dict[str, Any],
    *,
    config: SolverConfig,
    problem: ProblemConfig,
    steps_per_month: int,
    n_wealth: int,
    margin: float,
    equity_column: int,
) -> dict[str, Any]:
    """The third check of section 6: the optimal control must not involve the drift.

    The paper observes that for a single stock the optimal control "is in fact independent of the
    underlying drift parameter mu, as is apparent from (3.5)" (p. 557). That makes this the
    sharpest of the three: a recursion in which the drift leaked into the policy would reproduce
    the value function and the boundary and still fail here.

    The solve is repeated with the mean log return shifted by one monthly standard deviation in
    **both** directions, the volatility and the riskless rate held. The policy must stay within the
    declared number of action-grid steps on the required fraction of the comparison set, while the
    value table must move: a solver that ignored the drift entirely would pass the first half of
    this check for the wrong reason, and the second half is what rules that out.
    """
    settings = browne["checks"]["mu_independence"]
    columns, _ = load_matrix()
    gbm = load_gbm_params()
    _, _, base_parameters = browne_setup(browne, columns, gbm)
    delta = float(settings["delta_in_monthly_sigma"]) * base_parameters.sigma
    tolerance = float(settings["agreement_grid_steps"]) * config.action_step

    overrides = {"n_wealth": n_wealth}
    base = solve_browne({**browne, **overrides}, config=config, problem=problem, steps_per_month=steps_per_month)
    node, step, _ = comparison_set(base, base_parameters, steps_per_month, margin)

    perturbed: dict[str, Any] = {}
    for label, shift in (("plus", delta), ("minus", -delta)):
        other = solve_browne(
            {**browne, **overrides},
            config=config,
            problem=problem,
            steps_per_month=steps_per_month,
            mean_shift=shift,
        )
        difference = np.abs(
            other.policy[node, step, equity_column] - base.policy[node, step, equity_column]
        )
        value_shift = np.abs(other.value[node, step] - base.value[node, step])
        _, _, shifted_parameters = browne_setup(browne, columns, gbm, mean_shift=shift)
        perturbed[label] = {
            "mean_log_return_shift": float(shift),
            "mu": float(shifted_parameters.mu),
            "premium": float(shifted_parameters.premium),
            "satisfies_paper_assumption": bool(shifted_parameters.satisfies_paper_assumption),
            "policy_agreement": float((difference <= tolerance + 1e-12).mean()),
            "policy_max_difference": float(difference.max()),
            "value_max_difference": float(value_shift.max()),
            "value_table_moved": bool(value_shift.max() > 0.0),
        }

    # Reported beside the declared shift, never in place of it.
    sensitivity = []
    for multiple in DELTA_SENSITIVITY:
        shift = multiple * base_parameters.sigma
        agreement = []
        for direction in (1.0, -1.0):
            other = solve_browne(
                {**browne, **overrides},
                config=config,
                problem=problem,
                steps_per_month=steps_per_month,
                mean_shift=direction * shift,
            )
            moved = np.abs(
                other.policy[node, step, equity_column] - base.policy[node, step, equity_column]
            )
            agreement.append(float((moved <= tolerance + 1e-12).mean()))
        _, _, shifted = browne_setup(browne, columns, gbm, mean_shift=shift)
        sensitivity.append(
            {
                "delta_in_monthly_sigma": float(multiple),
                "delta_as_multiple_of_premium": float(shift / base_parameters.premium),
                "premium_at_minus": float(base_parameters.premium - shift),
                "worst_agreement": float(min(agreement)),
                "agreement_plus": agreement[0],
                "agreement_minus": agreement[1],
                "paper_assumption_holds_both_ways": bool(base_parameters.premium - shift >= 0.0)
                and bool(shifted.satisfies_paper_assumption),
            }
        )

    agreements = [entry["policy_agreement"] for entry in perturbed.values()]
    moved = all(entry["value_table_moved"] for entry in perturbed.values())
    return {
        "delta": float(delta),
        "delta_as_multiple_of_premium": float(delta / base_parameters.premium),
        "delta_in_monthly_sigma": float(settings["delta_in_monthly_sigma"]),
        "tolerance": float(tolerance),
        "minimum_agreement": float(settings["minimum_agreement"]),
        "n_nodes": int(node.size),
        "perturbations": perturbed,
        "worst_agreement": float(min(agreements)),
        "delta_sensitivity": sensitivity,
        "value_tables_moved": moved,
        "passed": bool(min(agreements) >= float(settings["minimum_agreement"]) and moved),
    }


def run_browne_validation(
    browne: dict[str, Any] | None = None,
    *,
    directory: Path = DP_DIRECTORY,
    checks_directory: Path = CHECKS_DIRECTORY,
    argv: list[str] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Solve at every declared refinement pair and score the three checks of section 6.

    The tolerances are read from the configuration and never chosen here, which is the whole of
    D47: a tolerance picked after seeing the error it bounds describes the error rather than
    bounding it.

    Returns:
        The document written to ``results/checks/browne_validation.json``.
    """
    browne = browne if browne is not None else load_browne_config()
    config, problem = load_solver_config(), load_problem()
    columns, _ = load_matrix()
    _, _, closed_form = browne_setup(browne, columns, load_gbm_params())
    # Checks one and two read the closed form, so the coefficients must be ones it speaks about.
    closed_form.require_paper_assumption()
    equity_column = columns.index(str(browne["assets"]["risky"]))

    settings = browne["checks"]
    margin = float(settings["margin"])
    value_tolerance = float(settings["value_tolerance"])
    policy_tolerance = float(settings["policy_grid_steps"]) * float(browne["action_step"])
    initial = float(browne["initial_wealth_over_goal"])

    levels: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for level in browne["refinement"]:
        steps, nodes = int(level["steps_per_month"]), int(level["n_wealth"])
        solution = solve_browne(
            {**browne, "n_wealth": nodes},
            config=config,
            problem=problem,
            steps_per_month=steps,
            progress=progress,
        )
        save(
            solution,
            f"browne_{steps}_per_month",
            SolverConfig(**{**config.__dict__, "n_wealth": nodes, "objective": OBJECTIVE_INDICATOR}),
            problem,
            directory=directory,
            argv=argv,
            extra={"validation": "browne", "steps_per_month": steps},
        )
        levels.append(
            {
                "steps_per_month": steps,
                "n_wealth": nodes,
                "value_at_initial_state": solution.value_at(initial),
                "seconds_elapsed": solution.seconds_elapsed,
                **score_browne_checks(solution, closed_form, steps, margin, equity_column),
            }
        )
        if level is browne["refinement"][-1]:
            sensitivity = margin_sensitivity(solution, closed_form, steps, equity_column)

    value_sequence = [level["value_max_error"] for level in levels]
    policy_sequence = [level["policy_max_error"] for level in levels]
    finest = browne["refinement"][-1]
    independence = score_mu_independence(
        browne,
        config=config,
        problem=problem,
        steps_per_month=int(finest["steps_per_month"]),
        n_wealth=int(finest["n_wealth"]),
        margin=margin,
        equity_column=equity_column,
    )

    document = {
        "reference": "Browne (1999), transcribed in docs/browne_reference.md, verified 2026-10-10",
        "closed_form_parameters": {
            "r": closed_form.r,
            "mu": closed_form.mu,
            "sigma": closed_form.sigma,
            "theta": closed_form.theta,
            "note": "mu is the drift of dS/S, that is m + s^2/2 (section 5)",
        },
        "comparison_set": {
            "rule": "z*(tau) + margin <= z < 1, with z = x e^{r (T - t)} and tau = sigma^2 (T - t)",
            "margin": margin,
            "borrowing_boundary_at_full_horizon": borrowing_boundary(
                closed_form.sigma**2 * float(browne["horizon_months"])
            )[1],
        },
        "levels": levels,
        "value_check": {
            "tolerance": value_tolerance,
            "sequence": value_sequence,
            "finest": value_sequence[-1],
            "monotone_decrease": _is_decreasing(value_sequence),
            "passed": bool(value_sequence[-1] < value_tolerance and _is_decreasing(value_sequence)),
        },
        "policy_check": {
            "tolerance": policy_tolerance,
            "grid_steps": int(settings["policy_grid_steps"]),
            "sequence": policy_sequence,
            "finest": policy_sequence[-1],
            "monotone_decrease": _is_decreasing(policy_sequence),
            "passed": bool(policy_sequence[-1] < policy_tolerance and _is_decreasing(policy_sequence)),
        },
        "mu_independence_check": independence,
        "margin_sensitivity": {
            "note": (
                "diagnostic, not a threshold: the declared margin is the one above. Reported so "
                "that a disagreement concentrated against the borrowing boundary, which section 6 "
                "predicts because future paths re-enter the borrowing region, can be told apart "
                "from one spread over the comparison set, which would point at the solver."
            ),
            "at_finest_pair": sensitivity,
        },
        "theorem_7_1": "not run: P1 only, behind --theorem-71 and the configuration flag (D72)",
        "status": "preliminary until the figures freeze",
    }
    document["all_passed"] = bool(
        document["value_check"]["passed"]
        and document["policy_check"]["passed"]
        and document["mu_independence_check"]["passed"]
    )

    checks_directory.mkdir(parents=True, exist_ok=True)
    path = checks_directory / "browne_validation.json"
    payload = {
        **document,
        "command_line": argv if argv is not None else sys.argv,
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment_fingerprint": collect_fingerprint(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    document["path"] = path
    return document


def _is_decreasing(sequence: list[float]) -> bool:
    """Whether every term of a sequence is strictly smaller than the one before it."""
    return len(sequence) >= 2 and all(
        second < first for first, second in zip(sequence, sequence[1:], strict=False)
    )


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


def _print_browne(argv: list[str], *, progress: bool = False) -> int:
    """Run the validation and report the three verdicts of section 6."""
    browne = load_browne_config()
    document = run_browne_validation(browne, argv=argv, progress=progress)
    parameters = document["closed_form_parameters"]
    comparison = document["comparison_set"]

    print(
        f"validation against Browne (1999), {browne['assets']['risky']} against "
        f"{browne['assets']['riskless']} at zero volatility\n"
        f"objective {browne['objective']}, no contributions, no costs\n"
    )
    print(
        f"closed form   r {parameters['r']:.6f}   mu {parameters['mu']:.6f} (= m + s^2/2)   "
        f"sigma {parameters['sigma']:.6f}   theta {parameters['theta']:.6f}"
    )
    print(
        f"comparison    z*(tau) + {comparison['margin']} <= z < 1; the boundary at the full "
        f"horizon is z* = {comparison['borrowing_boundary_at_full_horizon']:.4f}\n"
    )

    header = f"{'steps/mo':>9} {'nodes':>7} {'compared':>10} {'|dV| max':>11} {'|dw| max':>11} {'seconds':>8}"
    print(header)
    print("-" * len(header))
    for level in document["levels"]:
        print(
            f"{level['steps_per_month']:>9} {level['n_wealth']:>7} {level['n_nodes']:>10,} "
            f"{level['value_max_error']:>11.5f} {level['policy_max_error']:>11.5f} "
            f"{level['seconds_elapsed']:>8.1f}"
        )

    value, policy, independence = (
        document["value_check"],
        document["policy_check"],
        document["mu_independence_check"],
    )
    print(
        f"\n1 value          {'PASS' if value['passed'] else 'FAIL'}  max {value['finest']:.5f} "
        f"against {value['tolerance']}, decreasing {value['monotone_decrease']}"
    )
    print(
        f"2 policy         {'PASS' if policy['passed'] else 'FAIL'}  max {policy['finest']:.5f} "
        f"against {policy['tolerance']:.2f} ({policy['grid_steps']} grid steps), "
        f"decreasing {policy['monotone_decrease']}"
    )
    print(
        f"3 mu-independent {'PASS' if independence['passed'] else 'FAIL'}  agreement "
        f"{independence['worst_agreement']:.4f} against {independence['minimum_agreement']} "
        f"within {independence['tolerance']:.2f}, value tables moved "
        f"{independence['value_tables_moved']}"
    )
    for label, entry in independence["perturbations"].items():
        print(
            f"                   m {label:5} {entry['mean_log_return_shift']:+.5f}: agreement "
            f"{entry['policy_agreement']:.4f}, policy moves at most "
            f"{entry['policy_max_difference']:.4f}, value by {entry['value_max_difference']:.5f}"
        )

    print(f"\nwrote {document['path']}")
    print("Theorem 7.1 not run: P1 only, behind --theorem-71 and the configuration flag (D72).")
    print("PRELIMINARY: not frozen until the figures freeze.")
    return 0 if document["all_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backward induction on the wealth grid.")
    parser.add_argument("--config", default="A_gbm", help="configuration of the frozen recipe, e.g. A_gbm")
    parser.add_argument("--stability", action="store_true", help="also run the two internal checks")
    parser.add_argument(
        "--browne", action="store_true", help="run the validation of D44 against the closed form"
    )
    parser.add_argument(
        "--theorem-71",
        action="store_true",
        help="also score Theorem 7.1, the income case; P1 only, and the configuration must enable it",
    )
    parser.add_argument("--seed", type=int, default=None, help="unused: the induction is deterministic")
    parser.add_argument("--progress", action="store_true", help="print the month being solved")
    args = parser.parse_args(argv)

    try:
        if args.seed is not None:
            raise ValueError("--seed is not accepted: backward induction draws no random numbers")
        if args.theorem_71 and not args.browne:
            raise ValueError("--theorem-71 extends --browne and cannot be run on its own")
        if args.browne:
            if args.theorem_71 and not load_browne_config()["theorem_7_1"]["enabled"]:
                raise ValueError(
                    "Theorem 7.1 is disabled in config/validation_browne.yml. It is P1 rather than "
                    "part of the validation (D72), so both the flag and the configuration must "
                    "agree before an unrequested figure reaches the results."
                )
            return _print_browne(argv if argv is not None else sys.argv, progress=args.progress)
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
