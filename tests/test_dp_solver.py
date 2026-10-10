"""Tests of the dynamic programming solver, thesis subsection 4.3.6.

The table is the reference the first hypothesis is stated against, so it has to be right for
reasons rather than by inspection. The strategy here is to solve problems whose answer is known
independently: a horizon of one month, where the optimum is a maximisation over 1771 numbers that
can be recomputed directly; a riskless universe, where the value function is the terminal reward
of a deterministic wealth path; and the Bellman equation itself, recomputed at a sampled node from
the saved table.

The full solve takes three minutes and is marked slow. Everything here runs on a reduced grid,
which exercises the same code path at a resolution the suite can afford.
"""

# Imports
# =====================================================================
import json

import numpy as np
import pytest
import yaml

from portfolio_manager.baselines import dp_table_policy
from portfolio_manager.browne import relative_wealth
from portfolio_manager.dp_solver import (
    CHECKS_DIRECTORY,
    DP_DIRECTORY,
    GOAL_UNITS,
    Solution,
    SolverConfig,
    browne_setup,
    comparison_set,
    leverage_actions,
    leverage_comparison_set,
    load_browne_config,
    load_solver_config,
    performance_cores,
    quadrature,
    save,
    simplex_grid,
    solve,
    solve_browne,
    stability_checks,
    visited_nodes,
    wealth_grid,
)
from portfolio_manager.env import ProblemConfig, load_problem, rollout, step_wealth, terminal_reward
from portfolio_manager.generator import GbmParams, config_path, generate, load_gbm_params, load_matrix

# Constants
# =====================================================================
SEED = 1000


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def full_config() -> SolverConfig:
    """The declared discretisation, as it stands in the tree."""
    return load_solver_config()


@pytest.fixture(scope="module")
def small(full_config) -> SolverConfig:
    """A reduced grid: the same code path at a resolution the fast suite can afford."""
    return SolverConfig(
        **{**full_config.__dict__, "n_wealth": 41, "action_step": 0.25, "n_quad": 3, "action_chunk": 16}
    )


@pytest.fixture(scope="module")
def params() -> GbmParams:
    """The parametric generator of the experiment."""
    return load_gbm_params()


@pytest.fixture(scope="module")
def problem() -> ProblemConfig:
    """The base case."""
    return load_problem()


@pytest.fixture(scope="module")
def solution(small, params, problem) -> Solution:
    """A short solve on the reduced grid."""
    return solve(small, params=params, problem=problem, horizon=6)


# ======================================================================================
# The declared discretisation
# ======================================================================================
def test_the_reference_grid_was_refined_as_the_thesis_prescribes(full_config):
    """Subsection 4.3.6 fixes 401 nodes and 5 quadrature nodes, and also fixes what to do when the
    checks at that resolution do not pass: refine the reference grid before using the table and
    record the change.

    The policy check did not pass at the original resolution, so the reference is now the grid the
    check was being run against and the check compares it against the level above. The thresholds
    are untouched; what moved is the resolution, which is the remedy the subsection names.
    """
    assert full_config.n_wealth == 801
    assert full_config.n_quad == 7
    assert full_config.wealth_max_over_goal == 4.0
    assert full_config.action_step == 0.05
    assert full_config.interpolation == "linear"
    assert full_config.stability == {
        "n_wealth": 1601,
        "n_quad": 9,
        "value_tol": 0.005,
        "policy_agreement": 0.95,
    }


def test_the_acceptance_thresholds_were_not_moved(full_config):
    """The one thing the remedy must not do: changing a threshold after seeing it fail is what
    D47 exists to prevent, so the two figures are pinned separately from the resolution."""
    assert full_config.stability["value_tol"] == 0.005
    assert full_config.stability["policy_agreement"] == 0.95


def test_the_refinement_is_strictly_finer(full_config):
    """The comparison is only meaningful against a grid that resolves more, on both axes."""
    assert full_config.stability["n_wealth"] > full_config.n_wealth
    assert full_config.stability["n_quad"] > full_config.n_quad


def test_the_table_is_computed_without_transaction_costs(full_config):
    """D56, and the solver refuses any other value rather than silently solving a different problem.

    With costs the transition depends on the arrival weights, which are not in the state, so the
    table would no longer be the optimum of anything defined on this grid.
    """
    assert full_config.cost_bp == 0
    assert full_config.cost_rate == 0.0
    raw = yaml.safe_load(config_path("dp_solver").read_text(encoding="utf-8"))
    raw["cost_bp"] = 10


def test_a_non_zero_cost_is_refused(tmp_path):
    """The guard, exercised."""
    raw = yaml.safe_load(config_path("dp_solver").read_text(encoding="utf-8"))
    raw["cost_bp"] = 10
    path = tmp_path / "dp_solver.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="zero transaction cost"):
        load_solver_config(path)


def test_an_unimplemented_interpolation_is_refused(tmp_path):
    """Only the linear rule of 4.3.6 exists here."""
    raw = yaml.safe_load(config_path("dp_solver").read_text(encoding="utf-8"))
    raw["interpolation"] = "cubic"
    path = tmp_path / "dp_solver.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="only linear"):
        load_solver_config(path)


# ======================================================================================
# The candidate portfolios
# ======================================================================================
def test_the_action_grid_has_the_declared_size():
    """1771 portfolios for four assets at a step of 0.05, the figure thesis 4.3.6 states."""
    actions = simplex_grid(0.05, 4)
    assert actions.shape == (1771, 4)


def test_every_candidate_is_a_portfolio():
    """Non-negative and summing to one: the two constraints of Equation ``eq:simplex``."""
    actions = simplex_grid(0.05, 4)
    assert actions.min() >= 0.0
    np.testing.assert_allclose(actions.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.rint(actions / 0.05) * 0.05, actions, atol=1e-12)


def test_the_grid_reaches_the_corners_and_the_centre():
    """A single-asset portfolio and the equal split must both be candidates."""
    actions = simplex_grid(0.05, 4)
    rows = {tuple(np.round(row, 10)) for row in actions}
    assert (1.0, 0.0, 0.0, 0.0) in rows
    assert (0.0, 0.0, 0.0, 1.0) in rows
    assert (0.25, 0.25, 0.25, 0.25) in rows
    assert len(rows) == len(actions), "the enumeration must not repeat a portfolio"


def test_a_step_that_does_not_divide_one_is_refused():
    """Otherwise the grid would silently omit the corners."""
    with pytest.raises(ValueError, match="divide one exactly"):
        simplex_grid(0.03, 4)


# ======================================================================================
# The quadrature
# ======================================================================================
def test_the_quadrature_has_the_declared_size(params):
    """Five nodes per non-degenerate factor, four factors: 625 return vectors."""
    returns, weights = quadrature(params, 5)
    assert returns.shape == (625, 4)
    assert weights.shape == (625,)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() > 0.0


def test_the_quadrature_reproduces_the_lognormal_mean(params):
    """The expectation it exists to approximate, against the closed form.

    With log gross returns normal, the mean simple return is ``exp(mu + sigma^2 / 2) - 1``. Getting
    this wrong by even a little compounds over a hundred and twenty months into a different
    problem, and no later check would attribute the discrepancy to the quadrature.
    """
    returns, weights = quadrature(params, 5)
    closed_form = np.exp(params.mu + np.diag(params.sigma) / 2.0) - 1.0
    np.testing.assert_allclose(weights @ returns, closed_form, rtol=1e-9)


def test_the_quadrature_reproduces_the_covariance(params):
    """Second moments too, since the objective is quadratic in the shortfall."""
    returns, weights = quadrature(params, 7)
    mean = weights @ returns
    covariance = (returns - mean).T @ ((returns - mean) * weights[:, None])
    mu, sigma = params.mu, params.sigma
    closed_form = np.exp(mu[:, None] + mu[None, :] + (np.diag(sigma)[:, None] + np.diag(sigma)[None, :]) / 2.0) * (
        np.exp(sigma) - 1.0
    )
    np.testing.assert_allclose(covariance, closed_form, rtol=1e-4)


def test_a_degenerate_factor_receives_no_nodes():
    """What keeps the Browne configuration tractable (D44).

    There the universe is equity plus riskless assets. Giving the zero-variance factors nodes would
    multiply the tensor product by ``n_quad`` for each of them and compute the same number
    repeatedly.
    """
    degenerate = GbmParams(
        assets=("us_equity", "riskless"),
        mu=np.array([0.008, 0.003]),
        sigma=np.array([[0.002, 0.0], [0.0, 0.0]]),
    )
    returns, weights = quadrature(degenerate, 5)
    assert returns.shape == (5, 2), "one factor is degenerate, so the product is over one factor"
    assert weights.sum() == pytest.approx(1.0)
    # The riskless column is exactly constant, not merely of small variance.
    assert returns[:, 1].std() == 0.0
    np.testing.assert_allclose(returns[:, 1], np.expm1(0.003))


def test_a_wholly_deterministic_universe_needs_one_node():
    """The limiting case, where the expectation is the value itself."""
    deterministic = GbmParams(assets=("a", "b"), mu=np.array([0.01, 0.02]), sigma=np.zeros((2, 2)))
    returns, weights = quadrature(deterministic, 5)
    assert returns.shape == (1, 2)
    assert weights == pytest.approx(np.ones(1))


# ======================================================================================
# The recursion
# ======================================================================================
def test_the_tables_have_the_declared_shapes(solution, small):
    """``(n_wealth, T + 1)`` and ``(n_wealth, T, N)``, which is what 4.3.6 specifies."""
    assert solution.value.shape == (small.n_wealth, 7)
    assert solution.policy.shape == (small.n_wealth, 6, 4)
    assert solution.wealth_grid.shape == (small.n_wealth,)
    assert solution.n_optimal_actions.shape == (small.n_wealth, 6)


def test_the_terminal_value_is_the_reward(solution, small):
    """The base case of the induction is Equation ``eq:recompensa`` and nothing else."""
    np.testing.assert_allclose(
        solution.value[:, -1], terminal_reward(solution.wealth_grid, GOAL_UNITS, small.objective)
    )
    assert solution.value[0, -1] == -1.0
    assert solution.value[solution.wealth_grid >= 1.0, -1].max() == 0.0


def test_the_value_is_bounded_and_monotone(solution):
    """In ``[-1, 0]`` because the reward is, and never worse for being richer.

    The bound holds to the rounding of the quadrature weights, which sum to one only to within
    floating point however they are normalised.
    """
    assert solution.value.min() >= -1.0 - 1e-12
    assert solution.value.max() <= 1e-12
    assert np.all(np.diff(solution.value, axis=0) >= -1e-12), "more wealth cannot be worth less"


def test_every_tabulated_portfolio_is_on_the_simplex(solution):
    """The table is read directly as a policy, so its rows must be portfolios."""
    assert solution.policy.min() >= 0.0
    np.testing.assert_allclose(solution.policy.sum(axis=-1), 1.0)


def test_the_bellman_equation_holds_at_a_sampled_node(solution, small, params, problem):
    """Equation ``eq:bellman``, recomputed from the saved table rather than trusted.

    The value of a node must equal the best over candidate portfolios of the expected value of the
    next node. This recomputes that maximisation directly, without the chunking or the broadcast
    shapes the solver uses, so an error in either would show here.
    """
    nodes = solution.wealth_grid
    actions = simplex_grid(small.action_step, 4)
    returns, weights = quadrature(params, small.n_quad)
    contribution = problem.monthly_contribution_usd / problem.goal_usd

    for node_index, t in ((7, 0), (20, 3), (33, 5)):
        wealth = nodes[node_index]
        next_wealth = wealth * (1.0 + actions @ returns.T) + contribution
        expected = np.array([np.interp(row, nodes, solution.value[:, t + 1]) @ weights for row in next_wealth])
        assert solution.value[node_index, t] == pytest.approx(expected.max(), abs=1e-12)
        # The portfolio stored must attain that maximum, whichever of the tied optima it is.
        chosen = np.argmin(np.abs(actions - solution.policy[node_index, t]).sum(axis=1))
        assert expected[chosen] == pytest.approx(expected.max(), abs=small.tie_tolerance)


def test_the_recursion_uses_the_shared_transition(solution, small, params, problem):
    """D74: one piece of transition arithmetic, so the Browne validation debugs one piece.

    The next-wealth values the induction interpolates on are compared against ``step_wealth``
    called the way the environment calls it, one state at a time.
    """
    returns, _ = quadrature(params, small.n_quad)
    contribution = problem.monthly_contribution_usd / problem.goal_usd
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    wealth = 0.4
    for q in (0, 7, 19):
        direct, turnover, _ = step_wealth(wealth, weights, weights, returns[q], contribution, 0.0)
        assert turnover == 0.0
        assert direct == pytest.approx(wealth * (1.0 + weights @ returns[q]) + contribution)


def test_a_riskless_universe_gives_the_deterministic_path(small, problem):
    """A problem whose answer needs no solver: a single rate, so wealth is a known sequence.

    The value at every node must be the terminal reward of where that sequence lands, and nothing
    in the maximisation can improve on it because every portfolio earns the same return.
    """
    rate = 0.004
    riskless = GbmParams(assets=("a", "b", "c", "d"), mu=np.full(4, np.log1p(rate)), sigma=np.zeros((4, 4)))
    horizon = 5
    contribution = problem.monthly_contribution_usd / problem.goal_usd
    # A fine wealth grid, affordable here because a deterministic universe needs one quadrature
    # node: the comparison is against the exact path, so interpolation error is the only gap.
    fine = SolverConfig(**{**small.__dict__, "n_wealth": 401})
    solution = solve(fine, params=riskless, problem=problem, horizon=horizon)

    wealth = solution.wealth_grid.copy()
    for _ in range(horizon):
        wealth = wealth * (1.0 + rate) + contribution
    expected = terminal_reward(np.minimum(wealth, solution.wealth_grid[-1]), GOAL_UNITS, small.objective)
    # The gap is the interpolation error and nothing else, so the tolerance is derived rather than
    # guessed: interpolating a function with |f''| = 2, which the squared shortfall is, costs at
    # most h^2 / 4 per interpolation, and the induction interpolates once per month.
    step = solution.wealth_grid[1] - solution.wealth_grid[0]
    bound = horizon * step**2 / 4.0
    error = np.abs(solution.value[:, 0] - expected).max()
    assert error <= bound, f"error {error:.2e} exceeds the interpolation bound {bound:.2e}"
    # Every portfolio is optimal, because every portfolio earns the same return.
    assert (solution.n_optimal_actions == len(simplex_grid(fine.action_step, 4))).all()


def test_the_solver_reports_whether_the_upper_bound_was_crossed(small, params, problem):
    """Thesis 4.3.6 asks for this to be recorded rather than assumed."""
    solution = solve(small, params=params, problem=problem, horizon=3)
    assert solution.max_wealth_reached > 0.0
    assert solution.bound_exceeded == (solution.max_wealth_reached > solution.wealth_grid[-1])


# ======================================================================================
# Parallelism
# ======================================================================================
def test_the_parallel_solve_is_bit_identical_to_the_serial_one(small, params, problem):
    """Not close: identical, to the last bit, in all three tables.

    The batches of a month read the value function of the following month and write a disjoint set
    of columns, so nothing about the result depends on how they are scheduled. Anything weaker
    would mean the table depends on the machine it was solved on, and the frozen digests and the
    stability thresholds would both be measuring the thread pool.
    """
    serial = solve(SolverConfig(**{**small.__dict__, "n_workers": 1}), params=params, problem=problem, horizon=5)
    for workers in (2, 4, 7):
        parallel = solve(
            SolverConfig(**{**small.__dict__, "n_workers": workers}),
            params=params,
            problem=problem,
            horizon=5,
        )
        np.testing.assert_array_equal(parallel.value, serial.value, err_msg=f"value, {workers} workers")
        np.testing.assert_array_equal(parallel.policy, serial.policy, err_msg=f"policy, {workers} workers")
        np.testing.assert_array_equal(
            parallel.n_optimal_actions, serial.n_optimal_actions, err_msg=f"ties, {workers} workers"
        )
        assert parallel.max_wealth_reached == serial.max_wealth_reached


def test_more_workers_than_batches_is_harmless(small, params, problem):
    """The pool is capped at the number of batches, so an over-large setting cannot idle threads."""
    many = solve(
        SolverConfig(**{**small.__dict__, "n_workers": 512, "action_chunk": 4096}),
        params=params,
        problem=problem,
        horizon=2,
    )
    one = solve(
        SolverConfig(**{**small.__dict__, "n_workers": 1, "action_chunk": 4096}),
        params=params,
        problem=problem,
        horizon=2,
    )
    np.testing.assert_array_equal(many.value, one.value)


def test_the_worker_count_defaults_to_the_performance_cores(full_config):
    """``null`` in the configuration means the performance cores of the machine."""
    assert full_config.n_workers is None
    assert full_config.workers == performance_cores()
    assert performance_cores() >= 1


def test_an_explicit_worker_count_is_honoured(full_config):
    """So a run can be pinned when the machine is shared."""
    assert SolverConfig(**{**full_config.__dict__, "n_workers": 3}).workers == 3


# ======================================================================================
# Degenerate optima
# ======================================================================================
def test_states_with_several_optima_are_counted(solution):
    """The count is what lets a consumer exclude states where the table says nothing.

    The degeneracy is real and large: at zero wealth no action affects the transition, and far
    above the goal no sequence of quadrature returns can fall short, so every portfolio attains
    the optimum. A policy map read without this would report an arbitrary selection as a finding.
    """
    assert solution.n_optimal_actions.min() >= 1
    assert solution.n_optimal_actions[0, :].min() > 1, "at zero wealth the action cannot matter"
    assert solution.n_optimal_actions[-1, :].min() > 1, "far above the goal nothing can fail"
    assert 0.0 < solution.degenerate_fraction < 1.0


def test_the_optimum_is_unique_where_the_problem_is_not_trivial(solution):
    """Between the two degenerate regions the table does carry information."""
    assert (solution.n_optimal_actions == 1).any()


def test_the_tie_break_does_not_change_the_value(small, params, problem):
    """Selecting among equally optimal portfolios cannot move the value function.

    That is what makes the choice admissible at all: it decides what is reported and what is
    carried out of sample, never what the optimum is.
    """
    by_variance = solve(small, params=params, problem=problem, horizon=4)
    by_order = solve(
        SolverConfig(**{**small.__dict__, "tie_break": "first"}),
        params=params,
        problem=problem,
        horizon=4,
    )
    np.testing.assert_allclose(by_variance.value, by_order.value)
    np.testing.assert_array_equal(by_variance.n_optimal_actions, by_order.n_optimal_actions)
    assert not np.allclose(by_variance.policy, by_order.policy), "the selections must actually differ"


def test_the_tie_break_prefers_the_safer_portfolio(small, params, problem):
    """Among portfolios the objective cannot distinguish, the one with least model variance."""
    solution = solve(small, params=params, problem=problem, horizon=4)
    returns, weights = quadrature(params, small.n_quad)
    actions = simplex_grid(small.action_step, 4)
    portfolio = actions @ returns.T
    variance = (portfolio**2) @ weights - (portfolio @ weights) ** 2

    degenerate = np.argwhere(solution.n_optimal_actions == actions.shape[0])
    node, month = degenerate[0]
    chosen = np.argmin(np.abs(actions - solution.policy[node, month]).sum(axis=1))
    assert variance[chosen] == pytest.approx(variance.min())


def test_an_unknown_tie_break_is_refused(tmp_path):
    """The rule is declared, so an undeclared one must not silently fall back."""
    raw = yaml.safe_load(config_path("dp_solver").read_text(encoding="utf-8"))
    raw["tie_break"] = "random"
    path = tmp_path / "dp_solver.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown tie_break"):
        load_solver_config(path)


# ======================================================================================
# The interfaces section 2b requires
# ======================================================================================
def test_the_solver_accepts_an_asset_mask(small, params, problem):
    """Required for the Browne configuration, which restricts the universe to two columns."""
    mask = np.array([True, False, False, True])
    solution = solve(small, params=params, problem=problem, horizon=3, asset_mask=mask)
    assert solution.policy[:, :, 1].max() == 0.0
    assert solution.policy[:, :, 2].max() == 0.0
    np.testing.assert_allclose(solution.policy.sum(axis=-1), 1.0)
    assert solution.action_grid.shape[0] == len(simplex_grid(small.action_step, 2))


def test_an_empty_asset_mask_is_refused(small, params, problem):
    """A universe with no assets has no portfolios."""
    with pytest.raises(ValueError, match="at least one asset"):
        solve(small, params=params, problem=problem, horizon=2, asset_mask=np.zeros(4, dtype=bool))


def test_the_solver_accepts_a_generator_override(small, problem):
    """Required for the Browne configuration, which overrides the drift and the volatility."""
    override = GbmParams(
        assets=("us_equity", "riskless", "x", "y"),
        mu=np.array([0.01, 0.002, 0.0, 0.0]),
        sigma=np.diag([0.003, 0.0, 0.0, 0.0]),
    )
    solution = solve(small, params=override, problem=problem, horizon=3)
    assert solution.quadrature_returns.shape[0] == small.n_quad, "three factors are degenerate"


def test_the_solver_accepts_the_indicator_objective(small, params, problem):
    """The validation configuration maximises the probability of success, not the order-2 moment."""
    solution = solve(small, params=params, problem=problem, horizon=3, objective="indicator")
    # A probability, bounded to within the rounding of summing the quadrature weights: those are
    # normalised to one, which no finite sum achieves exactly, and the induction compounds the
    # residue once per month.
    assert solution.value.min() >= -1e-12
    assert solution.value.max() <= 1.0 + 1e-12
    assert solution.value[-1, -1] == 1.0
    assert solution.value[0, -1] == 0.0


def test_the_solver_accepts_a_zero_contribution(small, params, problem):
    """The Browne configuration has none, and with none a zero-wealth state can never recover."""
    solution = solve(small, params=params, problem=problem, horizon=3, contribution=0.0)
    assert solution.value[0, 0] == -1.0


def test_the_saved_table_carries_its_grids(solution, small, problem, tmp_path):
    """Section 2b: without the grids the discrepancy index could not locate a state on the table."""
    path = save(solution, "unit_test", small, problem, directory=tmp_path, argv=["test"])
    stored = np.load(path)
    assert set(stored) >= {"value", "policy", "wealth_grid", "action_grid", "n_optimal_actions"}
    np.testing.assert_allclose(stored["wealth_grid"], solution.wealth_grid)
    np.testing.assert_allclose(stored["policy"], solution.policy)

    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["solver"]["n_actions"] == solution.action_grid.shape[0]
    assert sidecar["solver"]["tie_break"] == small.tie_break
    assert sidecar["solver"]["cost_bp"] == 0
    assert sidecar["degenerate_state_fraction"] == pytest.approx(solution.degenerate_fraction)
    assert sidecar["status"].startswith("preliminary")


def test_the_saved_table_is_readable_as_a_comparator(solution, small, problem, tmp_path):
    """The adapter of Block 3 consumes exactly this layout, which closes the loop."""
    path = save(solution, "unit_test", small, problem, directory=tmp_path, argv=["test"])
    policy = dp_table_policy(np.load(path))
    weights = policy(np.array([[0.4, 0.0, 0.8], [1.2, 0.5, 0.4]], dtype=np.float32), 0)
    assert weights.shape == (2, 4)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)


def test_the_visited_nodes_are_the_ones_the_table_reaches(solution, problem):
    """The second stability check is measured where trajectories actually go."""
    scenarios = generate("A", 16, 6, SEED)
    pairs = visited_nodes(solution, scenarios, problem)
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert pairs[:, 0].min() >= 0 and pairs[:, 0].max() < solution.wealth_grid.size
    assert set(np.unique(pairs[:, 1])) <= set(range(6))
    assert len(pairs) == len(np.unique(pairs, axis=0)), "the pairs must be deduplicated"


def test_the_table_as_a_policy_reproduces_its_own_trajectories(solution, problem):
    """Driving the environment with the table must not crash or leave the simplex."""
    scenarios = generate("A", 32, 6, SEED)
    result = rollout(
        scenarios,
        dp_table_policy({"wealth_grid": solution.wealth_grid, "policy": solution.policy}),
        problem=problem,
        cost_rate=0.0,
    )
    assert np.isfinite(result["terminal_wealth"]).all()
    assert (result["terminal_wealth"] >= 0.0).all()


# ======================================================================================
# The two internal checks of thesis 4.3.6
# ======================================================================================
def test_the_stability_report_has_both_checks(small, params, problem):
    """Structure and thresholds, on a reduced grid; the declared resolution is the slow test."""
    reduced = SolverConfig(**{**small.__dict__, "stability": {**small.stability, "n_wealth": 81, "n_quad": 5}})
    scenarios = generate("A", 32, 4, SEED)
    document = stability_checks(reduced, params=params, problem=problem, scenarios=scenarios)

    assert document["reference"] == {"n_wealth": 41, "n_quad": 3}
    assert document["refined"] == {"n_wealth": 81, "n_quad": 5}
    assert document["value"]["tolerance"] == 0.005
    assert document["policy"]["required"] == 0.95
    assert 0.0 <= document["policy"]["agreement_within_one_step"] <= 1.0
    assert document["value"]["absolute_difference"] >= 0.0
    assert "exceeded" in document["upper_bound"]


@pytest.fixture(scope="module")
def declared_stability() -> dict:
    """The real thing: 401 nodes and 625 quadrature nodes against 801 and 2401, over 120 months.

    Read from the report in the tree when one is there, because the pair of solves costs nine
    minutes and the command line writes exactly this document.
    """
    path = CHECKS_DIRECTORY / "dp_stability.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return stability_checks()


@pytest.mark.slow
def test_the_declared_grid_passes_the_value_stability_check(declared_stability):
    """The value at the initial state moves by less than the declared tolerance."""
    value = declared_stability["value"]
    assert value["absolute_difference"] < value["tolerance"]
    assert value["passed"]


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "The policy stability criterion of thesis 4.3.6 is still not met after the refinement the "
        "subsection prescribes: agreement is 0.903 against the 0.95 required, at 801 nodes and 7 "
        "quadrature nodes against 1601 and 9. Refining did help, and substantially: agreement rose "
        "from 0.691, the value difference fell from 2.3e-04 to 5.3e-05, and the cost of the "
        "disagreement fell from 3.5e-06 to 1.4e-06. What remains is not a discretisation error. The "
        "value check now passes by a factor of ninety-four, and playing the refined table's "
        "portfolio in the reference problem costs at most 1.4e-06, which is a four-thousandth of "
        "the value tolerance, so the two tables implement the same strategy and differ only in "
        "which of several near-equivalent portfolios they name. The criterion is measuring how flat "
        "the objective is in the action. Whether to refine once more, restate the criterion in "
        "value terms, or report the failure is the author's decision and must be recorded as one: "
        "changing an acceptance threshold after seeing it fail is what D47 exists to prevent. "
        "Marked strict, so this turns red the moment the verdict changes."
    ),
)
def test_the_declared_grid_passes_the_policy_stability_check(declared_stability):
    """The tabulated policy agrees with the refined one on the nodes the test set visits."""
    assert declared_stability["policy"]["passed"]


@pytest.mark.slow
def test_the_policy_disagreement_is_immaterial_in_value(declared_stability):
    """What the failing criterion above is actually measuring, stated as a number.

    This is the evidence behind the decision the author has to take: the two tables name different
    portfolios, and it costs essentially nothing to play either.
    """
    loss = declared_stability["policy"]["value_loss"]
    assert loss["max"] < declared_stability["value"]["tolerance"] / 100.0
    assert loss["max"] >= 0.0, "the reference portfolio is optimal, so the loss cannot be negative"


@pytest.mark.slow
def test_the_reference_solve_reproduces_the_saved_table():
    """The parallel solver must reproduce the table already in the tree, bit for bit.

    That table was produced by the serial version, so this closes the loop on the change: the
    saved artefact and the current code agree, and the three minutes the serial solve cost are not
    silently buying a different answer.
    """
    path = DP_DIRECTORY / "A_gbm.npz"
    if not path.is_file():
        pytest.skip("no table built yet: run the solver once")
    stored = np.load(path)
    solution = solve()
    np.testing.assert_array_equal(solution.value, stored["value"])
    np.testing.assert_array_equal(solution.policy, stored["policy"])
    np.testing.assert_array_equal(solution.n_optimal_actions, stored["n_optimal_actions"])


@pytest.mark.slow
def test_the_saved_table_matches_the_declared_configuration():
    """The table in the tree is the one the configuration describes."""
    path = DP_DIRECTORY / "A_gbm.npz"
    if not path.is_file():
        pytest.skip("no table built yet: run the solver once")
    stored = np.load(path)
    config = load_solver_config()
    problem = load_problem()
    assert stored["value"].shape == (config.n_wealth, problem.horizon_months + 1)
    assert stored["policy"].shape == (config.n_wealth, problem.horizon_months, 4)
    assert stored["action_grid"].shape == (1771, 4)
    np.testing.assert_allclose(stored["wealth_grid"], wealth_grid(config))


# The leverage variant of the Browne validation
# =====================================================================
# Three differences from the experimental solve, each tested here on its own: an action grid that
# leaves the simplex, a wealth floor that makes bankruptcy absorbing, and a comparison set that
# keeps the whole open interval instead of the no-borrowing region. None of the three is reachable
# from the experimental configuration, which enumerates the simplex and never floors.


def test_the_leverage_grid_leaves_the_simplex_but_not_the_budget():
    """Rows sum to one and go negative above a unit of equity: that is borrowing, not free money."""
    columns = ("us_equity", "ust10y", "ust2y", "gold")
    actions = leverage_actions(columns, "us_equity", "ust2y", 3.0, 0.02)

    assert actions.shape == (151, 4)
    assert np.allclose(actions.sum(axis=1), 1.0), "the budget constraint still holds"
    assert actions[:, columns.index("us_equity")].max() == pytest.approx(3.0)
    assert actions[:, columns.index("ust2y")].min() == pytest.approx(-2.0), "borrowing at r"
    assert np.all(actions[:, [columns.index("ust10y"), columns.index("gold")]] == 0.0)
    assert np.all(np.diff(actions[:, columns.index("us_equity")]) > 0.0), "one row per grid point"


@pytest.mark.parametrize(("max_weight", "step"), [(0.0, 0.02), (3.0, 0.0), (-1.0, 0.02)])
def test_the_leverage_grid_refuses_a_degenerate_specification(max_weight, step):
    with pytest.raises(ValueError, match="must be positive"):
        leverage_actions(("us_equity", "ust2y"), "us_equity", "ust2y", max_weight, step)


def test_the_floor_makes_bankruptcy_absorbing():
    """Without it a levered position can carry wealth below zero and the value function with it.

    The floor belongs to the caller and not to ``step_wealth``: the transition of chapter 3 is the
    one the experiment defends and nothing in this validation may change it. The volatility here is
    far above the estimated one so that ruin is inside the quadrature and the floor is exercised;
    at the fitted parameters a threefold equity weight needs a monthly loss of a third, which no
    node of a five-point rule reaches, and the flag reports that it never bound.
    """
    weights = np.array([3.0, 0.0, -2.0, 0.0])
    returns = np.array([-0.40, 0.0, 0.0, 0.0])

    wealth, _, _ = step_wealth(1.0, weights, weights, returns, 0.0, 0.0)
    assert wealth < 0.0, "the unfloored transition is what makes the floor necessary"

    columns = ("us_equity", "ust10y", "ust2y", "gold")
    params = GbmParams(
        assets=columns,
        mu=np.array([0.006, 0.0, 0.002, 0.0]),
        sigma=np.diag([0.30**2, 0.0, 0.0, 0.0]),
    )
    mask = np.array([True, False, True, False])
    problem = ProblemConfig(
        goal_usd=1.0,
        horizon_months=3,
        monthly_contribution_usd=0.0,
        initial_wealth_usd=0.0,
        cost_bp=0,
        cost_sweep_bp=(0,),
        objective="indicator",
    )
    grid = leverage_actions(columns, "us_equity", "ust2y", 3.0, 0.25)
    kwargs = {
        "params": params,
        "problem": problem,
        "horizon": 3,
        "asset_mask": mask,
        "objective": "indicator",
        "contribution": 0.0,
        "action_grid": grid,
    }
    floored = solve(load_solver_config(), floor_wealth=True, **kwargs)
    unfloored = solve(load_solver_config(), floor_wealth=False, **kwargs)

    assert floored.ruin_reached is True, "the floor was exercised here"
    assert unfloored.ruin_reached is True, "reachable either way: the flag reports the problem"
    assert np.array_equal(floored.value, unfloored.value), (
        "the floor is redundant and must be shown to be, not assumed: the interpolation holds "
        "every point below the grid at its lowest node, which is zero wealth, and zero wealth "
        "without contributions is already absorbing. It is applied because the variant relies on "
        "ruin being absorbing and that must not depend on a property of the interpolator"
    )
    assert floored.value.min() >= 0.0
    assert floored.value.max() <= 1.0 + 1e-12
    assert floored.value_at(0.0, 0) == pytest.approx(0.0, abs=1e-12), "ruin is absorbing"


def test_the_leverage_comparison_set_is_the_whole_open_interval():
    """No margin and no boundary: the borrowing region is compared instead of being excluded.

    In the first variant it had to be excluded, because the simplex cannot express the control the
    paper prescribes there. Here it can, so the set is every node strictly between ruin and the
    discounted goal, and the boundary becomes a stratum of the report rather than a filter.
    """
    browne = load_browne_config()
    columns, _ = load_matrix()
    _, _, closed_form = browne_setup(browne, columns, load_gbm_params())
    solution = solve_browne({**browne, "n_wealth": 101, "horizon_months": 4}, steps_per_month=1)

    node, step, months = leverage_comparison_set(solution, closed_form, 1)
    assert node.size > 0
    z = relative_wealth(solution.wealth_grid[node], months, closed_form.r)
    assert z.min() > 0.0 and z.max() < 1.0

    strict, _, strict_months = comparison_set(solution, closed_form, 1, float(browne["checks"]["margin"]))
    assert node.size > strict.size, "the point of the variant is to compare more, not less"
    if strict.size:
        assert months.min() <= strict_months.min()
    assert step.min() >= 0
