"""Tests of the simulation environment, thesis subsection 4.3.4.

The four checks the thesis declares are not reimplemented here: the tests call
:func:`portfolio_manager.env.environment_checks`, which is the same function the command line
runs, so the figures reported in subsection 4.4 are the figures the suite asserts on. What is
tested directly is everything the four aggregate checks cannot see, chiefly the arrival weights
and the shaping identity.
"""

# Imports
# =====================================================================
import numpy as np
import pytest
import yaml

from portfolio_manager.env import (
    OBJECTIVE_INDICATOR,
    OBJECTIVE_LPM2,
    GeneratorSpec,
    GoalPortfolioEnv,
    ProblemConfig,
    drift_weights,
    environment_checks,
    load_problem,
    project_to_simplex,
    rollout,
    step_wealth,
    terminal_reward,
)
from portfolio_manager.generator import config_path, generate

# Constants
# =====================================================================
SEED = 1000  # validation band, never the training band 0-999 nor the test base
TOY_RETURNS = np.array([[[0.10, 0.00], [-0.20, 0.00], [0.50, 0.00]]], dtype=np.float64)


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def problem() -> ProblemConfig:
    """The base case as configured."""
    return load_problem()


@pytest.fixture(scope="module")
def scenarios(problem) -> np.ndarray:
    """A small mode-B tensor, from the validation band."""
    return generate("B", 32, problem.horizon_months, SEED, block_length=12.0)


# ======================================================================================
# The base case
# ======================================================================================
def test_problem_configuration_matches_the_base_case(problem):
    """The figures of thesis 4.2.1, read from configuration and never from a literal in code."""
    assert problem.goal_usd == 60000.0
    assert problem.horizon_months == 120
    assert problem.monthly_contribution_usd == 400.0
    assert problem.initial_wealth_usd == 0.0
    assert problem.total_contributions_usd == 48000.0
    assert problem.objective == OBJECTIVE_LPM2


def test_cost_is_expressed_in_basis_points(problem):
    """Ten basis points per unit of turnover is a rate of 0.001, not of 0.01 or 0.0001."""
    assert problem.cost_bp == 10.0
    assert problem.cost_rate == pytest.approx(0.0010)
    assert problem.cost_sweep_bp == (0.0, 10.0, 25.0)


def test_the_cost_agrees_with_the_comparator_configuration(problem):
    """``problem.yml`` and ``baselines.yml`` both carry lambda; disagreement would be silent."""
    baselines = yaml.safe_load(config_path("baselines").read_text(encoding="utf-8"))
    assert problem.cost_rate == pytest.approx(float(baselines["transaction_cost"]["lambda"]))


def test_an_unknown_objective_is_rejected(tmp_path):
    """The objective selects the reward, so a typo must fail at load and not at evaluation."""
    path = tmp_path / "problem.yml"
    raw = yaml.safe_load(config_path("problem").read_text(encoding="utf-8"))
    raw["objective"] = "sharpe"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown objective"):
        load_problem(path)


# ======================================================================================
# Projection onto the simplex
# ======================================================================================
def test_projection_clips_and_renormalises():
    """Negative components go to zero and the rest are divided by their sum."""
    projected = project_to_simplex(np.array([-1.0, 1.0, 3.0, 0.0]))
    np.testing.assert_allclose(projected, [0.0, 0.25, 0.75, 0.0])
    assert projected.sum() == pytest.approx(1.0)


def test_projection_reaches_the_corners():
    """A corner portfolio must be attainable, which is what rules out a softmax here."""
    projected = project_to_simplex(np.array([0.0, 0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(projected, [0.0, 0.0, 1.0, 0.0])


def test_projection_of_a_null_action_is_the_equal_split():
    """An all-zero action has no direction; the declared fallback is 1/N."""
    np.testing.assert_allclose(project_to_simplex(np.zeros(4)), np.full(4, 0.25))
    np.testing.assert_allclose(project_to_simplex(np.array([-3.0, -1.0, -2.0, 0.0])), np.full(4, 0.25))


def test_projection_leaves_a_simplex_point_untouched():
    """The Dirichlet policy already emits simplex points, so for it this must be the identity."""
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(project_to_simplex(weights), weights)


def test_projection_is_vectorised():
    """The solver and the rollout project whole batches at once."""
    batch = np.array([[-1.0, 1.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    projected = project_to_simplex(batch)
    assert projected.shape == batch.shape
    np.testing.assert_allclose(projected.sum(axis=-1), 1.0)
    np.testing.assert_allclose(projected[1], 0.25)


# ======================================================================================
# The transition
# ======================================================================================
def test_turnover_sums_purchases_and_sales():
    """A complete switch from one asset to another is a turnover of two, with no factor of a half.

    This single convention fixes the cost of everything downstream. Under the halved definition the
    closed form of check 2 would land on a different figure, which is why that check is the one
    that pins it.
    """
    _, turnover, _ = step_wealth(
        1000.0, np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.zeros(2), 0.0, 0.0
    )
    assert turnover == pytest.approx(2.0)


def test_moving_one_per_cent_costs_two_thousandths_of_a_per_cent():
    """The sentence of thesis 4.2.3: a 1 % transfer is a 2 % turnover and costs 0.002 % of wealth."""
    wealth, turnover, _ = step_wealth(
        1000.0, np.array([0.60, 0.40]), np.array([0.61, 0.39]), np.zeros(2), 0.0, 0.0010
    )
    assert turnover == pytest.approx(0.02)
    assert (1000.0 - wealth) / 1000.0 == pytest.approx(0.00002)


def test_no_cost_when_the_portfolio_is_already_in_place():
    """Holding costs nothing; only the rebalancing does."""
    weights = np.array([0.5, 0.5])
    wealth, turnover, _ = step_wealth(1000.0, weights, weights, np.zeros(2), 0.0, 0.0010)
    assert turnover == 0.0
    assert wealth == pytest.approx(1000.0)


def test_arrival_weights_drift_towards_the_asset_that_rose():
    """Equation ``eq:deriva-pesos``: the winner gains weight without anyone buying it."""
    arrival = drift_weights(np.array([0.5, 0.5]), np.array([0.20, 0.0]))
    np.testing.assert_allclose(arrival, [0.6 / 1.1, 0.5 / 1.1])
    assert arrival.sum() == pytest.approx(1.0)
    assert arrival[0] > 0.5


def test_arrival_weights_always_sum_to_one():
    """They are shares of the portfolio after the month, so the sum is one by construction."""
    rng = np.random.default_rng(SEED)
    weights = rng.dirichlet(np.ones(4), size=500)
    returns = rng.normal(0.0, 0.05, size=(500, 4))
    np.testing.assert_allclose(drift_weights(weights, returns).sum(axis=-1), 1.0)


def test_contribution_arrives_after_the_return_and_free_of_cost():
    """Order inside a step: cost on the opening wealth, then the month, then the contribution."""
    wealth, _, _ = step_wealth(
        1000.0, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.10, 0.0]), 400.0, 0.0010
    )
    # 1000 * (1 - 0.001*2) * 1.10 + 400. A contribution charged the cost would give less.
    assert wealth == pytest.approx(1000.0 * 0.998 * 1.10 + 400.0)


def test_step_wealth_is_vectorised_over_a_batch():
    """The batched call must equal the loop, element for element."""
    rng = np.random.default_rng(SEED)
    wealth = rng.uniform(100.0, 10_000.0, size=64)
    weights = rng.dirichlet(np.ones(4), size=64)
    arrival = rng.dirichlet(np.ones(4), size=64)
    returns = rng.normal(0.005, 0.04, size=(64, 4))

    batched = step_wealth(wealth, weights, arrival, returns, 400.0, 0.0010)
    for i in range(64):
        one = step_wealth(wealth[i], weights[i], arrival[i], returns[i], 400.0, 0.0010)
        assert batched[0][i] == pytest.approx(one[0], rel=0, abs=1e-12)
        assert batched[1][i] == pytest.approx(one[1], rel=0, abs=1e-15)
        np.testing.assert_allclose(batched[2][i], one[2], rtol=0.0, atol=1e-15)


def test_step_wealth_broadcasts_over_a_grid_of_actions():
    """What ``dp_solver.py`` needs: every wealth node against every candidate portfolio at once.

    Without this the solver would carry its own copy of the transition, and the validation against
    Browne would then debug one of two implementations (D74).
    """
    wealth = np.linspace(0.0, 4.0, 11).reshape(-1, 1)  # (11, 1) wealth nodes
    weights = np.eye(4)[None, :, :]  # (1, 4, 4) candidate portfolios
    arrival = weights
    returns = np.full((1, 1, 4), 0.01)

    new_wealth, turnover, _ = step_wealth(wealth, weights, arrival, returns, 0.0, 0.0)
    assert new_wealth.shape == (11, 4)
    np.testing.assert_allclose(turnover, 0.0)
    # Every asset returns one per cent here, so each of the four candidates grows every node alike.
    np.testing.assert_allclose(new_wealth, np.broadcast_to(wealth * 1.01, (11, 4)))

    # With the candidates distinguished, each column follows its own asset.
    returns = np.array([0.01, 0.02, 0.03, 0.04])[None, None, :]
    new_wealth, _, _ = step_wealth(wealth, weights, arrival, returns, 0.0, 0.0)
    np.testing.assert_allclose(new_wealth, wealth * (1.0 + returns[0, 0]))


# ======================================================================================
# The reward
# ======================================================================================
def test_lower_partial_moment_is_bounded_and_one_sided():
    """Equation ``eq:recompensa`` is already in goal units, hence in [-1, 0] (D74)."""
    rewards = terminal_reward(np.array([0.0, 30_000.0, 60_000.0, 90_000.0]), 60_000.0, OBJECTIVE_LPM2)
    np.testing.assert_allclose(rewards, [-1.0, -0.25, 0.0, 0.0])
    assert rewards.min() >= -1.0 and rewards.max() <= 0.0


def test_exceeding_the_goal_earns_nothing_extra():
    """The asymmetry is the point: for an investor with a goal, a surplus is not a problem."""
    assert terminal_reward(np.array([60_000.0]), 60_000.0, OBJECTIVE_LPM2)[0] == 0.0
    assert terminal_reward(np.array([600_000.0]), 60_000.0, OBJECTIVE_LPM2)[0] == 0.0


def test_a_double_shortfall_weighs_four_times_as_much():
    """Order 2 rather than order 1 or 0, which is what gives the gradient a sense of magnitude."""
    small = terminal_reward(np.array([54_000.0]), 60_000.0, OBJECTIVE_LPM2)[0]
    large = terminal_reward(np.array([48_000.0]), 60_000.0, OBJECTIVE_LPM2)[0]
    assert large == pytest.approx(4.0 * small)


def test_the_indicator_objective_is_the_order_zero_one():
    """Used only to validate the solver against Browne, never for the agent (D44)."""
    rewards = terminal_reward(np.array([59_999.0, 60_000.0]), 60_000.0, OBJECTIVE_INDICATOR)
    np.testing.assert_array_equal(rewards, [0.0, 1.0])


# ======================================================================================
# The Gymnasium environment
# ======================================================================================
def test_initial_observation_is_the_state_of_the_equation(scenarios, problem):
    """``(W_0/G, 0, cT/G)`` = ``(0, 0, 0.8)`` for the base case, in single precision."""
    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    observation, _ = env.reset()
    assert observation.dtype == np.float32
    np.testing.assert_allclose(observation, [0.0, 0.0, 0.8], atol=1e-7)
    assert env.observation_space.contains(observation)


def test_remaining_contributions_fall_to_zero_at_the_horizon(scenarios, problem):
    """The third component is ``c(T - t)/G``, so the state stops promising money already paid."""
    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    env.reset()
    for _ in range(problem.horizon_months):
        observation, _, terminated, _, _ = env.step(np.full(4, 0.25))
    assert terminated
    assert observation[1] == pytest.approx(1.0)
    assert observation[2] == pytest.approx(0.0)


def test_the_last_step_terminates_and_never_truncates(scenarios, problem):
    """D57: a time limit would make the agent add the value of the final state to the return."""
    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    env.reset()
    for month in range(problem.horizon_months):
        _, _, terminated, truncated, _ = env.step(np.full(4, 0.25))
        assert truncated is False
        assert terminated == (month == problem.horizon_months - 1)


def test_stepping_past_the_end_is_an_error(scenarios, problem):
    """Silently continuing would pay a second terminal reward."""
    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    env.reset()
    for _ in range(problem.horizon_months):
        env.step(np.full(4, 0.25))
    with pytest.raises(RuntimeError, match="after the episode terminated"):
        env.step(np.full(4, 0.25))


def test_stepping_before_reset_is_an_error(scenarios, problem):
    """There is no scenario to step through yet."""
    with pytest.raises(RuntimeError, match="before reset"):
        GoalPortfolioEnv(scenarios=scenarios, problem=problem).step(np.full(4, 0.25))


def test_scenarios_are_consumed_in_order(scenarios, problem):
    """Episode ``i`` uses row ``i``: this is what makes the comparison across policies paired.

    If the order drifted, two policies would be scored on different futures and the paired
    difference of proportions of H2 would lose its meaning.
    """
    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    for expected in range(4):
        _, info = env.reset()
        assert info["scenario_index"] == expected
        np.testing.assert_array_equal(env._scenario, scenarios[expected])


def test_a_scenario_can_be_selected_explicitly(scenarios, problem):
    """So that a failing scenario can be replayed without rolling the whole tensor forward."""
    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    _, info = env.reset(options={"scenario_index": 7})
    assert info["scenario_index"] == 7
    np.testing.assert_array_equal(env._scenario, scenarios[7])


def test_exhausting_the_tensor_raises(problem):
    """The evaluation loop must run exactly one episode per scenario, not wrap around silently."""
    env = GoalPortfolioEnv(scenarios=generate("B", 2, problem.horizon_months, SEED, block_length=12.0))
    env.reset()
    env.reset()
    with pytest.raises(IndexError, match="one episode per scenario"):
        env.reset()


def test_exactly_one_construction_mode_is_accepted(scenarios):
    """Evaluation reads a frozen tensor, training generates; an environment is one or the other."""
    with pytest.raises(ValueError, match="exactly one"):
        GoalPortfolioEnv()
    with pytest.raises(ValueError, match="exactly one"):
        GoalPortfolioEnv(scenarios=scenarios, generator_spec=GeneratorSpec(mode="B", block_length=12.0))


def test_the_action_space_contains_the_simplex(scenarios, problem):
    """``Box(0, 1, (4,))`` contains the simplex and is what the policy writes into."""
    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    assert env.action_space.shape == (4,)
    assert env.action_space.contains(np.full(4, 0.25, dtype=np.float32))
    assert env.action_space.contains(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def test_construction_overrides_reach_the_transition(problem):
    """Interface required by section 2b: the solver and the Browne runner need these.

    With no contribution, no cost and a single riskless month, the wealth is a pure compounding and
    the indicator objective reports success against the goal.
    """
    env = GoalPortfolioEnv(
        scenarios=np.full((1, 1, 4), 0.10),
        goal=1000.0,
        horizon=1,
        contribution=0.0,
        initial_wealth=1000.0,
        cost_rate=0.0,
        objective=OBJECTIVE_INDICATOR,
    )
    env.reset()
    _, reward, terminated, _, info = env.step(np.full(4, 0.25))
    assert terminated
    assert info["wealth"] == pytest.approx(1100.0)
    assert reward == 1.0


# ======================================================================================
# Training mode
# ======================================================================================
def test_training_mode_requires_a_seed():
    """An unseeded training run could not be reproduced, which the protocol does not allow."""
    with pytest.raises(ValueError, match="needs a seed"):
        GoalPortfolioEnv(generator_spec=GeneratorSpec(mode="B", block_length=12.0))


def test_training_mode_draws_a_new_scenario_at_every_reset(problem):
    """Thesis 4.3.4: the episodes are generated on the fly rather than cycled from a tensor."""
    env = GoalPortfolioEnv(generator_spec=GeneratorSpec(mode="B", block_length=12.0), seed=0, problem=problem)
    env.reset()
    first = env._scenario.copy()
    env.reset()
    assert not np.array_equal(first, env._scenario)


def test_training_mode_advances_one_stream_rather_than_reseeding(problem):
    """One seed per sub-environment, advanced across the run.

    Reseeding per episode would make the episodes a function of a derived seed sequence instead of
    the one declared seed, and two sub-environments could then overlap without it being visible.
    """
    spec = GeneratorSpec(mode="B", block_length=12.0)
    first = GoalPortfolioEnv(generator_spec=spec, seed=0, problem=problem)
    second = GoalPortfolioEnv(generator_spec=spec, seed=0, problem=problem)
    third = GoalPortfolioEnv(generator_spec=spec, seed=1, problem=problem)

    for env in (first, second, third):
        env.reset()
        env.reset()
    np.testing.assert_array_equal(first._scenario, second._scenario)
    assert not np.array_equal(first._scenario, third._scenario)


def test_training_mode_works_in_both_generator_modes(problem):
    """Mode A for the comparison against the table, mode B for the main engine."""
    for spec in (GeneratorSpec(mode="A"), GeneratorSpec(mode="B", block_length=12.0)):
        env = GoalPortfolioEnv(generator_spec=spec, seed=0, problem=problem)
        env.reset()
        assert env._scenario.shape == (problem.horizon_months, 4)


def test_the_generator_specification_comes_from_the_frozen_recipe():
    """Mode and block length are taken from the named configuration; the test seed never is."""
    assert GeneratorSpec.from_test_set("A_gbm") == GeneratorSpec(mode="A", block_length=None)
    assert GeneratorSpec.from_test_set("B_L12") == GeneratorSpec(mode="B", block_length=12.0)
    assert GeneratorSpec.from_test_set("B_L24").block_length == 24.0


# ======================================================================================
# The vectorised rollout
# ======================================================================================
def test_the_rollout_reproduces_the_environment_step_by_step(scenarios, problem):
    """The batched path and the Gymnasium path must agree exactly, not merely closely.

    They share :func:`step_wealth`, so the rollout is a speed-up and not a second model; this is
    the assertion that keeps it that way.
    """
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    batched = rollout(scenarios, lambda observations, t: np.broadcast_to(weights, (len(observations), 4)))

    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    for index in range(len(scenarios)):
        env.reset()
        terminated = False
        turnover = 0.0
        while not terminated:
            _, reward, terminated, _, info = env.step(weights)
            turnover += info["turnover"]
        assert info["wealth"] == pytest.approx(batched["terminal_wealth"][index], rel=0.0, abs=1e-9)
        assert turnover == pytest.approx(batched["total_turnover"][index], rel=0.0, abs=1e-12)
        assert reward == pytest.approx(batched["reward"][index], rel=0.0, abs=1e-15)


def test_the_rollout_projects_whatever_the_policy_returns(scenarios):
    """``evaluate.py`` must not have to trust a policy to stay on the simplex."""
    result = rollout(scenarios, lambda observations, t: np.full((len(observations), 4), -1.0))
    equal = rollout(scenarios, lambda observations, t: np.full((len(observations), 4), 0.25))
    np.testing.assert_allclose(result["terminal_wealth"], equal["terminal_wealth"])


def test_the_rollout_sees_the_state_the_environment_reports(scenarios, problem):
    """A time-dependent policy such as the glide path reads ``t/T``, so it must be correct here."""
    seen: list[np.ndarray] = []

    def policy(observations: np.ndarray, t: int) -> np.ndarray:
        seen.append(observations[0].copy())
        return np.full((len(observations), 4), 0.25)

    rollout(scenarios, policy, problem=problem)
    assert len(seen) == problem.horizon_months
    assert seen[0] == pytest.approx(np.float32([0.0, 0.0, 0.8]), abs=1e-7)
    assert seen[-1][1] == pytest.approx((problem.horizon_months - 1) / problem.horizon_months)
    assert seen[-1][2] == pytest.approx(problem.monthly_contribution_usd / problem.goal_usd)


# ======================================================================================
# Potential-based shaping
# ======================================================================================
def test_shaping_is_off_by_default(scenarios, problem):
    """Only the pre-declared rule of the unattended batch may turn it on."""
    assert GoalPortfolioEnv(scenarios=scenarios, problem=problem).shaping is False


def test_the_potential_vanishes_at_the_terminal_state(scenarios, problem):
    """The condition of Ng, Harada and Russell that the brief omitted.

    At ``t = T`` there are no contributions left, so the shortfall written by the potential is the
    terminal shortfall itself and the potential equals the terminal reward. Letting it through
    would add a second, policy-dependent copy of that reward and would change the optimal policy
    the shaping exists to preserve.
    """
    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem, shaping=True, shaping_gamma=0.99)
    assert env._potential(0.0, problem.horizon_months) == 0.0
    assert env._potential(10_000.0, problem.horizon_months) == 0.0
    # One month earlier the same wealth gives a strictly negative potential, so the zero above is
    # the terminal convention and not an accident of the arithmetic.
    assert env._potential(0.0, problem.horizon_months - 1) < 0.0


def test_shaping_adds_a_constant_to_the_discounted_return(scenarios, problem):
    """Policy invariance, verified as the identity it rests on.

    The discounted sum of the shaping terms telescopes to ``-Phi(s_0)``, which depends on the
    starting state alone. Every policy from the same start receives the same offset, so the
    ordering between policies is untouched and the comparison against the table stays valid.
    """
    gamma = 0.99
    weights = np.array([0.7, 0.1, 0.1, 0.1])
    plain = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    shaped = GoalPortfolioEnv(scenarios=scenarios, problem=problem, shaping=True, shaping_gamma=gamma)

    for _ in range(8):
        plain.reset()
        shaped.reset()
        offset = shaped._potential(problem.initial_wealth_usd, 0)
        plain_return = shaped_return = 0.0
        terminated = False
        month = 0
        while not terminated:
            _, reward, terminated, _, _ = plain.step(weights)
            plain_return += (gamma**month) * reward
            _, reward, _, _, _ = shaped.step(weights)
            shaped_return += (gamma**month) * reward
            month += 1
        assert shaped_return - plain_return == pytest.approx(-offset, rel=0.0, abs=1e-12)


def test_shaping_leaves_the_wealth_path_untouched(scenarios, problem):
    """It is a reward transformation; touching the dynamics would not be shaping at all."""
    weights = np.full(4, 0.25)
    plain = GoalPortfolioEnv(scenarios=scenarios, problem=problem)
    shaped = GoalPortfolioEnv(scenarios=scenarios, problem=problem, shaping=True, shaping_gamma=0.99)
    plain.reset()
    shaped.reset()
    for _ in range(problem.horizon_months):
        _, _, _, _, left = plain.step(weights)
        _, _, _, _, right = shaped.step(weights)
        assert left["wealth"] == right["wealth"]


def test_shaping_will_not_invent_a_discount(scenarios, problem, monkeypatch, tmp_path):
    """It must equal the discount the algorithm uses, so it cannot be defaulted here.

    Until ``config/ppo.yml`` exists the only correct behaviour is to refuse: a potential discounted
    at one rate while the algorithm optimises at another is no longer policy-invariant.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="same discount"):
        GoalPortfolioEnv(scenarios=scenarios, problem=problem, shaping=True)


def test_shaping_reads_the_discount_from_the_agent_configuration(scenarios, problem, monkeypatch, tmp_path):
    """When that file does exist, the value comes from it and not from a second declaration."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "ppo.yml").write_text("gamma: 0.97\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    env = GoalPortfolioEnv(scenarios=scenarios, problem=problem, shaping=True)
    assert env._shaping_gamma == pytest.approx(0.97)


# ======================================================================================
# The four checks of thesis 4.3.4
# ======================================================================================
@pytest.fixture(scope="module")
def checks() -> dict:
    """The declared checks, run once for the module."""
    return environment_checks()


@pytest.mark.parametrize(
    "check", ["contributions_only", "turnover_cost", "bootstrap_bounds", "interface", "toy_path"]
)
def test_every_declared_check_passes(checks, check):
    """Each check reported separately, so a failure names itself."""
    assert checks["checks"][check]["passed"], checks["checks"][check]


def test_contributions_alone_reach_the_sum_exactly(checks):
    """Check 1: with no return and no cost the result is 48,000 USD for any policy, exactly."""
    entry = checks["checks"]["contributions_only"]
    assert entry["expected_terminal_wealth"] == 48_000.0
    assert entry["max_absolute_error"] == 0.0
    assert len(entry["terminal_wealth"]) >= 4


def test_the_cost_of_alternating_vertices_matches_the_closed_form(checks):
    """Check 2: 42,712.23 USD, which is the figure thesis 4.3.4 states."""
    entry = checks["checks"]["turnover_cost"]
    assert entry["closed_form"] == pytest.approx(42_712.23, abs=0.01)
    assert entry["absolute_error"] < 1e-6


def test_terminal_wealth_sits_inside_the_declared_bounds(checks):
    """Check 3: above the contributions, below their compounding at the best asset."""
    entry = checks["checks"]["bootstrap_bounds"]
    assert entry["lower_bound"] == 48_000.0
    assert entry["upper_bound_asset"] == "us_equity"
    for policy in entry["policies"].values():
        assert entry["lower_bound"] < policy["median_terminal_wealth"] < entry["upper_bound"]


def test_rebalancing_a_constant_target_still_costs_turnover(checks):
    """A fixed target is not a no-trade strategy: the weights drift every month and are pulled back."""
    annualised = checks["checks"]["bootstrap_bounds"]["policies"]["constant_mix_60_40"]["annualised_turnover"]
    assert annualised > 0.0


def test_the_reward_stays_inside_its_bound_in_every_episode(checks):
    """Check 4: the interface holds and the reward is in [-1, 0] throughout."""
    entry = checks["checks"]["interface"]
    assert entry["check_env"] == "passed"
    assert entry["min_reward"] >= -1.0
    assert entry["max_reward"] <= 0.0


def test_every_reported_figure_reproduces(checks):
    """Running the checks twice must give the same numbers, not merely the same verdicts.

    Subsection 4.4 quotes these figures, so any of them drifting between runs would make the text
    unverifiable. The one that can drift is the reward bound, which comes from random actions: its
    stream is the action space, separate from the scenario stream, and has to be seeded too.
    """
    assert environment_checks()["checks"] == checks["checks"]


def test_the_hand_computed_path_is_reproduced_exactly(checks):
    """Check 5: the three-month path, including the drift of the arrival weights."""
    entry = checks["checks"]["toy_path"]
    assert entry["max_absolute_error"] == 0.0
    assert entry["terminated_on_last_step"] is True
    assert [row["turnover"] for row in entry["observed"]] == pytest.approx([0.0, 1.0, 8.0 / 9.0])
