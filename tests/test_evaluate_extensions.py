"""Tests of the conditional extension: the trained policy, the guard, the index and the test.

Four pieces that exist to serve work scheduled after the build day, so each is tested against
something that does not depend on that work having happened: the discrepancy index against a
synthetic table whose answer is known by construction, the paired test against counts worked out
by hand, the guard against itself, and the trained-policy wrapper against the smoke run.
"""

# Imports
# =====================================================================
import json
from dataclasses import replace

import numpy as np
import pytest

from portfolio_manager.dp_solver import (
    browne_setup,
    load_browne_config,
    run_browne_validation,
    scale_params,
    solve_browne,
)
from portfolio_manager.env import load_problem
from portfolio_manager.evaluate import (
    REFERENCE_TEST_SET,
    discrepancy_index,
    evaluate_policies,
    load_trained_policy,
    paired_proportions,
    reference_digest,
    reference_run,
    run,
    table_policy,
    trained_policies,
)
from portfolio_manager.generator import GbmParams, generate, load_matrix, load_matrix_metadata

# Constants
# =====================================================================
SEED = 1000
RUNS = "results/runs"


# ======================================================================================
# A trained policy, scored like any other
# ======================================================================================
@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory) -> object:
    """A trained policy to wrap. Reuses the one in the tree, and trains a short one otherwise."""
    from pathlib import Path

    existing = Path(RUNS) / "A_gbm" / "seed_0" / "policy.zip"
    if existing.is_file():
        return existing
    from portfolio_manager.train import train

    directory = tmp_path_factory.mktemp("runs")
    train("A_gbm", 0, 2048, runs_directory=directory, argv=["test"])
    return directory / "A_gbm" / "seed_0" / "policy.zip"


def test_a_saved_policy_becomes_a_comparator(smoke_run):
    """Section 2b: the agent has to look like every other strategy to the evaluation code."""
    policy = load_trained_policy(smoke_run)
    weights = policy(np.array([[0.0, 0.0, 0.8], [1.2, 0.5, 0.4]], dtype=np.float32), 0)
    assert weights.shape == (2, 4)
    assert weights.min() >= 0.0
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


def test_the_wrapped_policy_is_deterministic(smoke_run):
    """The deterministic action is the Dirichlet mean, so two calls must agree exactly."""
    policy = load_trained_policy(smoke_run)
    observations = np.array([[0.4, 0.3, 0.5]], dtype=np.float32)
    np.testing.assert_array_equal(policy(observations, 0), policy(observations, 0))


def test_a_missing_policy_is_reported_clearly(tmp_path):
    """The batch writes runs over several nights; a missing one must not fail obscurely."""
    with pytest.raises(FileNotFoundError, match="no trained policy"):
        load_trained_policy(tmp_path)


def test_the_trained_seeds_are_discovered_and_ordered():
    """Five seeds per configuration (D70), scored individually because the spread is the result."""
    found = trained_policies("A_gbm")
    assert all(name.startswith("agent_seed_") for name in found)
    assert list(found) == sorted(found, key=lambda name: int(name.rsplit("_", 1)[1]))


def test_an_untrained_configuration_yields_no_policies():
    """Not an error: the batch has not reached that configuration yet."""
    assert trained_policies("B_L24") == {}


def test_the_table_is_offered_when_the_solver_has_produced_one():
    """The fifth strategy of thesis 4.3.5, read from the solver output."""
    offered = table_policy("A_gbm")
    assert set(offered) == {"dp_table"}
    assert table_policy("B_L12") == {}


# ======================================================================================
# The post-freeze guard
# ======================================================================================
def test_the_digest_depends_only_on_the_figures():
    """Reordering the policies or the metrics must not move it; changing a number must."""
    table = {"a": {"p": 0.5, "q": 1.0}, "b": {"p": 0.25, "q": 2.0}}
    reordered = {"b": {"q": 2.0, "p": 0.25}, "a": {"q": 1.0, "p": 0.5}}
    assert reference_digest(table) == reference_digest(reordered)

    moved = {"a": {"p": 0.5000000001, "q": 1.0}, "b": {"p": 0.25, "q": 2.0}}
    assert reference_digest(table) != reference_digest(moved)


def test_the_digest_notices_the_smallest_change():
    """One part in 1e-15 of one metric of one policy, which is the point of the guard.

    A change meant to be non-functional that moves a figure at all has not been non-functional,
    and nothing else in the suite would catch it: the unit tests check properties, not the
    particular numbers the memoir quotes.
    """
    table = {"glide_path": {"probability_of_reaching_goal": 0.8832}}
    nudged = {"glide_path": {"probability_of_reaching_goal": 0.8832 + 1e-15}}
    assert reference_digest(table) != reference_digest(nudged)


def test_the_reference_run_reproduces_itself(tmp_path):
    """Recorded on the first run, matched on the second."""
    path = tmp_path / "reference_run.json"
    first = reference_run(seed=SEED, n_bootstrap=200, path=path, argv=["test"])
    assert first["verdict"] == "recorded"
    assert path.is_file()

    second = reference_run(seed=SEED, n_bootstrap=200, path=path, argv=["test"])
    assert second["verdict"] == "match"
    assert second["digest"] == first["digest"]
    assert second["test_set"] == REFERENCE_TEST_SET


def test_a_moved_figure_is_a_mismatch_and_the_reference_is_kept(tmp_path):
    """On a mismatch the recorded reference must survive, or the comparison cannot be repeated."""
    path = tmp_path / "reference_run.json"
    reference_run(seed=SEED, n_bootstrap=200, path=path, argv=["test"])
    recorded = json.loads(path.read_text(encoding="utf-8"))
    tampered = {**recorded, "digest": "0" * 64}
    path.write_text(json.dumps(tampered), encoding="utf-8")

    verdict = reference_run(seed=SEED, n_bootstrap=200, path=path, argv=["test"])
    assert verdict["verdict"] == "MISMATCH"
    assert verdict["previous_digest"] == "0" * 64
    assert json.loads(path.read_text(encoding="utf-8"))["digest"] == "0" * 64


# ======================================================================================
# The discrepancy index
# ======================================================================================
def _synthetic_table(n_wealth: int = 11, horizon: int = 4, degenerate_above: float = 0.8) -> dict:
    """A table whose answer is known: all equity below a threshold, all short Treasury above it."""
    grid = np.linspace(0.0, 2.0, n_wealth)
    policy = np.zeros((n_wealth, horizon, 4))
    policy[grid < degenerate_above, :, 0] = 1.0
    policy[grid >= degenerate_above, :, 2] = 1.0
    n_optimal = np.ones((n_wealth, horizon), dtype=np.int32)
    n_optimal[grid >= degenerate_above, :] = 1771  # the degenerate region, to be excluded
    return {
        "wealth_grid": grid,
        "policy": policy,
        "n_optimal_actions": n_optimal,
        "action_grid": np.array([[1.0, 0, 0, 0], [0.05, 0.95, 0, 0]]),
    }


def test_a_policy_that_matches_the_table_scores_zero():
    """The index is a distance, so agreement must be exactly nothing."""
    problem = load_problem()
    table = _synthetic_table()
    index = discrepancy_index(
        table,
        lambda observations, t: np.tile([1.0, 0.0, 0.0, 0.0], (len(observations), 1)),
        np.zeros((8, 4, 4)),
        problem,
    )
    assert index["mean_distance"] == 0.0
    assert index["median_distance"] == 0.0
    assert index["share_beyond_one_action_step"] == 0.0


def test_a_policy_that_contradicts_the_table_scores_one():
    """Half the L1 distance, so two portfolios sharing nothing are at distance one."""
    problem = load_problem()
    index = discrepancy_index(
        _synthetic_table(),
        lambda observations, t: np.tile([0.0, 1.0, 0.0, 0.0], (len(observations), 1)),
        np.zeros((8, 4, 4)),
        problem,
    )
    assert index["mean_distance"] == pytest.approx(1.0)
    assert index["share_beyond_one_action_step"] == pytest.approx(1.0)


def test_a_policy_halfway_scores_a_half():
    """The scale is interpretable, which is what makes the index worth reporting as a number."""
    problem = load_problem()
    index = discrepancy_index(
        _synthetic_table(),
        lambda observations, t: np.tile([0.5, 0.5, 0.0, 0.0], (len(observations), 1)),
        np.zeros((8, 4, 4)),
        problem,
    )
    assert index["mean_distance"] == pytest.approx(0.5)


def test_degenerate_states_are_excluded_and_counted():
    """Where the table names one of many equally optimal portfolios it says nothing to compare to.

    Most of the real plane is like this, so an index that counted those disagreements would be
    reporting the tie-break rule rather than the agent.
    """
    # Starting at half the goal and gaining fifty per cent a month, the trajectories cross the
    # threshold during the episode and so pass through both regions.
    base = load_problem()
    problem = replace(base, initial_wealth_usd=0.5 * base.goal_usd)
    scenarios = np.full((8, 4, 4), 0.5)
    index = discrepancy_index(
        _synthetic_table(),
        lambda observations, t: np.tile([1.0, 0.0, 0.0, 0.0], (len(observations), 1)),
        scenarios,
        problem,
    )
    assert index["degenerate_share"] > 0.0
    assert index["comparable_states"] > 0
    # Only the states with a unique optimum are compared, and there the policy matches.
    assert index["mean_distance"] == 0.0


def test_an_entirely_degenerate_table_reports_nothing_rather_than_zero():
    """No comparable state is not the same as perfect agreement, and must not read as it."""
    problem = load_problem()
    table = _synthetic_table()
    table["n_optimal_actions"][:] = 1771
    index = discrepancy_index(
        table,
        lambda observations, t: np.tile([1.0, 0.0, 0.0, 0.0], (len(observations), 1)),
        np.zeros((4, 4, 4)),
        problem,
    )
    assert index["comparable_states"] == 0
    assert index["degenerate_share"] == 1.0
    assert np.isnan(index["mean_distance"])


# ======================================================================================
# The paired difference of proportions
# ======================================================================================
def test_the_paired_test_on_counts_worked_out_by_hand():
    """Four scenarios: one where only A succeeds, one where only B does, two concordant."""
    a = np.array([True, False, True, False])
    b = np.array([False, True, True, False])
    result = paired_proportions(a, b)
    assert result["a_only"] == 1
    assert result["b_only"] == 1
    assert result["discordant"] == 2
    assert result["difference"] == 0.0
    assert result["p_value"] == 1.0


def test_only_the_discordant_scenarios_carry_information():
    """Scenarios where both policies agree cancel, which is why the pairing is worth having.

    Adding concordant scenarios moves the denominator and nothing else; the evidence against the
    null is unchanged, and the p-value with it.
    """
    a = np.array([True, False])
    b = np.array([False, True])
    few = paired_proportions(a, b)
    many = paired_proportions(
        np.concatenate([a, np.ones(98, dtype=bool)]), np.concatenate([b, np.ones(98, dtype=bool)])
    )
    assert few["discordant"] == many["discordant"] == 2
    assert few["p_value"] == many["p_value"]
    assert many["n_scenarios"] == 100


def test_a_one_sided_difference_is_detected():
    """Twenty scenarios where only A succeeds and none the other way is not a coin."""
    a = np.concatenate([np.ones(20, dtype=bool), np.zeros(80, dtype=bool)])
    b = np.zeros(100, dtype=bool)
    result = paired_proportions(a, b)
    assert result["a_only"] == 20
    assert result["b_only"] == 0
    assert result["difference"] == pytest.approx(0.20)
    assert result["p_value"] < 1e-5


def test_identical_policies_give_no_evidence():
    """Nothing discordant means nothing to test, and a standard error of zero rather than a crash."""
    success = np.array([True, False, True, True])
    result = paired_proportions(success, success)
    assert result["discordant"] == 0
    assert result["difference"] == 0.0
    assert result["standard_error"] == 0.0
    assert result["p_value"] == 1.0


def test_the_difference_matches_the_two_success_rates():
    """The paired difference must agree with the unpaired one; only the uncertainty differs."""
    rng = np.random.default_rng(SEED)
    a = rng.random(500) < 0.7
    b = rng.random(500) < 0.6
    result = paired_proportions(a, b)
    assert result["difference"] == pytest.approx(a.mean() - b.mean())
    assert result["standard_error"] > 0.0


def test_mismatched_lengths_are_refused():
    """Two policies scored on different scenarios are not paired and must not be treated as if."""
    with pytest.raises(ValueError, match="paired"):
        paired_proportions(np.ones(3, dtype=bool), np.ones(4, dtype=bool))


def test_the_paired_test_reads_the_evaluation_output():
    """The two pieces have to fit together, which is what this closes."""
    problem = load_problem()
    scenarios = generate("B", 256, problem.horizon_months, SEED, block_length=12.0)
    policies = {
        "all_equity": lambda observations, t: np.tile([1.0, 0, 0, 0], (len(observations), 1)),
        "all_short": lambda observations, t: np.tile([0, 0, 1.0, 0], (len(observations), 1)),
    }
    table = evaluate_policies(policies, scenarios, problem=problem, seed=SEED, n_bootstrap=200)

    from portfolio_manager.env import rollout

    success = {
        name: rollout(scenarios, policy, problem=problem)["terminal_wealth"] >= problem.goal_usd
        for name, policy in policies.items()
    }
    result = paired_proportions(success["all_equity"], success["all_short"])
    assert result["difference"] == pytest.approx(
        table["all_equity"]["probability_of_reaching_goal"]
        - table["all_short"]["probability_of_reaching_goal"]
    )


# ======================================================================================
# The validation configuration (D44)
# ======================================================================================
def test_the_validation_universe_is_two_columns():
    """Equity against one riskless column, the rest masked away (D44)."""
    browne = load_browne_config()
    columns, _ = load_matrix()
    params, mask = browne_setup(browne, columns, load_matrix_metadata()["statistics"])
    assert mask.sum() == 2
    assert mask[columns.index(browne["assets"]["risky"])]
    assert mask[columns.index(browne["assets"]["riskless"])]


def test_the_riskless_column_has_no_volatility():
    """Which is what makes the comparison against a one-risky-asset closed form legitimate."""
    browne = load_browne_config()
    columns, _ = load_matrix()
    params, _ = browne_setup(browne, columns, load_matrix_metadata()["statistics"])
    riskless = columns.index(browne["assets"]["riskless"])
    assert params.sigma[riskless, riskless] == 0.0
    assert params.mu[riskless] > 0.0
    # One non-degenerate factor, so the tensor product is over one axis and not four.
    assert int((np.abs(params.factor()).sum(axis=0) > 0).sum()) == 1


def test_the_validation_objective_is_the_indicator():
    """Browne maximises the probability of success; comparing against order 2 would measure the
    difference between objectives rather than the error of the solver."""
    assert load_browne_config()["objective"] == "indicator"
    assert load_browne_config()["contribution_usd"] == 0
    assert load_browne_config()["cost_bp"] == 0


def test_rescaling_is_linear_in_the_step():
    """Log returns and their covariance both add over time, which is what makes the sequence a
    refinement of one problem rather than several different ones."""
    params = GbmParams(assets=("a", "b"), mu=np.array([0.01, 0.02]), sigma=np.diag([0.004, 0.0]))
    quarter = scale_params(params, 4)
    np.testing.assert_allclose(quarter.mu, params.mu / 4.0)
    np.testing.assert_allclose(quarter.sigma, params.sigma / 4.0)
    with pytest.raises(ValueError, match="must be positive"):
        scale_params(params, 0)


def test_the_validation_action_grid_is_one_dimensional():
    """With two columns the portfolio is one number, so a step of 0.01 gives 101 candidates."""
    browne = load_browne_config()
    solution = solve_browne({**browne, "n_wealth": 101, "horizon_months": 2}, steps_per_month=1)
    assert solution.action_grid.shape[0] == 101
    assert solution.quadrature_returns.shape[0] == browne["n_quad"]
    # The masked columns are never held.
    columns, _ = load_matrix()
    for name in ("ust10y", "gold"):
        assert solution.policy[:, :, columns.index(name)].max() == 0.0


def test_the_validation_value_is_a_probability():
    """Under the indicator objective the value is the chance of reaching the goal."""
    browne = load_browne_config()
    solution = solve_browne({**browne, "n_wealth": 201, "horizon_months": 6}, steps_per_month=1)
    assert solution.value.min() >= -1e-12
    assert solution.value.max() <= 1.0 + 1e-12
    assert solution.value_at(0.0, 0) == pytest.approx(0.0, abs=1e-12), "no contributions, so zero stays zero"
    assert solution.value_at(1.5, 0) == pytest.approx(1.0, abs=1e-6), "already past the goal"


def test_the_refinement_sequence_settles(tmp_path):
    """The requirement the memoir states: the error must fall as the step does.

    Without the closed form the error cannot be measured, so what is checked is the necessary
    condition, that the sequence of values converges. The wealth grid is refined with the time
    step; under the indicator objective the terminal value is a step function and refining time
    alone accumulates interpolation error faster than it removes time error.
    """
    browne = load_browne_config()
    # The declared horizon at reduced resolution. A short horizon will not do: with no
    # contributions and half the goal in hand, twelve months makes the value about 8e-05 and the
    # sequence is then pure floating-point noise, which tests nothing.
    reduced = {
        **browne,
        "refinement": [
            {"steps_per_month": 1, "n_wealth": 401},
            {"steps_per_month": 2, "n_wealth": 801},
            {"steps_per_month": 4, "n_wealth": 1601},
        ],
    }
    summary = run_browne_validation(reduced, directory=tmp_path, argv=["test"])
    assert len(summary["levels"]) == 3
    assert summary["converging"], summary["successive_differences"]
    differences = summary["successive_differences"]
    assert abs(differences[-1]) < abs(differences[0]), differences


def test_the_validation_tables_say_they_are_not_validated(tmp_path):
    """The closed form is transcribed from the paper by the author, never written from memory."""
    browne = load_browne_config()
    reduced = {**browne, "horizon_months": 2, "refinement": [{"steps_per_month": 1, "n_wealth": 101}]}
    run_browne_validation(reduced, directory=tmp_path, argv=["test"])
    sidecar = json.loads((tmp_path / "browne_1_per_month.json").read_text(encoding="utf-8"))
    assert sidecar["validation"] == "browne"
    assert "not implemented" in sidecar["closed_form"]
    assert sidecar["solver"]["objective"] == "indicator"


# ======================================================================================
# The extended command line
# ======================================================================================
def test_scoring_the_table_beside_the_comparators(tmp_path):
    """One code path for every strategy, which is what section 2b asks for."""
    table, path, _ = run(
        "A_gbm", 0.0, seed=SEED, which="all", n_bootstrap=200, results_directory=tmp_path
    )
    assert "dp_table" in table
    assert any(name.startswith("agent_") for name in table)
    assert "constant_mix" in table
    assert path.name == "all_A_gbm_0bp.csv"

    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["policies"] == "all"
    assert set(sidecar["policy_names"]) == set(table)


def test_the_tabulated_optimum_beats_the_comparators_on_its_own_objective(tmp_path):
    """An independent check of the solver, from the other end of the pipeline.

    Under the parametric generator at zero cost the table is the optimum for the lower partial
    moment of order 2. Rolling it out over the frozen test set recomputes that quantity by
    simulation rather than by backward induction, through code that shares nothing with the solver
    beyond the transition, so the table coming first is evidence the solver is right.
    """
    table, _, _ = run("A_gbm", 0.0, seed=SEED, which="all", n_bootstrap=200, results_directory=tmp_path)
    rewards = {name: row["mean_terminal_reward"] for name, row in table.items()}
    best = max(rewards, key=lambda name: rewards[name])
    assert best == "dp_table", rewards


def test_an_empty_policy_set_is_reported(tmp_path):
    """Asking for an agent the batch has not trained yet must say so."""
    with pytest.raises(ValueError, match="no policies to score"):
        run("B_L24", 0.0, seed=SEED, which="agent", n_bootstrap=100, results_directory=tmp_path)
