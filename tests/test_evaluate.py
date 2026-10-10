"""Tests of the evaluation protocol, thesis subsection 4.2.4.

Two properties carry the weight here. The metrics code must not know which strategy it is scoring,
or the fairness of the H2 comparison would rest on inspection rather than on construction; and the
bootstrap must resample every policy on the same scenario indices, or the intervals would not be
comparable and the paired test of H2 could not be built from them.
"""

# Imports
# =====================================================================
import csv
import json

import numpy as np
import pytest

from portfolio_manager.baselines import build_baselines, load_baselines
from portfolio_manager.env import load_problem, rollout
from portfolio_manager.evaluate import (
    COLUMNS,
    QUANTILES,
    bootstrap_blocks,
    evaluate_policies,
    metrics,
    run,
    write_table,
)
from portfolio_manager.generator import generate, load_matrix, load_matrix_metadata

# Constants
# =====================================================================
SEED = 1000
SMALL_SCENARIOS = 256
TEST_SET = "B_L12"


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def problem():
    """The base case."""
    return load_problem()


@pytest.fixture(scope="module")
def scenarios(problem) -> np.ndarray:
    """A small mode-B tensor, from the validation band rather than the test one."""
    return generate("B", SMALL_SCENARIOS, problem.horizon_months, SEED, block_length=12.0)


@pytest.fixture(scope="module")
def policies() -> dict:
    """The comparator set."""
    return build_baselines(load_baselines(), statistics=load_matrix_metadata()["statistics"])


@pytest.fixture(scope="module")
def table(policies, scenarios, problem) -> dict:
    """The scored comparator set on the small tensor."""
    return evaluate_policies(policies, scenarios, problem=problem, seed=SEED, n_bootstrap=500)


# ======================================================================================
# Metrics
# ======================================================================================
def test_metrics_on_a_hand_checkable_sample():
    """Every figure of thesis 4.2.4 against one small vector computed by hand."""
    wealth = np.array([30_000.0, 50_000.0, 60_000.0, 90_000.0])
    reward = -np.clip(1.0 - wealth / 60_000.0, 0.0, None) ** 2
    turnover = np.array([1.2, 1.2, 1.2, 1.2])

    computed = metrics(wealth, reward, turnover, goal=60_000.0, horizon=120)
    assert computed["n_scenarios"] == 4
    assert computed["probability_of_reaching_goal"] == 0.5
    assert computed["probability_of_shortfall"] == 0.5
    # Only the two failures count, and they count by how much they failed: (30000 + 10000) / 2.
    assert computed["conditional_expected_shortfall_usd"] == pytest.approx(20_000.0)
    assert computed["terminal_wealth_p50_usd"] == pytest.approx(55_000.0)
    assert computed["mean_terminal_reward"] == pytest.approx(reward.mean())
    assert computed["annualised_turnover"] == pytest.approx(1.2 * 12.0 / 120.0)


def test_reaching_the_goal_exactly_is_not_a_shortfall():
    """``P(W_T >= G)`` is inclusive, so a portfolio landing on the goal has succeeded."""
    computed = metrics(
        np.array([60_000.0]), np.array([0.0]), np.array([0.0]), goal=60_000.0, horizon=120
    )
    assert computed["probability_of_reaching_goal"] == 1.0


def test_the_conditional_shortfall_is_not_a_number_when_nothing_fails():
    """An expectation conditioned on an empty set has no value; reporting zero would be a lie.

    Zero is the one figure a reader would misread as "the failures cost nothing" rather than as
    "there were none".
    """
    computed = metrics(
        np.array([70_000.0, 80_000.0]), np.zeros(2), np.zeros(2), goal=60_000.0, horizon=120
    )
    assert computed["probability_of_reaching_goal"] == 1.0
    assert np.isnan(computed["conditional_expected_shortfall_usd"])


def test_every_declared_quantile_is_reported():
    """The distribution behind the single probability, which thesis 4.2.4 asks for."""
    computed = metrics(
        np.linspace(0.0, 100_000.0, 1001), np.zeros(1001), np.zeros(1001), goal=60_000.0, horizon=120
    )
    for q in QUANTILES:
        assert computed[f"terminal_wealth_p{q}_usd"] == pytest.approx(q * 1000.0, abs=100.0)


# ======================================================================================
# The bootstrap
# ======================================================================================
def test_the_interval_brackets_the_point_estimate(table):
    """A percentile interval around a proportion must contain it."""
    for name, row in table.items():
        assert row["probability_ci_low"] <= row["probability_of_reaching_goal"] <= row["probability_ci_high"], name


def test_the_interval_narrows_with_more_scenarios(policies, problem):
    """The width is the sampling error of the test set, so it must fall as the set grows."""
    widths = []
    for size in (200, 2000):
        scenarios = generate("B", size, problem.horizon_months, SEED, block_length=12.0)
        row = evaluate_policies(
            {"constant_mix": policies["constant_mix"]}, scenarios, problem=problem, seed=SEED, n_bootstrap=500
        )["constant_mix"]
        widths.append(row["probability_ci_high"] - row["probability_ci_low"])
    assert widths[1] < widths[0] / 2.0


def test_the_bootstrap_resamples_whole_scenarios(scenarios):
    """Indices, not values: the pairing across policies lives in the index blocks."""
    rng = np.random.default_rng(SEED)
    blocks = list(bootstrap_blocks(rng, len(scenarios), 500))
    assert sum(len(block) for block in blocks) == 500
    for block in blocks:
        assert block.shape[1] == len(scenarios)
        assert block.min() >= 0 and block.max() < len(scenarios)


def test_policies_are_resampled_on_the_same_worlds(policies, scenarios, problem):
    """Common random numbers across policies.

    Independent draws per policy would leave the intervals incomparable and would discard the
    pairing that the frozen test set exists to provide, which is what the difference of proportions
    of H2 is computed from.
    """
    rng = np.random.default_rng(SEED)
    blocks = np.concatenate(list(bootstrap_blocks(rng, len(scenarios), 500)))

    table = evaluate_policies(policies, scenarios, problem=problem, seed=SEED, n_bootstrap=500)
    for name, policy in policies.items():
        wealth = rollout(scenarios, policy, problem=problem)["terminal_wealth"]
        success = wealth >= problem.goal_usd
        draws = success[blocks].mean(axis=1)
        assert table[name]["probability_ci_low"] == pytest.approx(float(np.quantile(draws, 0.025)))
        assert table[name]["probability_ci_high"] == pytest.approx(float(np.quantile(draws, 0.975)))


def test_the_bootstrap_is_reproducible(policies, scenarios, problem):
    """Same seed, same interval; the figures of the memoir must not move between runs."""
    first = evaluate_policies(policies, scenarios, problem=problem, seed=SEED, n_bootstrap=500)
    again = evaluate_policies(policies, scenarios, problem=problem, seed=SEED, n_bootstrap=500)
    for name in policies:
        assert first[name]["probability_ci_low"] == again[name]["probability_ci_low"]
        assert first[name]["probability_ci_high"] == again[name]["probability_ci_high"]


def test_the_seed_moves_the_resampling_and_not_the_estimate(policies, scenarios, problem):
    """The point estimate is a property of the test set; only the interval is resampled.

    The comparison is made on the index blocks rather than on the reported bounds: with a few
    hundred scenarios a percentile of a proportion is quantised in steps of ``1/n``, so two seeds
    can land on the same bound without the draws being the same at all.
    """
    one = np.concatenate(list(bootstrap_blocks(np.random.default_rng(SEED), len(scenarios), 500)))
    two = np.concatenate(list(bootstrap_blocks(np.random.default_rng(SEED + 1), len(scenarios), 500)))
    assert not np.array_equal(one, two)

    first = evaluate_policies(policies, scenarios, problem=problem, seed=SEED, n_bootstrap=500)
    other = evaluate_policies(policies, scenarios, problem=problem, seed=SEED + 1, n_bootstrap=500)
    for name in policies:
        assert first[name]["probability_of_reaching_goal"] == other[name]["probability_of_reaching_goal"]
        assert first[name]["mean_terminal_reward"] == other[name]["mean_terminal_reward"]


# ======================================================================================
# The protocol knows nothing about the comparators
# ======================================================================================
def test_an_arbitrary_callable_is_scored_the_same_way(scenarios, problem):
    """Section 2b: the metrics code never special-cases a strategy.

    A policy invented here, which no part of the package has heard of, must be scored exactly as
    the comparators are. That is what will let the agent and the table enter the same table later
    without touching this module.
    """

    def invented(observations: np.ndarray, t: int) -> np.ndarray:
        # Spends the first half in equity and then retreats to the short Treasury.
        weights = np.zeros((len(observations), 4))
        weights[:, 0 if t < problem.horizon_months // 2 else 2] = 1.0
        return weights

    table = evaluate_policies({"invented": invented}, scenarios, problem=problem, seed=SEED, n_bootstrap=200)
    assert set(table) == {"invented"}
    assert 0.0 <= table["invented"]["probability_of_reaching_goal"] <= 1.0
    assert table["invented"]["annualised_turnover"] > 0.0


def test_a_state_dependent_policy_is_scored_per_scenario(scenarios, problem):
    """The rollout hands each scenario its own observation, which a comparator never exercises."""
    seen: list[int] = []

    def adaptive(observations: np.ndarray, t: int) -> np.ndarray:
        seen.append(len(observations))
        behind = observations[:, 0] < observations[:, 1]
        weights = np.tile([0.0, 0.0, 1.0, 0.0], (len(observations), 1))
        weights[behind] = [1.0, 0.0, 0.0, 0.0]
        return weights

    evaluate_policies({"adaptive": adaptive}, scenarios, problem=problem, seed=SEED, n_bootstrap=100)
    assert seen == [len(scenarios)] * problem.horizon_months


def test_every_policy_meets_the_same_scenarios(policies, scenarios, problem):
    """Paired by construction: scoring one policy alone must give exactly the same figures."""
    together = evaluate_policies(policies, scenarios, problem=problem, seed=SEED, n_bootstrap=200)
    alone = evaluate_policies(
        {"glide_path": policies["glide_path"]}, scenarios, problem=problem, seed=SEED, n_bootstrap=200
    )
    assert together["glide_path"]["probability_of_reaching_goal"] == alone["glide_path"][
        "probability_of_reaching_goal"
    ]


def test_a_higher_cost_cannot_raise_terminal_wealth(policies, scenarios, problem):
    """The sweep of thesis 4.2.3 must move the figures in the only direction it can."""
    free = evaluate_policies(
        policies, scenarios, problem=problem, cost_rate=0.0, seed=SEED, n_bootstrap=100
    )
    dear = evaluate_policies(
        policies, scenarios, problem=problem, cost_rate=0.0025, seed=SEED, n_bootstrap=100
    )
    for name in policies:
        assert dear[name]["mean_terminal_wealth_usd"] <= free[name]["mean_terminal_wealth_usd"] + 1e-9


def test_a_single_asset_portfolio_never_rebalances(table):
    """The Merton corner holds one column, so its weights cannot drift away from their target.

    Zero turnover there is a property of the corner solution and not a missing cost: the other four
    portfolios all trade, because holding a fixed target across four assets requires it.
    """
    assert table["merton_gamma_2"]["annualised_turnover"] == pytest.approx(0.0, abs=1e-12)
    for name in ("constant_mix", "glide_path", "merton_gamma_5", "equal_weight"):
        assert table[name]["annualised_turnover"] > 0.0, name


# ======================================================================================
# Output
# ======================================================================================
def test_the_table_is_written_with_the_declared_columns(table, tmp_path):
    """One row per policy, in the order thesis 4.2.4 introduces the metrics."""
    path = write_table(table, tmp_path / "heuristics.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["policy"] for row in rows] == list(table)
    assert tuple(rows[0]) == COLUMNS
    assert float(rows[0]["probability_of_reaching_goal"]) == pytest.approx(
        table[rows[0]["policy"]]["probability_of_reaching_goal"]
    )


@pytest.mark.parametrize("cost_bp", [0.0, 10.0])
def test_the_command_line_run_produces_a_documented_table(cost_bp, tmp_path):
    """End to end on the frozen tensor, including the side-car the freeze will be checked against."""
    table, path, _, _, _ = run(TEST_SET, cost_bp, seed=SEED, n_bootstrap=200, results_directory=tmp_path)
    assert path.name == f"heuristics_{TEST_SET}_{cost_bp:g}bp.csv"
    assert path.is_file()

    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["test_set"] == TEST_SET
    assert sidecar["cost_bp"] == cost_bp
    assert sidecar["n_scenarios"] == 20_000
    assert sidecar["frozen_digest_verdict"] in ("match", "skipped")
    assert sidecar["status"].startswith("preliminary")
    # The side-car must pin what produced the figures, so a later run can be told apart from this one.
    assert set(sidecar["config_sha256"]) == {"baselines", "problem", "test_set"}
    assert sidecar["baselines"]["merton"]["gamma_2"]["weight"] == 1.0
    assert sidecar["environment_fingerprint"]["numpy_version"]
    assert len(table) == 5


def test_the_evaluated_tensor_is_the_frozen_one(tmp_path):
    """The recipe stores a procedure, so the tensor is regenerated and reconciled rather than trusted."""
    _, path, _, _, _ = run(TEST_SET, 10.0, seed=SEED, n_bootstrap=100, results_directory=tmp_path)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    columns, matrix = load_matrix()
    assert sidecar["source_matrix_sha256"] == load_matrix_metadata()["sha256"]
    assert sidecar["effective_seed"] == 20260810
    assert sidecar["mean_block_length_months"] == 12
    assert matrix.shape[1] == len(columns)
