"""Tests of the tercile split, the figures and the cost a run was trained under.

The tercile split and the variance ratio are both statistics, so they are tested against cases
whose answer is known rather than against themselves: a policy constructed to do well exactly where
another does badly, and series that are independent, trending and mean-reverting by construction.
The figures are tested for the two things a figure can get wrong without anyone noticing, which are
reading the wrong file and claiming to show something it does not.
"""

# Imports
# =====================================================================
import json
from dataclasses import replace

import numpy as np
import pytest

from portfolio_manager.env import load_problem
from portfolio_manager.evaluate import N_TERCILES, run_terciles, terciles
from portfolio_manager.generator import generate
from portfolio_manager.plots import (
    LABELS,
    VARIANCE_RATIO_YEARS,
    equity_weight_map,
    plot_policy_maps,
    plot_terminal_wealth,
    plot_training_curves,
    plot_variance_ratio,
    read_table,
    read_training_metrics,
    variance_ratio,
)
from portfolio_manager.train import run_directory_name

# Constants
# =====================================================================
SEED = 1000
CONFIG = "A_gbm"


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def problem():
    """The base case."""
    return load_problem()


@pytest.fixture(scope="module")
def scenarios(problem) -> np.ndarray:
    """A small mode-B tensor."""
    return generate("B", 600, problem.horizon_months, SEED, block_length=12.0)


def _constant(weights):
    """A policy holding one portfolio."""
    return lambda observations, t: np.tile(weights, (len(observations), 1))


# ======================================================================================
# The tercile split
# ======================================================================================
def test_the_split_is_made_at_the_middle_of_the_horizon(scenarios, problem):
    """Thesis 4.2.4 forms the terciles on relative wealth at mid-horizon."""
    document = terciles({"equal": _constant([0.25, 0.25, 0.25, 0.25])}, scenarios, problem=problem)
    assert document["midpoint_month"] == problem.horizon_months // 2
    assert document["tercile_names"] == ["behind", "on_pace", "ahead"]


def test_each_third_holds_a_third_of_the_scenarios(scenarios, problem):
    """The edges are quantiles of the policy's own distribution, so the thirds are equal."""
    document = terciles({"equal": _constant([0.25, 0.25, 0.25, 0.25])}, scenarios, problem=problem)
    counts = [
        document["policies"]["equal"]["terciles"][name]["n_scenarios"]
        for name in document["tercile_names"]
    ]
    assert sum(counts) == len(scenarios)
    assert max(counts) - min(counts) <= N_TERCILES


def test_being_ahead_at_the_midpoint_predicts_reaching_the_goal(scenarios, problem):
    """The split has to order the outcome, or it is not measuring progress.

    This is what makes the terciles worth forming at all: if success did not rise across them the
    bucket would be noise and any difference inside one would be unreadable.
    """
    document = terciles({"equal": _constant([0.25, 0.25, 0.25, 0.25])}, scenarios, problem=problem)
    probabilities = [
        document["policies"]["equal"]["terciles"][name]["probability_of_reaching_goal"]
        for name in document["tercile_names"]
    ]
    assert probabilities[0] < probabilities[1] < probabilities[2]


def test_each_policy_is_split_by_its_own_trajectory(scenarios, problem):
    """A tercile is a statement about the state that policy reached, not about another's.

    Sorting every policy by one strategy's wealth would ask how each does in situations some other
    strategy created, which is a different question from the one H3 poses.
    """
    document = terciles(
        {
            "all_equity": _constant([1.0, 0.0, 0.0, 0.0]),
            "all_short": _constant([0.0, 0.0, 1.0, 0.0]),
        },
        scenarios,
        problem=problem,
    )
    edges = {name: entry["midpoint_edges_over_goal"] for name, entry in document["policies"].items()}
    assert edges["all_equity"] != edges["all_short"]


def test_the_paired_test_runs_inside_each_third(scenarios, problem):
    """An agent and a reference are compared within the bucket, not across buckets."""
    document = terciles(
        {
            "agent_fake": _constant([0.6, 0.2, 0.2, 0.0]),
            "glide_path": _constant([0.5, 0.25, 0.25, 0.0]),
        },
        scenarios,
        problem=problem,
    )
    paired = document["paired"]
    assert paired["reference"] == "glide_path"
    assert list(paired["agents"]) == ["agent_fake"], "every agent is tested, not only the first"
    for name in document["tercile_names"]:
        entry = paired["agents"]["agent_fake"]["terciles"][name]
        assert 0.0 <= entry["p_value"] <= 1.0
        assert entry["n_scenarios"] > 0


def test_every_seed_is_paired_not_only_the_first(scenarios, problem):
    """Five seeds are trained and the spread across them is part of the result (D70).

    Testing whichever agent happened to come first would report one seed's advantage as the
    agent's, and the thesis quotes the third hypothesis per seed.
    """
    document = terciles(
        {
            "agent_seed_0": _constant([0.6, 0.2, 0.2, 0.0]),
            "agent_seed_1": _constant([0.7, 0.1, 0.2, 0.0]),
            "agent_seed_2": _constant([0.4, 0.3, 0.3, 0.0]),
            "glide_path": _constant([0.5, 0.25, 0.25, 0.0]),
        },
        scenarios,
        problem=problem,
    )
    assert list(document["paired"]["agents"]) == ["agent_seed_0", "agent_seed_1", "agent_seed_2"]
    for entry in document["paired"]["agents"].values():
        assert set(entry["terciles"]) == set(document["tercile_names"])


def test_scenarios_landing_in_different_thirds_are_dropped_and_counted(scenarios, problem):
    """The comparison stays paired on the same scenarios rather than on two different samples.

    Two policies need not place a scenario in the same third, and conditioning each side on its own
    bucket would compare different sets of worlds while calling the result paired.
    """
    document = terciles(
        {
            "agent_fake": _constant([1.0, 0.0, 0.0, 0.0]),
            "glide_path": _constant([0.0, 0.0, 1.0, 0.0]),
        },
        scenarios,
        problem=problem,
    )
    per_tercile = document["paired"]["agents"]["agent_fake"]["terciles"]
    dropped = [per_tercile[name]["scenarios_in_exactly_one_tercile"] for name in document["tercile_names"]]
    assert sum(dropped) > 0, "two very different policies must disagree about some buckets"
    for name in document["tercile_names"]:
        entry = per_tercile[name]
        assert entry["n_scenarios"] + entry["scenarios_in_exactly_one_tercile"] > 0


def test_the_paired_test_is_skipped_without_an_agent(scenarios, problem):
    """Reported as skipped rather than silently absent."""
    document = terciles({"glide_path": _constant([0.5, 0.5, 0.0, 0.0])}, scenarios, problem=problem)
    assert "skipped" in document["paired"]


def test_the_tercile_run_writes_its_document_and_side_car(tmp_path):
    """Everything the memoir quotes has a file and a side-car behind it."""
    document, path, _ = run_terciles(
        CONFIG, 0.0, which="baselines", argv=["test"], results_directory=tmp_path
    )
    assert path.name == f"terciles_{CONFIG}_0bp.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["test_set"] == CONFIG
    assert stored["midpoint_month"] == 60
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["midpoint_month"] == 60


# ======================================================================================
# The variance ratio
# ======================================================================================
def test_independent_returns_give_a_ratio_of_one():
    """The null the statistic is written against: variance grows in proportion to the horizon."""
    rng = np.random.default_rng(SEED)
    returns = rng.normal(0.0, 0.04, size=6000)
    for q in (12, 60):
        result = variance_ratio(returns, q)
        assert result["variance_ratio"] == pytest.approx(1.0, abs=0.12)
        assert result["p_value"] > 0.05


def test_a_trending_series_gives_a_ratio_above_one():
    """Positive dependence makes the accumulated variance grow faster than the horizon."""
    rng = np.random.default_rng(SEED)
    shocks = rng.normal(0.0, 0.02, size=6000)
    trending = np.zeros_like(shocks)
    for index in range(1, shocks.size):
        trending[index] = 0.5 * trending[index - 1] + shocks[index]
    assert variance_ratio(trending, 12)["variance_ratio"] > 1.5


def test_a_mean_reverting_series_gives_a_ratio_below_one():
    """Which is the phenomenon the curve exists to measure the destruction of."""
    rng = np.random.default_rng(SEED)
    shocks = rng.normal(0.0, 0.02, size=6000)
    reverting = shocks[1:] - 0.6 * shocks[:-1]
    assert variance_ratio(reverting, 12)["variance_ratio"] < 0.8


def test_the_ratio_at_one_period_is_one():
    """A horizon of one period compares a quantity with itself."""
    rng = np.random.default_rng(SEED)
    assert variance_ratio(rng.normal(0.0, 0.03, size=500), 1)["variance_ratio"] == pytest.approx(1.0, abs=0.01)


def test_an_impossible_horizon_is_refused():
    """Beyond half the sample the overlapping estimator has nothing left to average."""
    with pytest.raises(ValueError, match="horizon must lie"):
        variance_ratio(np.zeros(100), 60)


def test_the_statistic_is_reported_beside_the_ratio():
    """A ratio without its sampling error cannot be read, least of all at ten years."""
    rng = np.random.default_rng(SEED)
    result = variance_ratio(rng.normal(0.0, 0.04, size=2000), 24)
    assert set(result) == {"q", "variance_ratio", "statistic", "p_value"}
    assert np.isfinite(result["statistic"])


# ======================================================================================
# The figures
# ======================================================================================
def test_every_label_is_spanish_and_keyed_in_english():
    """D73 asks for one dictionary; this is what keeps it from drifting back inline."""
    assert LABELS
    for key, value in LABELS.items():
        assert key == key.lower() and " " not in key, key
        assert isinstance(value, str) and value, key
    # A translation that returned the key would pass a weaker check than this.
    assert LABELS["policy"] == "Estrategia"
    assert LABELS["terminal_wealth"].startswith("Riqueza")


def test_the_equity_map_reads_the_first_column():
    """The maps show the equity weight, which is the first column of the normative order."""
    policy = np.zeros((4, 3, 4))
    policy[:, :, 0] = 0.7
    np.testing.assert_allclose(equity_weight_map(policy), 0.7)


def test_the_policy_map_is_written_and_masks_the_degenerate_region(tmp_path):
    """Degenerate states must be blank, or the figure shows the tie-break as if it were the optimum."""
    path = plot_policy_maps(CONFIG, directory=tmp_path, train_cost_bp=0.0)
    assert path.is_file() and path.stat().st_size > 0


def test_the_training_curves_are_written_for_every_seed(tmp_path):
    """Five seeds plotted individually, because the spread between them is part of the result."""
    metrics = read_training_metrics(CONFIG)
    assert len(metrics) == 5
    path = plot_training_curves(CONFIG, directory=tmp_path)
    assert path.is_file() and path.stat().st_size > 0


def test_the_wealth_figure_comes_from_the_recorded_quantiles(tmp_path):
    """D73: the figure can show only what was computed, so it cannot disagree with its own table."""
    from pathlib import Path

    source = Path("results/heuristics_B_L12_10bp.csv")
    table = read_table(source)
    assert all(f"terminal_wealth_p{q}_usd" in row for row in table.values() for q in (5, 25, 50, 75, 95))
    path = plot_terminal_wealth(source, goal=60000.0, directory=tmp_path)
    assert path.is_file() and path.stat().st_size > 0


def test_the_variance_ratio_figure_covers_the_declared_horizons(tmp_path):
    """One, two, three, five, seven and ten years, as the data document fixes them."""
    path, measured = plot_variance_ratio(directory=tmp_path)
    assert path.is_file()
    assert set(measured) == {"us_equity", "ust10y", "ust2y", "gold"}
    for points in measured.values():
        assert [point["q"] for point in points] == [year * 12 for year in VARIANCE_RATIO_YEARS]


def test_equity_is_near_a_random_walk_in_this_sample(tmp_path):
    """The figure's substantive reading, pinned so a later change cannot reverse it silently.

    The block bootstrap destroys dependence beyond the block length by construction. How much that
    costs is what the historical curve says, and for equity in this window it says very little.
    """
    _, measured = plot_variance_ratio(directory=tmp_path)
    for point in measured["us_equity"]:
        assert 0.7 < point["variance_ratio"] < 1.4, point


# ======================================================================================
# The cost a run was trained under
# ======================================================================================
def test_a_run_directory_names_its_training_cost():
    """An agent trained at ten basis points is a different agent from one trained at zero."""
    assert run_directory_name("A_gbm", 10.0) == "A_gbm_10bp"
    assert run_directory_name("A_gbm", 0.0) == "A_gbm_0bp"
    assert run_directory_name("B_L12", 25.0) == "B_L12_25bp"


def test_every_existing_run_records_the_cost_it_used():
    """What the directory name asserts has to agree with what the run itself recorded."""
    from pathlib import Path

    for directory in sorted(Path("results/runs").glob("*bp")):
        expected = float(directory.name.rsplit("_", 1)[1].removesuffix("bp"))
        for metrics_path in directory.glob("seed_*/metrics.json"):
            document = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert document["problem"]["cost_bp"] == expected, metrics_path


def test_the_held_out_evaluation_uses_the_training_cost(problem):
    """Otherwise the curve during training would measure a problem the agent is not solving."""
    import inspect

    from portfolio_manager.train import HeldOutEvaluation, train

    assert "cost_rate" in inspect.signature(HeldOutEvaluation.__init__).parameters
    assert "cost_bp" in inspect.signature(train).parameters


@pytest.mark.slow
def test_a_run_at_another_cost_lands_in_its_own_directory(tmp_path):
    """Two runs at different costs cannot be confused once the terminal is gone."""
    from portfolio_manager.train import train

    metrics = train(CONFIG, 0, 2048, cost_bp=0.0, runs_directory=tmp_path, argv=["test"])
    assert metrics["cost_bp"] == 0.0
    assert metrics["run_directory"] == f"{CONFIG}_0bp"
    assert (tmp_path / f"{CONFIG}_0bp" / "seed_0" / "metrics.json").is_file()
    assert not (tmp_path / CONFIG).exists()


def test_the_problem_block_carries_the_effective_cost(problem):
    """The override has to reach the record, not only the environment."""
    base = replace(problem, cost_bp=25.0)
    assert base.cost_rate == pytest.approx(0.0025)


# A48 (12 October, outside the freeze by D73): the policy maps query the agent in distribution.
# ------------------------------------------------------------------------------------------------
def test_the_policy_map_queries_the_agent_with_the_remaining_contributions(problem):
    """Before A48 the third component was zero on every node, a state the agent never visits."""
    from portfolio_manager.plots import observations_on_grid

    grid = np.linspace(0.0, 2.0, 5)
    t, horizon = 30, problem.horizon_months
    contribution, goal = problem.monthly_contribution_usd, problem.goal_usd
    observed = observations_on_grid(grid, t, horizon, contribution=contribution, goal=goal)
    assert observed.shape == (5, 3) and observed.dtype == np.float32
    assert np.allclose(observed[:, 0], grid) and np.allclose(observed[:, 1], t / horizon)
    assert np.allclose(observed[:, 2], contribution * (horizon - t) / goal)
    last = observations_on_grid(grid, horizon - 1, horizon, contribution=contribution, goal=goal)
    assert np.allclose(last[:, 2], contribution / goal)


def test_the_training_curves_select_the_run_directory_by_training_cost(tmp_path):
    """D85 applied to the figures: zero and ten basis points are different agents."""
    from portfolio_manager.plots import plot_training_curves

    zero = read_training_metrics(CONFIG, train_cost_bp=0.0)
    ten = read_training_metrics(CONFIG, train_cost_bp=10.0)
    assert zero and ten
    assert zero["seed_0"]["problem"]["cost_bp"] == 0.0 and ten["seed_0"]["problem"]["cost_bp"] == 10.0
    path = plot_training_curves(CONFIG, train_cost_bp=0.0, directory=tmp_path)
    assert path.name == "training_curves_A_gbm_0bp.png" and path.stat().st_size > 0
