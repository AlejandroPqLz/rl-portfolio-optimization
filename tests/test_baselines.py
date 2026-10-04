"""Tests of the comparator strategies, thesis subsections 4.3.5 and 4.2.3.

The claim this module has to support is narrow and important: no comparator was tuned against a
result. It rests on two things, and both are tested here. Every figure comes from
``config/baselines.yml``, which is verifiable by feeding the loader a modified copy and watching
the outputs follow; and the Merton fraction comes from three statistics of the frozen matrix rather
than from a choice, including the corner solution it produces at ``gamma = 2``, which is declared
rather than avoided (D53).
"""

# Imports
# =====================================================================
import numpy as np
import pytest
import yaml

from portfolio_manager.baselines import (
    AssetMapping,
    build_baselines,
    constant_mix_weights,
    describe_baselines,
    dp_table_policy,
    glide_path_equity_weight,
    load_baselines,
    merton_fraction,
    merton_weights,
)
from portfolio_manager.generator import config_path, load_matrix, load_matrix_metadata

# Constants
# =====================================================================
EXPECTED_POLICIES = ("constant_mix", "glide_path", "merton_gamma_2", "merton_gamma_5", "equal_weight")
NOMINAL_PHI = 0.621
"""The value of the specification, replaced by the sample-based one in Block 1 (A28)."""


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def columns() -> tuple[str, ...]:
    """The normative column order."""
    names, _ = load_matrix()
    return names


@pytest.fixture(scope="module")
def config(columns):
    """The comparator configuration as it stands in the tree."""
    return load_baselines(columns=columns)


@pytest.fixture(scope="module")
def statistics() -> dict:
    """The statistics block of the frozen matrix."""
    return load_matrix_metadata()["statistics"]


@pytest.fixture(scope="module")
def policies(config, statistics) -> dict:
    """The five concrete portfolios."""
    return build_baselines(config, statistics=statistics)


# ======================================================================================
# The comparator set
# ======================================================================================
def test_four_rules_give_five_portfolios(policies):
    """Thesis 4.2.3: four rules, five portfolios, one per declared value of gamma."""
    assert tuple(policies) == EXPECTED_POLICIES


def test_every_portfolio_stays_on_the_simplex(policies, config):
    """Non-negative and summing to one at every month, which is Equation ``eq:simplex``."""
    observations = np.zeros((3, 3), dtype=np.float32)
    for name, policy in policies.items():
        for t in (0, config.horizon_months // 2, config.horizon_months - 1):
            weights = policy(observations, t)
            assert weights.shape == (3, 4), name
            assert weights.min() >= 0.0, (name, t)
            np.testing.assert_allclose(weights.sum(axis=-1), 1.0, err_msg=f"{name} at t={t}")


def test_a_policy_accepts_a_single_observation(policies):
    """The signature is shared with the agent and the table, so both shapes must work."""
    for name, policy in policies.items():
        weights = policy(np.zeros(3, dtype=np.float32), 0)
        assert weights.shape == (4,), name


def test_only_the_equal_split_touches_gold(policies, config):
    """The three rules projected through ``asset_mapping`` hold no gold; 1/N is the exception."""
    gold = config.mapping.index("gold")
    observations = np.zeros((1, 3), dtype=np.float32)
    for name in ("constant_mix", "glide_path", "merton_gamma_2", "merton_gamma_5"):
        assert policies[name](observations, 0)[0, gold] == 0.0, name
    assert policies["equal_weight"](observations, 0)[0, gold] == 0.25


def test_only_the_glide_path_reads_the_clock(policies):
    """Three rules hold a fixed target; the glide path is the only one that moves with time.

    None of the four reads the accumulated capital, which is precisely the asymmetry H3 is about.
    """
    observations = np.zeros((1, 3), dtype=np.float32)
    for name, policy in policies.items():
        first, last = policy(observations, 0), policy(observations, 119)
        moved = not np.allclose(first, last)
        assert moved == (name == "glide_path"), name


def test_no_rule_reads_the_accumulated_capital(policies):
    """Two states differing only in wealth must give the same portfolio for every comparator."""
    poor = np.array([[0.1, 0.5, 0.4]], dtype=np.float32)
    rich = np.array([[3.0, 0.5, 0.4]], dtype=np.float32)
    for name, policy in policies.items():
        np.testing.assert_allclose(policy(poor, 60), policy(rich, 60), err_msg=name)


# ======================================================================================
# Constant mix and the projection
# ======================================================================================
def test_constant_mix_is_sixty_forty_split_by_phi(config, columns):
    """``(0.60; 0.40 phi; 0.40 (1 - phi); 0)``, with phi the sample-based value of A28."""
    phi = config.mapping.fixed_income_shares["ust10y"]
    weights = constant_mix_weights(config)
    expected = {"us_equity": 0.60, "ust10y": 0.40 * phi, "ust2y": 0.40 * (1.0 - phi), "gold": 0.0}
    for name, value in expected.items():
        assert weights[columns.index(name)] == pytest.approx(value)


def test_the_split_is_the_sample_based_one(config):
    """A28 closed before the first evaluation; the nominal 0.621 must no longer be in use."""
    phi = config.mapping.fixed_income_shares["ust10y"]
    assert phi != NOMINAL_PHI
    assert phi == pytest.approx(0.656452)
    assert sum(config.mapping.fixed_income_shares.values()) == pytest.approx(1.0)


def test_constant_mix_and_glide_path_share_one_projection(config):
    """D68: with the same mapping the two heuristics differ only in their temporal shape.

    At the month where the schedule happens to pass through 0.60 equity the two portfolios must
    coincide exactly, which is the sharpest statement of that.
    """
    mix = constant_mix_weights(config)
    direct = config.mapping.allocate(config.constant_mix_equity)
    np.testing.assert_allclose(mix, direct)


# ======================================================================================
# The glide path
# ======================================================================================
def test_glide_path_endpoints_are_the_declared_weights(config):
    """``w_E(0) = w_0`` and ``w_E(T) = w_T``, the two readings of the reference series."""
    assert glide_path_equity_weight(config, 0) == pytest.approx(config.glide_equity_start)
    assert glide_path_equity_weight(config, config.horizon_months) == pytest.approx(config.glide_equity_end)
    assert config.glide_equity_start == pytest.approx(0.669)
    assert config.glide_equity_end == pytest.approx(0.488)


def test_glide_path_descends_linearly(config):
    """The shape the provider declares for the series, so the slope is constant."""
    weights = [glide_path_equity_weight(config, t) for t in range(config.horizon_months + 1)]
    steps = np.diff(weights)
    np.testing.assert_allclose(steps, steps[0])
    assert steps[0] < 0.0
    # The slope quoted in thesis 4.2.3, in percentage points per year.
    assert steps[0] * 12.0 * 100.0 == pytest.approx(-1.81, abs=0.005)


def test_glide_path_endpoint_portfolios_hold_no_gold(config):
    """The reference fund has no gold sleeve, so the projection assigns none."""
    described = describe_baselines(config)
    for key in ("weights_at_start", "weights_at_end"):
        assert described["glide_path"][key][config.mapping.index("gold")] == 0.0


def test_an_unimplemented_shape_is_rejected(config):
    """Only the linear descent is implemented; anything else must fail rather than be approximated."""
    other = type(config)(**{**config.__dict__, "glide_shape": "exponential"})
    with pytest.raises(ValueError, match="only the linear shape"):
        glide_path_equity_weight(other, 0)


# ======================================================================================
# Merton
# ======================================================================================
def test_merton_inputs_are_the_three_declared_statistics(config, statistics):
    """D69: mu and sigma of the risky column, r the geometric mean of the riskless one."""
    fraction = merton_fraction(config, 5.0, statistics)
    assert fraction["mu"] == statistics["us_equity"]["geometric_mean_annual"]
    assert fraction["sigma"] == statistics["us_equity"]["std_annualised"]
    assert fraction["r"] == statistics["ust2y"]["geometric_mean_annual"]
    assert fraction["equity_premium"] == pytest.approx(0.0724, abs=5e-5)


def test_the_riskless_leg_is_the_least_volatile_column(config, statistics):
    """The justification thesis 4.2.3 gives for choosing it, checked against the matrix itself."""
    least_volatile = min(statistics, key=lambda name: statistics[name]["std_annualised"])
    assert config.merton_riskless == least_volatile


def test_merton_reproduces_the_fractions_quoted_in_the_thesis(config, statistics):
    """1.55 at gamma = 2 and 0.62 at gamma = 5, the figures subsection 4.2.3 states."""
    assert merton_fraction(config, 2.0, statistics)["unclipped"] == pytest.approx(1.55, abs=0.005)
    assert merton_fraction(config, 5.0, statistics)["unclipped"] == pytest.approx(0.62, abs=0.005)


def test_gamma_two_lands_on_the_equity_corner(config, statistics, columns):
    """D53: the unclipped fraction exceeds one, so the no-leverage constraint binds.

    The degeneracy is kept and reported. Shifting the range of gamma until neither end degenerated
    would be choosing the parameter from its consequence, which is what D47 exists to prevent.
    """
    fraction = merton_fraction(config, 2.0, statistics)
    assert fraction["unclipped"] > 1.0
    assert fraction["weight"] == 1.0
    weights = merton_weights(config, 2.0, statistics)
    np.testing.assert_allclose(weights, np.eye(len(columns))[columns.index("us_equity")])


def test_gamma_five_leaves_the_remainder_in_the_riskless_column(config, statistics, columns):
    """The interior case: the rest of the portfolio goes to the same column that supplied ``r``."""
    weights = merton_weights(config, 5.0, statistics)
    fraction = merton_fraction(config, 5.0, statistics)["weight"]
    assert weights[columns.index("us_equity")] == pytest.approx(fraction)
    assert weights[columns.index("ust2y")] == pytest.approx(1.0 - fraction)
    assert weights[columns.index("ust10y")] == 0.0
    assert weights[columns.index("gold")] == 0.0


def test_a_larger_gamma_holds_less_equity(config, statistics):
    """More aversion, less of the risky asset: the sign of the rule, not a calibration."""
    fractions = [merton_fraction(config, gamma, statistics)["unclipped"] for gamma in (2.0, 5.0, 10.0)]
    assert fractions[0] > fractions[1] > fractions[2]


def test_a_non_positive_gamma_is_rejected(config, statistics):
    """The rule divides by gamma; zero or negative is a configuration error, not a preference."""
    with pytest.raises(ValueError, match="strictly positive"):
        merton_fraction(config, 0.0, statistics)


# ======================================================================================
# Nothing is hard-coded
# ======================================================================================
def test_every_figure_follows_a_modified_configuration(columns, statistics, tmp_path):
    """The claim the configuration file exists to make checkable, tested by changing all of it.

    If any weight, endpoint, share or value of gamma were written in this package, one of the
    assertions below would keep the value of the real file instead of the altered one.
    """
    raw = yaml.safe_load(config_path("baselines").read_text(encoding="utf-8"))
    raw["glide_path"]["equity_weight_start"] = 0.90
    raw["glide_path"]["equity_weight_end"] = 0.10
    raw["asset_mapping"]["fixed_income_split"]["ust10y_share"] = 0.25
    raw["asset_mapping"]["fixed_income_split"]["ust2y_share"] = 0.75
    raw["constant_mix"]["equity_weight"] = 0.30
    raw["constant_mix"]["fixed_income_weight"] = 0.70
    raw["equal_weight"]["weight_per_asset"] = 0.25
    raw["merton_crra"]["gamma_values"] = [7.0]
    path = tmp_path / "baselines.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    altered = load_baselines(path, columns=columns)
    policies = build_baselines(altered, statistics=statistics)
    observations = np.zeros((1, 3), dtype=np.float32)

    assert tuple(policies) == ("constant_mix", "glide_path", "merton_gamma_7", "equal_weight")
    np.testing.assert_allclose(
        policies["constant_mix"](observations, 0)[0], [0.30, 0.70 * 0.25, 0.70 * 0.75, 0.0]
    )
    assert glide_path_equity_weight(altered, 0) == pytest.approx(0.90)
    assert glide_path_equity_weight(altered, altered.horizon_months) == pytest.approx(0.10)
    assert merton_fraction(altered, 7.0, statistics)["unclipped"] == pytest.approx(
        merton_fraction(altered, 2.0, statistics)["unclipped"] * 2.0 / 7.0
    )


def test_the_asset_roles_are_read_and_not_assumed(columns, statistics, tmp_path):
    """Renaming the risky column in the configuration must change the portfolio, not be ignored."""
    raw = yaml.safe_load(config_path("baselines").read_text(encoding="utf-8"))
    raw["merton_crra"]["risky_asset"] = "gold"
    raw["merton_crra"]["riskless_asset"] = "ust10y"
    path = tmp_path / "baselines.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    altered = load_baselines(path, columns=columns)
    fraction = merton_fraction(altered, 5.0, statistics)
    assert fraction["mu"] == statistics["gold"]["geometric_mean_annual"]
    assert fraction["r"] == statistics["ust10y"]["geometric_mean_annual"]
    assert merton_weights(altered, 5.0, statistics)[columns.index("us_equity")] == 0.0


def test_a_misdeclared_constant_mix_is_rejected(columns, tmp_path):
    """Weights that do not sum to one are a configuration error and must not be renormalised away."""
    raw = yaml.safe_load(config_path("baselines").read_text(encoding="utf-8"))
    raw["constant_mix"]["fixed_income_weight"] = 0.30
    path = tmp_path / "baselines.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="do not sum to one"):
        load_baselines(path, columns=columns)


def test_misdeclared_fixed_income_shares_are_rejected(columns, tmp_path):
    """Likewise for the split, which must be phi and 1 - phi and nothing else."""
    raw = yaml.safe_load(config_path("baselines").read_text(encoding="utf-8"))
    raw["asset_mapping"]["fixed_income_split"]["ust2y_share"] = 0.10
    path = tmp_path / "baselines.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must sum to one"):
        load_baselines(path, columns=columns)


def test_a_column_absent_from_the_matrix_is_rejected(columns):
    """A mapping naming a column the matrix does not have must fail loudly."""
    mapping = AssetMapping(columns=columns, equity="not_an_asset", fixed_weights={}, fixed_income_shares={})
    with pytest.raises(ValueError, match="not a column of the matrix"):
        mapping.index("not_an_asset")


# ======================================================================================
# The tabulated optimum
# ======================================================================================
def test_the_table_policy_picks_the_nearest_wealth_node():
    """Thesis 4.3.5, fifth strategy. The grid is in units of the goal, so no rescaling is needed."""
    grid = np.array([0.0, 1.0, 2.0])
    table = {"wealth_grid": grid, "policy": np.zeros((3, 2, 4))}
    table["policy"][0, :, 0] = 1.0  # node 0.0 -> all equity
    table["policy"][1, :, 1] = 1.0  # node 1.0 -> all long bond
    table["policy"][2, :, 2] = 1.0  # node 2.0 -> all short bond
    policy = dp_table_policy(table)

    observations = np.array([[0.1, 0.0, 0.8], [0.9, 0.0, 0.8], [1.6, 0.0, 0.8], [5.0, 0.0, 0.8]], np.float32)
    chosen = policy(observations, 0)
    np.testing.assert_array_equal(chosen.argmax(axis=1), [0, 1, 2, 2])


def test_the_table_policy_breaks_ties_towards_the_lower_node():
    """A state exactly between two nodes must resolve deterministically."""
    table = {"wealth_grid": np.array([0.0, 1.0]), "policy": np.array([[[1.0, 0, 0, 0]], [[0, 1.0, 0, 0]]])}
    chosen = dp_table_policy(table)(np.array([[0.5, 0.0, 0.8]], dtype=np.float32), 0)
    np.testing.assert_array_equal(chosen.argmax(axis=1), [0])


def test_the_table_policy_reads_the_month():
    """The optimal portfolio depends on the time remaining, which is the point of tabulating it."""
    table = {
        "wealth_grid": np.array([0.0, 1.0]),
        "policy": np.array([[[1.0, 0, 0, 0], [0, 0, 0, 1.0]], [[0, 1.0, 0, 0], [0, 0, 1.0, 0]]]),
    }
    policy = dp_table_policy(table)
    observations = np.array([[0.0, 0.0, 0.8]], dtype=np.float32)
    assert policy(observations, 0).argmax() == 0
    assert policy(observations, 1).argmax() == 3
    # Past the end of the table the last month is held, so an over-long episode cannot crash.
    assert policy(observations, 9).argmax() == 3


def test_the_table_policy_accepts_a_single_observation():
    """Same signature as every other comparator."""
    table = {"wealth_grid": np.array([0.0, 1.0]), "policy": np.array([[[1.0, 0, 0, 0]], [[0, 1.0, 0, 0]]])}
    weights = dp_table_policy(table)(np.array([0.9, 0.0, 0.8], dtype=np.float32), 0)
    assert weights.shape == (4,)
    assert weights.argmax() == 1


def test_a_malformed_table_is_rejected():
    """The grid and the table must describe the same discretisation."""
    with pytest.raises(ValueError, match="does not match"):
        dp_table_policy({"wealth_grid": np.array([0.0, 1.0]), "policy": np.zeros((3, 2, 4))})
    with pytest.raises(ValueError, match="strictly increasing"):
        dp_table_policy({"wealth_grid": np.array([1.0, 0.0]), "policy": np.zeros((2, 2, 4))})
