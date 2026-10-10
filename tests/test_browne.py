"""Tests of the closed form of Browne (1999), transcribed in ``docs/browne_reference.md``.

The reference exists to be an answer this project did not produce, so the tests are written against
things outside it: the twenty-nine rows of Table 1 as the paper prints them, the single value the
reference document quotes for the parameters of this experiment, and the internal identities that
tie its own equations together. A formula that agreed only with itself would be no reference at
all.
"""

# Imports
# =====================================================================
import numpy as np
import pytest
import yaml

from portfolio_manager.browne import (
    BrowneParameters,
    borrowing_boundary,
    borrows,
    income_amount,
    income_lambda,
    income_present_value,
    income_value,
    inverse_mills,
    kappa,
    optimal_amount,
    optimal_fraction,
    parameters_from_gbm,
    relative_wealth,
    value,
)
from portfolio_manager.generator import config_path, load_gbm_params

# Constants
# =====================================================================
TABLE_1 = (
    (0.001, 2.26, 0.99),
    (0.05, 1.19, 0.88),
    (0.10, 0.93, 0.82),
    (0.15, 0.75, 0.77),
    (0.20, 0.63, 0.73),
    (0.25, 0.52, 0.70),
    (0.30, 0.43, 0.67),
    (0.35, 0.35, 0.64),
    (0.40, 0.27, 0.61),
    (0.45, 0.21, 0.58),
    (0.50, 0.15, 0.56),
    (0.55, 0.09, 0.54),
    (0.60, 0.04, 0.51),
    (0.65, -0.01, 0.49),
    (0.70, -0.06, 0.48),
    (0.75, -0.11, 0.46),
    (0.80, -0.15, 0.44),
    (0.85, -0.19, 0.43),
    (0.90, -0.23, 0.41),
    (0.95, -0.27, 0.39),
    (1.00, -0.30, 0.38),
    (1.5, -0.61, 0.27),
    (2.0, -0.86, 0.19),
    (2.5, -1.07, 0.14),
    (3.0, -1.25, 0.10),
    (3.5, -1.42, 0.08),
    (4.0, -1.57, 0.06),
    (4.5, -1.71, 0.04),
    (5.0, -1.84, 0.03),
)
"""Table 1 of the paper, p. 565, transcribed in section 4 of the reference document."""

TABLE_TOLERANCE = 0.01
HORIZON_MONTHS = 120


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def parameters() -> BrowneParameters:
    """The coefficients of this experiment, through the mapping of section 5."""
    gbm = load_gbm_params()
    return parameters_from_gbm(float(gbm.mu[0]), float(gbm.sigma[0, 0]), float(gbm.mu[2]))


# ======================================================================================
# Section 4 and Table 1
# ======================================================================================
@pytest.mark.parametrize(("tau", "nu_star", "z_star"), TABLE_1)
def test_every_row_of_table_1_is_reproduced(tau, nu_star, z_star):
    """The twenty-nine rows of the paper's own table, each from the root of its own equation.

    Nothing is tabulated in the code: the boundary is found by solving
    ``phi(nu)/Phi(nu) = sqrt(tau)``, so agreement with the printed table is agreement with the
    paper and not with a copy of it.
    """
    computed_nu, computed_z = borrowing_boundary(tau)
    assert abs(computed_nu - nu_star) < TABLE_TOLERANCE, f"nu*({tau})"
    assert abs(computed_z - z_star) < TABLE_TOLERANCE, f"z*({tau})"


def test_the_table_is_complete():
    """Twenty-nine rows, as the reference document records."""
    assert len(TABLE_1) == 29


def test_the_boundary_of_this_experiment(parameters):
    """``tau = 0.237`` and ``z* = 0.707`` for this universe and horizon, section 4.

    The one figure in the reference document computed from the parameters of this project rather
    than from the paper, and the one thesis 3.2.3 quotes as "about 70 %".
    """
    tau = parameters.sigma**2 * HORIZON_MONTHS
    assert tau == pytest.approx(0.237, abs=0.001)
    _, z_star = borrowing_boundary(tau)
    assert z_star == pytest.approx(0.707, abs=0.001)


def test_the_boundary_falls_as_the_time_to_play_rises():
    """More risk-adjusted time to play, less wealth needed before borrowing stops being optimal."""
    boundaries = [borrowing_boundary(tau)[1] for tau in (0.05, 0.5, 1.0, 2.0, 5.0)]
    assert all(second < first for first, second in zip(boundaries, boundaries[1:], strict=False))


def test_no_time_left_means_no_borrowing():
    """At ``tau = 0`` there is nothing left to play for and the region is empty."""
    nu_star, z_star = borrowing_boundary(0.0)
    assert nu_star == float("inf")
    assert z_star == 1.0


def test_a_negative_time_to_play_is_refused():
    """``tau`` is a variance times a time; neither can be negative."""
    with pytest.raises(ValueError, match="cannot be negative"):
        borrowing_boundary(-1.0)


def test_the_inverse_mills_ratio_survives_the_far_left_tail():
    """Both terms underflow together there, and the ratio is near ``|nu|`` and perfectly finite.

    Taking the difference of the logarithms is what keeps the root finder from meeting a zero
    divided by a zero at large ``tau``.
    """
    assert np.isfinite(inverse_mills(-30.0))
    assert inverse_mills(-30.0) == pytest.approx(30.0, rel=0.01)
    assert inverse_mills(0.0) == pytest.approx(2.0 * np.exp(-0.0) / np.sqrt(2 * np.pi), rel=1e-12)


def test_the_boundary_is_where_the_optimal_fraction_reaches_one(parameters):
    """The definition behind Table 1: borrowing is ``w* >= 1`` and the table is its boundary.

    Computed two independent ways, from the root of the boundary equation and from equation (5.2)
    evaluated at the wealth that root implies, which is the identity the table rests on.
    """
    months = 60.0
    tau = parameters.sigma**2 * months
    _, z_star = borrowing_boundary(tau)
    wealth = z_star * np.exp(-parameters.r * months)
    assert float(optimal_fraction(wealth, months, parameters)) == pytest.approx(1.0, abs=1e-6)
    assert bool(borrows(wealth * 0.99, months, parameters))
    assert not bool(borrows(wealth * 1.01, months, parameters))


# ======================================================================================
# Section 5: the parameter mapping
# ======================================================================================
def test_the_drift_adds_half_the_variance_back():
    """``mu = m + s^2 / 2``, the step of section 5 that is easy to get wrong and silent when it is.

    The paper's ``mu`` is the drift of ``dS/S``; the log return of that process has mean
    ``mu - s^2/2``. Reading ``m`` as ``mu`` would understate the drift by half the variance.
    """
    computed = parameters_from_gbm(0.008, 0.002, 0.003)
    assert computed.mu == pytest.approx(0.008 + 0.002 / 2.0)
    assert computed.sigma == pytest.approx(np.sqrt(0.002))
    assert computed.r == 0.003


def test_the_mapping_is_in_monthly_units(parameters):
    """Everything the closed form sees is per month, as the generator's parameters are."""
    gbm = load_gbm_params()
    assert parameters.sigma == pytest.approx(np.sqrt(float(gbm.sigma[0, 0])))
    assert parameters.sigma == pytest.approx(0.04442, abs=1e-5)
    assert parameters.r == pytest.approx(float(gbm.mu[2]))
    assert parameters.mu == pytest.approx(float(gbm.mu[0]) + float(gbm.sigma[0, 0]) / 2.0)


def test_the_market_price_of_risk(parameters):
    """``theta = (mu - r) / sigma``, equation (2.6) with one stock."""
    assert parameters.theta == pytest.approx((parameters.mu - parameters.r) / parameters.sigma)
    assert parameters.premium == pytest.approx(parameters.mu - parameters.r)
    assert parameters.theta > 0.0


def test_the_drift_shift_moves_only_the_drift():
    """The third check perturbs ``m`` and holds the volatility and the rate."""
    base = parameters_from_gbm(0.008, 0.002, 0.003)
    shifted = parameters_from_gbm(0.008, 0.002, 0.003, mean_shift=0.01)
    assert shifted.mu == pytest.approx(base.mu + 0.01)
    assert shifted.sigma == base.sigma
    assert shifted.r == base.r


def test_coefficients_outside_the_paper_can_be_described_but_not_used():
    """The third check moves the drift below the rate, and has to be able to say so.

    A set of coefficients is data; whether the paper speaks about it is a separate question, which
    is why the assumption is enforced where the formulas are used rather than where the numbers are
    built.
    """
    outside = parameters_from_gbm(0.001, 0.002, 0.05)
    assert outside.premium < 0.0
    assert not outside.satisfies_paper_assumption
    with pytest.raises(ValueError, match="non-negative market price of risk"):
        outside.require_paper_assumption()

    inside = parameters_from_gbm(0.008, 0.002, 0.003)
    assert inside.satisfies_paper_assumption
    inside.require_paper_assumption()


def test_a_riskless_risky_asset_is_refused():
    """The closed form divides by the volatility throughout."""
    with pytest.raises(ValueError, match="positive variance"):
        parameters_from_gbm(0.008, 0.0, 0.003)
    with pytest.raises(ValueError, match="positive volatility"):
        BrowneParameters(r=0.003, mu=0.008, sigma=0.0)


# ======================================================================================
# Sections 2 and 3
# ======================================================================================
def test_the_value_is_a_probability(parameters):
    """It is the chance of reaching the goal, so it lives in the unit interval."""
    wealth = np.linspace(0.0, 1.5, 200)
    computed = value(wealth, 60.0, parameters)
    assert computed.min() >= 0.0
    assert computed.max() <= 1.0


def test_the_value_rises_with_wealth(parameters):
    """More money cannot make the goal less likely.

    Strictly, below the discounted goal. At or above it the value is flat at one, because the bond
    alone already reaches the target and more wealth buys nothing (Remark 3.1).
    """
    months = 60.0
    discounted_goal = float(np.exp(-parameters.r * months))
    below = np.linspace(0.01, discounted_goal * 0.999, 200)
    assert np.all(np.diff(value(below, months, parameters)) > 0.0)
    above = np.linspace(discounted_goal, discounted_goal * 1.5, 20)
    np.testing.assert_array_equal(value(above, months, parameters), 1.0)


def test_above_the_discounted_goal_the_bond_suffices(parameters):
    """Remark 3.1: the value is one and the optimal stock holding is zero."""
    months = 60.0
    discounted_goal = np.exp(-parameters.r * months)
    for wealth in (discounted_goal, discounted_goal * 1.2):
        assert float(value(wealth, months, parameters)) == 1.0
        assert float(optimal_amount(wealth, months, parameters)) == 0.0
        assert float(optimal_fraction(wealth, months, parameters)) == 0.0


def test_the_fraction_is_the_amount_over_the_wealth(parameters):
    """Equation (5.2) against equation (3.5), which is the identity relating the two."""
    wealth = np.linspace(0.05, 0.9, 50)
    months = 48.0
    np.testing.assert_allclose(
        optimal_fraction(wealth, months, parameters),
        optimal_amount(wealth, months, parameters) / wealth,
        rtol=1e-12,
    )


def test_the_optimal_amount_does_not_involve_the_drift(parameters):
    """The property the third check of section 6 exists to test, in the closed form itself.

    The paper states of equation (3.5) that "it is only for this single-stock case that the optimal
    control is in fact independent of the underlying drift parameter mu" (p. 557). Here it is
    exact: changing the drift leaves the control untouched and moves the value.
    """
    other = BrowneParameters(r=parameters.r, mu=parameters.mu * 2.0, sigma=parameters.sigma)
    wealth = np.linspace(0.05, 0.9, 50)
    np.testing.assert_array_equal(optimal_amount(wealth, 48.0, parameters), optimal_amount(wealth, 48.0, other))
    assert not np.allclose(value(wealth, 48.0, parameters), value(wealth, 48.0, other))


def test_the_relative_wealth_is_the_only_state(parameters):
    """Wealth, time and the rate enter the solution only through ``z``, equation (5.1).

    Which is why Table 1 is a table in one variable: two states sharing ``z`` and ``tau`` share
    their fraction.
    """
    months_a, months_b = 40.0, 80.0
    z = 0.8
    wealth_a = z * np.exp(-parameters.r * months_a)
    wealth_b = z * np.exp(-parameters.r * months_b)
    assert float(relative_wealth(wealth_a, months_a, parameters.r)) == pytest.approx(z)
    assert float(relative_wealth(wealth_b, months_b, parameters.r)) == pytest.approx(z)
    # Same z, different tau, so the fractions scale with the square root of the time.
    ratio = float(optimal_fraction(wealth_a, months_a, parameters)) / float(
        optimal_fraction(wealth_b, months_b, parameters)
    )
    assert ratio == pytest.approx(np.sqrt(months_b / months_a), rel=1e-12)


def test_the_value_is_the_stated_composition(parameters):
    """``V = Phi( Phi^{-1}(z) + theta sqrt(T - t) )``, equation (3.3), written out."""
    from scipy.stats import norm

    months, wealth = 36.0, 0.6
    z = float(relative_wealth(wealth, months, parameters.r))
    expected = norm.cdf(norm.ppf(z) + parameters.theta * np.sqrt(months))
    assert float(value(wealth, months, parameters)) == pytest.approx(expected, rel=1e-12)


# ======================================================================================
# Section 7, Theorem 7.1
# ======================================================================================
def test_theorem_7_1_reduces_to_the_no_income_case(parameters):
    """With no income, (7.4) and (7.5) must be (3.3) and (3.5) exactly.

    The sharpest check available on the income case without a second reference: at ``l = 0`` the
    definitions (7.2) and (7.3) collapse to ``kappa/Lambda = z``, and the two pairs of equations
    have to agree to the last bit rather than merely closely.
    """
    wealth = np.array([0.2, 0.4, 0.6, 0.8])
    horizon, elapsed = 120.0, 30.0
    # To the last bit but one: the two routes reach kappa/Lambda = z by different arithmetic,
    # e^{-rt} x / (b e^{-rT}) against x e^{r(T-t)} / b, which can differ in the final unit.
    np.testing.assert_allclose(
        value(wealth, horizon - elapsed, parameters),
        income_value(wealth, elapsed, parameters, goal=1.0, horizon=horizon, income=0.0),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        optimal_amount(wealth, horizon - elapsed, parameters),
        income_amount(wealth, elapsed, parameters, goal=1.0, horizon=horizon, income=0.0),
        rtol=0.0,
        atol=1e-15,
    )


def test_the_present_value_of_the_income(parameters):
    """``integral_0^T e^{-rs} l ds``, with the zero-rate limit handled rather than divided by."""
    assert income_present_value(400.0, 10.0, 0.0) == pytest.approx(4000.0)
    r = 0.01
    assert income_present_value(400.0, 10.0, r) == pytest.approx(400.0 * (1 - np.exp(-r * 10)) / r)
    assert income_present_value(400.0, 10.0, r) < 4000.0, "discounting cannot raise the value"


def test_kappa_and_lambda(parameters):
    """Equations (7.2) and (7.3): wealth and goal, both net of the income stream."""
    r, income, horizon = parameters.r, 0.01, 120.0
    assert float(kappa(0.5, 0.0, income, r)) == pytest.approx(0.5), "nothing received at time zero"
    assert income_lambda(1.0, horizon, income, r) == pytest.approx(
        np.exp(-r * horizon) - income_present_value(income, horizon, r)
    )


def test_an_income_that_reaches_the_goal_alone_is_refused(parameters):
    """Remark 7.1: at ``Lambda <= 0`` the stock holding stops and (7.4) does not apply."""
    with pytest.raises(ValueError, match="Remark 7.1|income alone"):
        income_value(0.5, 0.0, parameters, goal=1.0, horizon=120.0, income=1.0)


def test_the_stock_holding_stops_at_both_bounds(parameters):
    """Remark 7.1, equations (7.6) and (7.7): nothing is held above ``Lambda`` or below zero."""
    horizon, elapsed, income = 120.0, 24.0, 0.001
    scale = income_lambda(1.0, horizon, income, parameters.r)
    received = income_present_value(income, elapsed, parameters.r)
    # kappa >= Lambda: bond and future income reach the goal alone.
    rich = (scale + received) * np.exp(parameters.r * elapsed) * 1.01
    assert float(income_amount(rich, elapsed, parameters, goal=1.0, horizon=horizon, income=income)) == 0.0
    assert float(income_value(rich, elapsed, parameters, goal=1.0, horizon=horizon, income=income)) == 1.0
    # kappa <= 0: wealth has fallen to what the income alone would have earned in the bond.
    poor = received * np.exp(parameters.r * elapsed) * 0.99
    assert float(income_amount(poor, elapsed, parameters, goal=1.0, horizon=horizon, income=income)) == 0.0


# ======================================================================================
# The tolerances, declared before the run
# ======================================================================================
@pytest.fixture(scope="module")
def browne_config() -> dict:
    """The validation configuration as it stands in the tree."""
    return yaml.safe_load(config_path("validation_browne").read_text(encoding="utf-8"))


def test_the_tolerances_are_the_ones_section_6_suggests(browne_config):
    """Written into the configuration before the first run, which is the whole of D47.

    A tolerance chosen after seeing the error it is meant to bound describes the error rather than
    bounding it, so these are pinned here against the figures the reference document suggests.
    """
    checks = browne_config["checks"]
    assert checks["margin"] == 0.05
    assert checks["value_tolerance"] == 0.02
    assert checks["policy_grid_steps"] == 2
    assert checks["policy_grid_steps"] * browne_config["action_step"] == pytest.approx(0.02)


def test_the_independence_check_is_declared_in_full(browne_config):
    """Shift, tolerance, required agreement and the level it is scored at."""
    settings = browne_config["checks"]["mu_independence"]
    assert settings["delta_in_monthly_sigma"] == 1.0
    assert settings["agreement_grid_steps"] == 1
    assert settings["minimum_agreement"] == 0.95
    assert settings["refinement"] == "finest"


def test_the_refinement_pairs_are_those_of_d79(browne_config):
    """``{steps per month, wealth nodes}`` = (1, 801), (2, 1601), (4, 3201)."""
    pairs = [(level["steps_per_month"], level["n_wealth"]) for level in browne_config["refinement"]]
    assert pairs == [(1, 801), (2, 1601), (4, 3201)]


def test_theorem_7_1_is_disabled(browne_config):
    """P1 rather than part of the validation (D72): the runner needs the flag and this together."""
    assert browne_config["theorem_7_1"]["enabled"] is False


def test_the_parameters_come_from_the_generator(browne_config):
    """Section 5: the closed form and the solver describe the same market by construction."""
    assert browne_config["parameter_source"] == "config/gbm_params.yml"
    assert browne_config["objective"] == "indicator"
    assert browne_config["contribution_usd"] == 0
    assert browne_config["cost_bp"] == 0


# --------------------------------------------------------------------------------------------
# The leverage variant, declared before it is run
# --------------------------------------------------------------------------------------------


def test_the_leverage_block_is_the_one_asked_for(browne_config):
    """The second variant lives in this file alone and leaves the experiment alone.

    A leverage grid is meaningless for the thesis portfolio, where the simplex is a constraint of
    the problem and not of the numerics. It exists here because the paper's investor may borrow,
    and comparing against a control the solver cannot express measures the feasible set instead of
    the solution.
    """
    leverage = browne_config["leverage"]
    assert leverage["enabled"] is True
    assert leverage["max_equity_weight"] == 3.0
    assert leverage["grid_step"] == 0.02


def test_the_leverage_tolerances_are_declared_before_the_run(browne_config):
    """Written into the configuration before anything is solved, so the run cannot choose them."""
    checks = browne_config["leverage"]["checks"]
    assert checks["value_tolerance"] == 0.03
    assert checks["policy_grid_steps"] == 3
    assert checks["policy_grid_steps"] * browne_config["leverage"]["grid_step"] == pytest.approx(0.06)
    assert checks["require_monotone_decrease"] is True
    assert checks["report_signed_mean"] is True
    assert checks["report_value_loss"] is True


def test_the_leverage_independence_check_is_declared_in_full(browne_config):
    """A perturbation proportional to the premium, so that ``mu >= r`` survives both directions.

    The no-leverage check moves the drift by one monthly standard deviation, which is several
    times the premium: the downward move crosses into ``mu < r``, outside the paper's assumption.
    Here the step is a quarter of the premium and the assumption is verified on both sides.
    """
    independence = browne_config["leverage"]["checks"]["mu_independence"]
    assert independence["delta_as_fraction_of_premium"] == 0.25
    assert independence["agreement_grid_steps"] == 2
    assert independence["require_paper_assumption_both_ways"] is True


def test_the_no_leverage_configuration_is_untouched(browne_config):
    """The first variant's three tolerances are the ones its verdicts were recorded against."""
    checks = browne_config["checks"]
    assert checks["margin"] == 0.05
    assert checks["value_tolerance"] == 0.02
    assert checks["policy_grid_steps"] == 2
    assert checks["mu_independence"]["delta_in_monthly_sigma"] == 1.0
    assert checks["mu_independence"]["minimum_agreement"] == 0.95
    assert checks["mu_independence"]["agreement_grid_steps"] == 1
