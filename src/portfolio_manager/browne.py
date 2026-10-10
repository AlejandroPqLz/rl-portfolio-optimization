"""Closed-form reference of Browne (1999), for the validation of D44.

Transcribed from ``docs/browne_reference.md``, which the author verified against the paper on
10 October 2026. **Nothing here is written from memory**, and nothing here is derived: every
formula is the one that document states, with its equation number, and the only liberties taken
are numerical ones that leave the value unchanged. That constraint is the point of the exercise.
A reference written from recollection would be a second thing produced by this project, and
agreement between two of those would say nothing about either.

What it is for
--------------
The internal stability checks of thesis 4.3.6 bound the discretisation error of the solver and
nothing else: an error in the recursion reproduces itself at any resolution and both of them pass
on a table that is wrong. This module supplies the independent answer. Where the paper's problem
and this one coincide, the backward induction must reproduce a formula it had no part in
producing.

They coincide only in a declared region. The paper's investor trades continuously, may borrow and
may sell short; this one trades monthly on the simplex. The comparison is therefore restricted to
the part of the state space where the optimal control asks for neither leverage nor a short
position, which is what :func:`borrowing_boundary` computes.

Notation
--------
``Phi`` and ``phi`` are the standard normal distribution and density, ``Phi^{-1}`` the quantile
(paper, "Remark on notation", p. 556). All rates are **monthly**: ``r`` is continuously
compounded, as the paper's bond, and ``sigma`` is the monthly volatility.

Sections of the reference document map to this module as follows:
``value``                section 2, equation (3.3)
``optimal_amount``      section 3, equation (3.5)
``optimal_fraction``    section 3, equation (5.2)
``borrowing_boundary``  section 4, Table 1
``parameters_from_gbm`` section 5
``income_value``        section 7, equation (7.4), behind a flag
"""

# Imports
# =====================================================================
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Constants
# =====================================================================
BOUNDARY_BRACKET = 60.0
"""Half-width of the initial bracket for the root of the boundary equation. The inverse Mills
ratio ``phi/Phi`` grows like ``|nu|`` as ``nu`` falls, so a bracket of this width covers every
``tau`` up to roughly its square, far beyond the five that Table 1 reaches."""

BOUNDARY_TOLERANCE = 1e-12

MONTHS_PER_YEAR = 12


# ======================================================================================
# Section 5: the parameters of the paper, from the parameters of the generator
# ======================================================================================
@dataclass(frozen=True)
class BrowneParameters:
    """The three coefficients the closed form needs, in monthly units.

    Attributes:
        r: Continuously compounded monthly rate of the riskless asset, as in ``dB = r B dt``.
        mu: Drift of ``dS/S``, monthly. **Not** the mean log return: see
            :func:`parameters_from_gbm`.
        sigma: Monthly volatility of the risky asset.
    """

    r: float
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError(f"the risky asset must have positive volatility, got {self.sigma}")

    @property
    def premium(self) -> float:
        """``mu - r``, the risk premium per month."""
        return self.mu - self.r

    @property
    def theta(self) -> float:
        """Market price of risk ``(mu - r) / sigma``, equation (2.6) with one stock."""
        return self.premium / self.sigma

    @property
    def satisfies_paper_assumption(self) -> bool:
        """Whether ``mu >= r``, which section 1 records the paper as assuming."""
        return self.premium >= 0.0

    def require_paper_assumption(self) -> None:
        """Fail unless the closed form applies to these coefficients.

        Checked where the formulas are *used* rather than where the numbers are built, because the
        third check of section 6 deliberately moves the drift and has to be able to describe where
        it moved it to, including out of the region the paper covers. A set of coefficients is
        data; whether the paper speaks about it is a separate question with a separate answer.
        """
        if not self.satisfies_paper_assumption:
            raise ValueError(
                f"the paper assumes a non-negative market price of risk, mu >= r (section 1), but "
                f"mu = {self.mu} and r = {self.r}. With a negative premium the optimal control of "
                "equation (3.5) is not the solution to the problem stated, so the closed form "
                "cannot be used as a reference here."
            )


def parameters_from_gbm(
    log_mean: float, log_variance: float, riskless_log_rate: float, *, mean_shift: float = 0.0
) -> BrowneParameters:
    """The mapping of section 5, as a function so that it can be tested rather than assumed.

    The one step that is easy to get wrong, and silently: **the paper's ``mu`` is the drift of
    ``dS/S``, not the mean of the log return**. Under the generator the log gross return of a month
    has mean ``m`` and variance ``s^2``, and the paper's own solution of (4.1),
    ``S_t = S_0 exp{(mu - s^2/2) t + s W_t}`` (p. 559), gives that log return a mean of
    ``mu - s^2/2``. Hence ``mu = m + s^2/2``. Using ``m`` directly would understate the drift by
    half the variance, which for monthly equity is about one part in a thousand of the premium: too
    small to look wrong in a plot and large enough to move the value function.

    The result is returned whether or not the paper's assumption ``mu >= r`` survives the shift;
    :meth:`BrowneParameters.require_paper_assumption` is where that is enforced.

    Args:
        log_mean: ``m``, mean monthly log return of the risky asset.
        log_variance: ``s^2``, variance of that log return.
        riskless_log_rate: ``r``, monthly log return of the zero-volatility asset.
        mean_shift: Added to ``m`` before the conversion. Used by the third check of section 6,
            which perturbs the drift and expects the optimal control not to move.

    Returns:
        The coefficients of the closed form.
    """
    if log_variance <= 0.0:
        raise ValueError(f"the risky asset must have positive variance, got {log_variance}")
    sigma = float(np.sqrt(log_variance))
    return BrowneParameters(
        r=float(riskless_log_rate),
        mu=float(log_mean + mean_shift + log_variance / 2.0),
        sigma=sigma,
    )


# ======================================================================================
# Sections 2 and 3: the value and the optimal control
# ======================================================================================
def relative_wealth(
    wealth: np.ndarray | float, months_remaining: np.ndarray | float, r: float, goal: float = 1.0
) -> np.ndarray:
    """``z = (x / b) e^{r (T - t)}``, equation (5.1).

    Wealth measured against the goal discounted back to the present. It is the only combination of
    wealth, time and the riskless rate the solution depends on, which is why Table 1 is a table in
    one variable.
    """
    return np.asarray(wealth, dtype=np.float64) / goal * np.exp(r * np.asarray(months_remaining, dtype=np.float64))


def value(
    wealth: np.ndarray | float,
    months_remaining: np.ndarray | float,
    parameters: BrowneParameters,
    goal: float = 1.0,
) -> np.ndarray:
    """The optimal value function, section 2, equation (3.3).

    ``V(t, x; b) = Phi( Phi^{-1}(z) + sqrt(theta^2 (T - t)) )`` for ``0 < x < b e^{-r(T-t)}``, and
    one at or above that, where the bond alone reaches the goal and the optimal stock holding is
    zero (Remark 3.1).

    Returns:
        The probability of reaching the goal under the optimal control.
    """
    z = relative_wealth(wealth, months_remaining, parameters.r, goal)
    remaining = np.asarray(months_remaining, dtype=np.float64)
    shift = parameters.theta * np.sqrt(np.maximum(remaining, 0.0))
    # Phi^{-1} is evaluated only where it is finite; the two closed branches are filled in after.
    interior = (z > 0.0) & (z < 1.0)
    result = np.where(z >= 1.0, 1.0, 0.0)
    safe = np.where(interior, z, 0.5)
    return np.where(interior, norm.cdf(norm.ppf(safe) + shift), result)


def optimal_amount(
    wealth: np.ndarray | float,
    months_remaining: np.ndarray | float,
    parameters: BrowneParameters,
    goal: float = 1.0,
) -> np.ndarray:
    """The optimal **amount of money** in the stock, section 3, equation (3.5).

    ``f*_t(x; b) = [1 / (sigma sqrt(T - t))] b e^{-r(T-t)} phi( Phi^{-1}(z) )``.

    The paper notes of this expression that "it is only for this single-stock case that the optimal
    control is in fact independent of the underlying drift parameter mu" (p. 557), which is the
    third check of section 6: ``mu`` appears nowhere on the right.
    """
    z = relative_wealth(wealth, months_remaining, parameters.r, goal)
    remaining = np.asarray(months_remaining, dtype=np.float64)
    interior = (z > 0.0) & (z < 1.0) & (remaining > 0.0)
    safe_z = np.where(interior, z, 0.5)
    safe_time = np.where(remaining > 0.0, remaining, 1.0)
    discounted_goal = goal * np.exp(-parameters.r * safe_time)
    amount = discounted_goal * norm.pdf(norm.ppf(safe_z)) / (parameters.sigma * np.sqrt(safe_time))
    return np.where(interior, amount, 0.0)


def optimal_fraction(
    wealth: np.ndarray | float,
    months_remaining: np.ndarray | float,
    parameters: BrowneParameters,
    goal: float = 1.0,
) -> np.ndarray:
    """The optimal **fraction of wealth** in the stock, section 3, equation (5.2).

    ``w* = phi(nu) / ( Phi(nu) sigma sqrt(T - t) )`` with ``nu = Phi^{-1}(z)``, which is
    :func:`optimal_amount` divided by wealth. It is reported separately because the solver's action
    is a fraction, so this is the quantity the second check of section 6 compares.

    Below the boundary of :func:`borrowing_boundary` this exceeds one, and the solver cannot follow
    it: the comparison is confined to the region where it does not.
    """
    z = relative_wealth(wealth, months_remaining, parameters.r, goal)
    remaining = np.asarray(months_remaining, dtype=np.float64)
    interior = (z > 0.0) & (z < 1.0) & (remaining > 0.0)
    safe_z = np.where(interior, z, 0.5)
    safe_time = np.where(remaining > 0.0, remaining, 1.0)
    fraction = norm.pdf(norm.ppf(safe_z)) / (safe_z * parameters.sigma * np.sqrt(safe_time))
    # At or above the discounted goal the stock holding is zero; at zero wealth the fraction
    # diverges, which is reported as infinity rather than as a quiet nan.
    return np.where(interior, fraction, np.where(z >= 1.0, 0.0, np.inf))


# ======================================================================================
# Section 4: the borrowing boundary
# ======================================================================================
def inverse_mills(nu: np.ndarray | float) -> np.ndarray:
    """``phi(nu) / Phi(nu)``, evaluated so that it survives a very negative argument.

    Both terms underflow together far into the left tail, where the ratio is near ``|nu|`` and
    perfectly well behaved; taking the difference of the logarithms keeps it that way instead of
    returning zero divided by zero.
    """
    nu = np.asarray(nu, dtype=np.float64)
    return np.exp(norm.logpdf(nu) - norm.logcdf(nu))


def borrowing_boundary(tau: float) -> tuple[float, float]:
    """The boundary of the borrowing region, section 4 and Table 1.

    Borrowing takes place where ``w* >= 1``, which by equation (5.2) is where
    ``phi(nu)/Phi(nu) >= sqrt(tau)`` with ``tau = sigma^2 (T - t)``, the risk-adjusted time to
    play. The ratio falls strictly from infinity to zero as ``nu`` rises, so the boundary is the
    unique root of ``phi(nu)/Phi(nu) - sqrt(tau)``, found here rather than tabulated.

    The investor borrows when ``z < z*(tau)`` and does not when ``z > z*(tau)``, with
    ``z* = Phi(nu*)``. The caption of Table 1 prints the inverse of this; the text of p. 564 and
    the tabulated values settle it the other way, as the reference document records.

    Args:
        tau: Risk-adjusted time to play, ``sigma^2 (T - t)``.

    Returns:
        Tuple ``(nu_star, z_star)``. At ``tau = 0`` there is no time left to play and no borrowing
        at any wealth, which is the limit ``(inf, 1)``.
    """
    if tau < 0.0:
        raise ValueError(f"the risk-adjusted time to play cannot be negative, got {tau}")
    if tau == 0.0:
        return float("inf"), 1.0

    target = np.sqrt(tau)

    def excess(nu: float) -> float:
        return float(inverse_mills(nu) - target)

    lower, upper = -BOUNDARY_BRACKET, BOUNDARY_BRACKET
    while excess(lower) < 0.0:
        lower *= 2.0
        if lower < -1e6:
            raise ValueError(f"no boundary found for tau = {tau}")
    nu_star = brentq(excess, lower, upper, xtol=BOUNDARY_TOLERANCE)
    return float(nu_star), float(norm.cdf(nu_star))


def borrows(
    wealth: np.ndarray | float,
    months_remaining: np.ndarray | float,
    parameters: BrowneParameters,
    goal: float = 1.0,
) -> np.ndarray:
    """Whether the optimal control asks for leverage at this state."""
    return np.asarray(optimal_fraction(wealth, months_remaining, parameters, goal)) >= 1.0


# ======================================================================================
# Section 7: exogenous income, Theorem 7.1. Behind a flag (P1 only).
# ======================================================================================
def income_present_value(income: float, horizon: float, r: float) -> float:
    """``integral_0^T e^{-r s} l ds`` for an income paid at a constant rate.

    The discounted value at time zero of the whole income stream, which is what Theorem 7.1
    subtracts from the discounted goal.
    """
    if r == 0.0:
        return float(income * horizon)
    return float(income * (1.0 - np.exp(-r * horizon)) / r)


def kappa(
    wealth: np.ndarray | float, elapsed: np.ndarray | float, income: float, r: float
) -> np.ndarray:
    """``kappa(x, t) = e^{-rt} x - integral_0^t e^{-rs} l(s) ds``, equation (7.2)."""
    elapsed = np.asarray(elapsed, dtype=np.float64)
    received = income * elapsed if r == 0.0 else income * (1.0 - np.exp(-r * elapsed)) / r
    return np.exp(-r * elapsed) * np.asarray(wealth, dtype=np.float64) - received


def income_lambda(goal: float, horizon: float, income: float, r: float) -> float:
    """``Lambda = b e^{-rT} - integral_0^T e^{-rs} l(s) ds``, equation (7.3)."""
    return float(goal * np.exp(-r * horizon) - income_present_value(income, horizon, r))


def income_value(
    wealth: np.ndarray | float,
    elapsed: np.ndarray | float,
    parameters: BrowneParameters,
    *,
    goal: float,
    horizon: float,
    income: float,
) -> np.ndarray:
    """Value function with exogenous income, equation (7.4), single stock.

    ``Psi(t, x; b, T) = Phi( Phi^{-1}( kappa / Lambda ) + sqrt(theta^2 (T - t)) )``.

    Not part of the three mandatory checks: D72 reduced the validation to the value, the policy in
    the no-leverage region and the independence from the drift, and moved this to P1. The runner
    reaches it only behind an explicit flag.
    """
    scale = income_lambda(goal, horizon, income, parameters.r)
    if scale <= 0.0:
        raise ValueError(
            f"Lambda = {scale}: the income alone reaches the goal, so equation (7.4) does not apply "
            "(Remark 7.1, the investment in the stock stops)"
        )
    ratio = np.asarray(kappa(wealth, elapsed, income, parameters.r)) / scale
    remaining = float(horizon) - np.asarray(elapsed, dtype=np.float64)
    shift = parameters.theta * np.sqrt(np.maximum(remaining, 0.0))
    interior = (ratio > 0.0) & (ratio < 1.0)
    safe = np.where(interior, ratio, 0.5)
    return np.where(interior, norm.cdf(norm.ppf(safe) + shift), np.where(ratio >= 1.0, 1.0, 0.0))


def income_amount(
    wealth: np.ndarray | float,
    elapsed: np.ndarray | float,
    parameters: BrowneParameters,
    *,
    goal: float,
    horizon: float,
    income: float,
) -> np.ndarray:
    """Optimal amount in the stock with exogenous income, equation (7.5), single stock.

    ``f^{(l)}_t(x; b) = [1 / (sigma sqrt(T - t))] e^{rt} Lambda phi( Phi^{-1}( kappa / Lambda ) )``.

    By Remark 7.1 the stock holding stops at ``kappa >= Lambda``, where bond and future income
    reach the goal alone (7.6), and at ``kappa <= 0``, where wealth has fallen to what the income
    alone would have earned in the bond (7.7). Both are returned as zero.
    """
    scale = income_lambda(goal, horizon, income, parameters.r)
    if scale <= 0.0:
        raise ValueError(f"Lambda = {scale}: equation (7.5) does not apply (Remark 7.1)")
    elapsed = np.asarray(elapsed, dtype=np.float64)
    ratio = np.asarray(kappa(wealth, elapsed, income, parameters.r)) / scale
    remaining = float(horizon) - elapsed
    interior = (ratio > 0.0) & (ratio < 1.0) & (remaining > 0.0)
    safe_ratio = np.where(interior, ratio, 0.5)
    safe_time = np.where(remaining > 0.0, remaining, 1.0)
    amount = (
        np.exp(parameters.r * elapsed)
        * scale
        * norm.pdf(norm.ppf(safe_ratio))
        / (parameters.sigma * np.sqrt(safe_time))
    )
    return np.where(interior, amount, 0.0)
