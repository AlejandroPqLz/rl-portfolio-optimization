"""Fixed-income reconstruction: Treasury constant maturity yields to monthly returns.

The two Treasury files hold *yields*, not returns. A yield states the terms on which a new issue is placed today;
it says nothing about what a holder of an older bond earned.

Turning one into the other is what this module does. Return of month ``t``, simple and in decimal form:

    r_t = y_{t-1}/12  -  D(y_{t-1})*(y_t - y_{t-1})  +  0.5*C(y_{t-1})*(y_t - y_{t-1})**2

The first term is the coupon accrued over the month. The second is the price move along the slope of the price-yield
relation. The third corrects for its curvature: the slope softens as yields rise and steepens as they fall, which
favours the holder in both directions, so the correction never subtracts. Both sensitivities are evaluated at the
start-of-period yield, because they describe the bond actually held when the month begins.

Sensitivities are recomputed every month from the observed yield under a par-bond assumption. That assumption is imposed
by the source rather than chosen: constant maturity rates are *par* yields, so the coupon of the notional instrument
equals the observed yield and both sensitivities follow. Fixing them at a constant would not be equivalent,
as the modified duration of the ten-year leg runs from 9.74 at yields of 0.5 % to 6.50 at yields of 9 %, and both
extremes occur inside the window.
"""

# Imports
# =====================================================================
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Constants
# =====================================================================
CMT_COUPON_FREQUENCY: int = 2
"""Coupons per year assumed for the notional bond. Treasury constant maturity rates are bond-equivalent yields for
securities paying semiannual interest."""

MINIMUM_YIELD: float = 1e-6
"""Below this level the par-bond closed form is numerically meaningless; the guard exists because a
percent-versus-decimal error produces yields of this magnitude and must fail loudly instead of propagating."""

FRED_MISSING_MARKER: str = "."


@dataclass(frozen=True)
class BondSpec:
    """Notional constant-maturity bond behind one yield column.

    Attributes:
        name: Output column name. English, like all code in this project.
        filename: Raw file under ``data/raw/``.
        maturity_years: Constant maturity of the notional bond. Together with the coupon
            frequency it fixes the payment calendar that gets discounted, and that is the
            only input the sensitivities need beyond the observed yield.
    """

    name: str
    filename: str
    maturity_years: float

    @classmethod
    def from_mapping(cls, name: str, values: dict[str, object]):  # -> "BondSpec":
        """Build a spec from the configuration file.

        Args:
            name: Column name of the leg, e.g. ``ust10y``.
            values: The leg's block of ``config/dataset.yml``, with ``filename`` and ``maturity_years``.

        Returns:
            The bond specification.
        """
        return cls(
            name=name,
            filename=str(values["filename"]),
            maturity_years=float(values["maturity_years"]),
        )


UST10Y = BondSpec(name="ust10y", filename="ust10y_dgs10.csv", maturity_years=10.0)
UST2Y = BondSpec(name="ust2y", filename="ust2y_dgs2.csv", maturity_years=2.0)


# ======================================================================================
# Sensitivities of a par bond
# ======================================================================================
def par_bond_sensitivities(
    yields: np.ndarray | float,
    maturity_years: float,
    freq: int = CMT_COUPON_FREQUENCY,
) -> tuple[np.ndarray, np.ndarray]:
    """Modified duration and convexity of a par bond, from its yield.

    Both are derivatives of the price with respect to the annual yield:
        modified duration = -(1 / P) * dP/dy
        convexity         =  (1 / P) * d2P/dy2

    Args:
        yields: Annual nominal yield in decimal form (``0.045`` for 4.5 %).
        maturity_years: Maturity of the notional bond, in years.
        freq: Coupon payments per year.

    Returns:
        Tuple ``(modified_duration, convexity)``: duration in years, convexity in years
        squared. Scalar input gives scalar output.
    """
    # Yield checks and reshaping to (m, 1) for broadcasting ------
    y = np.atleast_1d(np.asarray(yields, dtype=np.float64))
    if y.ndim > 1:
        raise ValueError("par_bond_sensitivities accepts a scalar or a one-dimensional array")
    if np.any(y <= MINIMUM_YIELD):
        raise ValueError(
            "par_bond sensitivities require strictly positive yields in decimal form; "
            f"minimum received: {float(np.min(y)):.6g}. Check the percent-to-decimal conversion."
        )

    n_periods = int(round(maturity_years * freq))
    periods = np.arange(1, n_periods + 1, dtype=np.float64)  # shape (n,)
    periodic = (y / freq)[:, None]  # shape (m, 1): yield per coupon period

    cashflows = np.tile(100.0 * periodic, (1, n_periods))  # par bond: coupon rate equals yield
    cashflows[:, -1] += 100.0  # redemption of the face value
    discount = (1.0 + periodic) ** periods  # shape (m, n)

    price = (cashflows / discount).sum(axis=1)
    macaulay_periods = (periods * cashflows / discount).sum(axis=1) / price
    modified = macaulay_periods / (1.0 + periodic[:, 0]) / freq

    second_derivative = (periods * (periods + 1.0) * cashflows / (discount * (1.0 + periodic) ** 2)).sum(axis=1)
    convexity = second_derivative / price / freq**2

    if np.ndim(yields) == 0:
        return modified[0], convexity[0]
    return modified, convexity


def _par_bond_price(
    yield_now: float,
    maturity_years: float,
    coupon_rate: float,
    freq: int = CMT_COUPON_FREQUENCY,
) -> float:
    """Price of a 100-face bond with the given coupon rate, valued at the given yield."""
    n_periods = int(round(maturity_years * freq))
    periods = np.arange(1, n_periods + 1, dtype=np.float64)
    cashflows = np.full(n_periods, 100.0 * coupon_rate / freq, dtype=np.float64)
    cashflows[-1] += 100.0
    return float(np.sum(cashflows / (1.0 + yield_now / freq) ** periods))


# ======================================================================================
# Reading the raw yield files
# ======================================================================================
def read_fred_daily(path: Path) -> pd.Series:
    """Read a raw daily yield file, in percent, without transforming it.

    Args:
        path: Path to the raw file under ``data/raw/``.

    Returns:
        Series of daily yields in percent, indexed by observation date and sorted.
    """
    frame = pd.read_csv(path)
    date_column, value_column = frame.columns[0], frame.columns[1]
    values = pd.to_numeric(
        frame[value_column].astype(str).str.strip().replace(FRED_MISSING_MARKER, np.nan),
        errors="coerce",
    )
    series = pd.Series(
        values.to_numpy(dtype=np.float64),
        index=pd.to_datetime(frame[date_column]),
        name=value_column,
    )
    return series.dropna().sort_index()


def month_end_observations(daily_yields: pd.Series) -> pd.DataFrame:
    """Select the last available observation of each calendar month.

    The selection is on the last observation actually present, not on the last calendar business day:
    a market holiday on the 31st must not open a gap. The day actually used is kept because the provenance appendix
    has to state it without recomputing anything.

    Args:
        daily_yields: Daily series in percent, already free of missing values.

    Returns:
        Frame indexed by month end, with columns ``observation_date`` and ``yield_pct``.
    """
    grouped = daily_yields.groupby(pd.Grouper(freq="ME"))
    frame = pd.DataFrame(
        {
            "observation_date": grouped.apply(lambda s: s.index[-1] if len(s) else pd.NaT),
            "yield_pct": grouped.last(),
        }
    )
    if frame["yield_pct"].isna().any():
        empty = frame.index[frame["yield_pct"].isna()].strftime("%Y-%m").tolist()
        raise ValueError(f"months with no observation in the source series: {empty}")
    frame.index.name = "month_end"
    return frame


# ======================================================================================
# Reconstruction
# ======================================================================================
def bond_total_return(month_end_yields_pct: pd.Series, spec: BondSpec) -> pd.Series:
    """Reconstruct monthly simple total returns from a month-end yield series.

    Args:
        month_end_yields_pct: Month-end yields in percent per annum.
        spec: Notional bond behind the series.

    Returns:
        Simple monthly returns in decimal form, indexed by the month of the return. One element shorter than
        the input: the first month has no prior yield and therefore no return.
    """
    if len(month_end_yields_pct) < 2:
        raise ValueError("at least two month-end yields are required to produce one return")

    y = month_end_yields_pct.to_numpy(dtype=np.float64) / 100.0  # the only unit conversion
    y_prev, y_now = y[:-1], y[1:]
    duration, convexity = par_bond_sensitivities(y_prev, spec.maturity_years)

    delta = y_now - y_prev
    accrual = y_prev / 12.0
    price_effect = -duration * delta
    convexity_effect = 0.5 * convexity * delta**2

    return pd.Series(
        accrual + price_effect + convexity_effect,
        index=month_end_yields_pct.index[1:],
        name=spec.name,
        dtype=np.float64,
    )


def return_components(month_end_yields_pct: pd.Series, spec: BondSpec) -> pd.DataFrame:
    """Same computation as :func:`bond_total_return`, with the three terms kept apart.

    The split between income and price change is what lets the text state how much of the fixed-income return was
    coupon and how much was revaluation, and how little the convexity term contributes in a normal month.

    Args:
        month_end_yields_pct: Month-end yields in percent per annum.
        spec: Notional bond behind the series.

    Returns:
        DataFrame that represents the return decomposition, indexed by the month of the return.
    """
    y = month_end_yields_pct.to_numpy(dtype=np.float64) / 100.0
    y_prev, y_now = y[:-1], y[1:]
    duration, convexity = par_bond_sensitivities(y_prev, spec.maturity_years)
    delta = y_now - y_prev
    return pd.DataFrame(
        {
            "yield_prev": y_prev,
            "yield_now": y_now,
            "modified_duration": duration,
            "convexity": convexity,
            "accrual": y_prev / 12.0,
            "price_effect": -duration * delta,
            "convexity_effect": 0.5 * convexity * delta**2,
            "total_return": y_prev / 12.0 - duration * delta + 0.5 * convexity * delta**2,
        },
        index=month_end_yields_pct.index[1:],
    )


def load_bond_returns(path: Path, spec: BondSpec) -> pd.Series:
    """Raw daily yield file to monthly simple returns, indexed by month.

    Args:
        path: Path to the daily yield file.
        spec: Notional bond behind the series.

    Returns:
        Series of monthly simple returns.
    """
    monthly = month_end_observations(read_fred_daily(path))
    returns = bond_total_return(monthly["yield_pct"], spec)
    returns.index = returns.index.to_period("M")
    return returns


# ======================================================================================
# Mandatory scale check
# ======================================================================================
def scale_check(spec: BondSpec, shock_bp: float = 100.0, base_yield_pct: float = 4.0) -> dict[str, float]:
    """Price effect of an instantaneous yield shock, approximated and exact.

    It detects a unit error. Yields arrive from the provider in percent and the formula needs them in decimal form.
    It also bounds the approximation. The exact figure comes from repricing the bond by discounting every one of its
    cashflows at the new yield, so the gap between the two columns is the cost of the three-term formula, measured
    rather than asserted: under two basis points for a 100 basis point move.

    Args:
        spec: Notional bond.
        shock_bp: Instantaneous parallel shock, in basis points.
        base_yield_pct: Starting yield in percent.

    Returns:
        Mapping with the approximate and exact price effects in percent, and the two
        sensitivities implied by the starting yield.
    """
    delta = shock_bp / 10_000.0
    y0 = base_yield_pct / 100.0

    duration, convexity = par_bond_sensitivities(y0, spec.maturity_years)
    approximate = -float(duration) * delta + 0.5 * float(convexity) * delta**2
    exact = _par_bond_price(y0 + delta, spec.maturity_years, y0) / 100.0 - 1.0

    return {
        "shock_bp": shock_bp,
        "base_yield_pct": base_yield_pct,
        "price_effect_approximate_pct": approximate * 100.0,
        "price_effect_exact_pct": exact * 100.0,
        "approximation_error_bp": (approximate - exact) * 10_000.0,
        "modified_duration": float(duration),
        "convexity": float(convexity),
    }
