"""The comparator strategies, thesis subsections 4.3.5 and 4.2.3.

Contract
--------
Every comparator is a callable ``policy(observations, t) -> weights`` with observations shaped
``(..., 3)`` as in Equation ``eq:estado`` and weights shaped ``(..., N)`` on the simplex. That is
the same signature the learned policy and the dynamic programming table expose, so ``evaluate.py``
scores all of them through one code path and never asks which strategy it is holding.

Four rules and five concrete portfolios (thesis 4.2.3): constant mix, glide path, Merton at each of
the two declared values of gamma, and the equal split. A fifth strategy, the optimal table, joins
them through :func:`dp_table_policy` once the solver has produced one.

**No figure in this module is written in code.** Every weight, endpoint, share and value of gamma is
read from ``config/baselines.yml``, which carries the provenance of each one inline and the field
``calibrated: false``. That file was written before this module existed, so the claim that no
comparator was tuned against a result is checkable by inspecting the history of a single file; a
literal here would silently break that claim. The three statistics behind the Merton fraction come
from ``data/returns_matrix.json``, which is the frozen matrix.

Decisions involved
------------------
D53 (gamma stays at {2, 5} and the corner solution is declared rather than avoided), D67 (phi names
the fixed-income split), D68 (the constant mix shares the projection of the glide path; the equal
split is 0.25 per column), D69 (the inputs of the Merton fraction).
"""

# Imports
# =====================================================================
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from portfolio_manager.generator import config_path, load_matrix, load_matrix_metadata

# Types
# =====================================================================
Policy = Callable[[np.ndarray, int], np.ndarray]
"""A comparator: observations ``(..., 3)`` and the month index, to weights ``(..., N)``."""

# Constants
# =====================================================================
EQUITY_SLEEVE = "equity"
"""Value marking the column that carries the whole equity sleeve in ``asset_mapping``."""

SHARE_SUFFIX = "_share"
"""Suffix of the keys of ``fixed_income_split`` that name a column and its share of the sleeve."""

WEIGHT_TOLERANCE = 1e-9
"""Weights must sum to one to within this; anything larger is a misdeclared configuration."""


# ======================================================================================
# Projection of a strategy onto the four-asset universe
# ======================================================================================
@dataclass(frozen=True)
class AssetMapping:
    """How a rule stated in terms of sleeves becomes weights on the four columns.

    The reference lifecycle fund holds domestic and foreign equity, aggregate bonds, hedged foreign
    bonds and inflation-linked securities; this universe holds four columns in dollars. The
    projection is an author's decision, declared in thesis 4.2.3, and the constant mix uses the very
    same one so that the two heuristics differ only in their temporal shape (D68).

    Attributes:
        columns: The normative column order of the matrix.
        equity: Column carrying the whole equity sleeve.
        fixed_weights: Columns held at a declared constant, namely gold at zero.
        fixed_income_shares: Split of the fixed-income sleeve across the Treasury columns,
            ``phi`` and ``1 - phi``.
    """

    columns: tuple[str, ...]
    equity: str
    fixed_weights: dict[str, float]
    fixed_income_shares: dict[str, float]

    def index(self, name: str) -> int:
        """Position of a column in the normative order.

        Args:
            name: Column name, as in the header of ``data/returns_matrix.csv``.

        Returns:
            Zero-based index of the column.

        Raises:
            ValueError: If the name is not a column of the matrix.
        """
        if name not in self.columns:
            raise ValueError(f"{name!r} is not a column of the matrix; the header is {list(self.columns)}")
        return self.columns.index(name)

    def allocate(self, equity_weight: float) -> np.ndarray:
        """Weights for a rule that states only how much equity it wants.

        The remainder, after the columns held at a declared constant, goes to the fixed-income
        sleeve and is split there by ``phi``.

        Args:
            equity_weight: Share of the portfolio in the equity column.

        Returns:
            ``(N,)`` weights on the simplex.
        """
        weights = np.zeros(len(self.columns), dtype=np.float64)
        weights[self.index(self.equity)] = equity_weight
        for name, weight in self.fixed_weights.items():
            weights[self.index(name)] = weight
        remainder = 1.0 - equity_weight - sum(self.fixed_weights.values())
        for name, share in self.fixed_income_shares.items():
            weights[self.index(name)] = remainder * share
        _assert_on_simplex(weights, f"equity weight {equity_weight}")
        return weights


def _assert_on_simplex(weights: np.ndarray, context: str) -> None:
    """Fail loudly on a portfolio that is not a portfolio."""
    if weights.min() < -WEIGHT_TOLERANCE or abs(weights.sum() - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"the weights for {context} are not on the simplex: {weights.tolist()}")


@dataclass(frozen=True)
class BaselinesConfig:
    """Everything the comparators need, read from ``config/baselines.yml``.

    Attributes:
        horizon_months: ``T``, which the glide path needs to know where it is.
        mapping: Projection onto the four columns.
        glide_equity_start: Equity weight ten years from the target date.
        glide_equity_end: Equity weight at the target date.
        glide_shape: Declared shape of the descent; only ``linear`` is implemented.
        constant_mix_equity: Equity weight of the constant mix.
        constant_mix_fixed_income: Its fixed-income weight.
        equal_weight: Weight per column of the equal split.
        merton_gammas: The declared values of the risk aversion coefficient.
        merton_risky: Column treated as the risky asset.
        merton_riskless: Column supplying the riskless rate and holding the remainder.
    """

    horizon_months: int
    mapping: AssetMapping
    glide_equity_start: float
    glide_equity_end: float
    glide_shape: str
    constant_mix_equity: float
    constant_mix_fixed_income: float
    equal_weight: float
    merton_gammas: tuple[float, ...]
    merton_risky: str
    merton_riskless: str


def load_baselines(path: Path | None = None, columns: tuple[str, ...] | None = None) -> BaselinesConfig:
    """Read the comparator configuration.

    The asset roles are read rather than assumed: the equity column is the one ``asset_mapping``
    marks as carrying the equity sleeve, the columns held at a constant are the numeric entries
    beside it, and the fixed-income columns are the ``*_share`` keys of the split. Renaming a column
    in the configuration therefore changes the output instead of being silently ignored.

    Args:
        path: Path to the file; resolved from the configuration directory when omitted.
        columns: Normative column order; read from the matrix header when omitted.

    Returns:
        The configuration.
    """
    path = path or config_path("baselines")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: no comparator parameter has a default in code")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if columns is None:
        columns, _ = load_matrix()

    asset_mapping = raw["asset_mapping"]
    equity = [name for name, value in asset_mapping.items() if value == EQUITY_SLEEVE]
    if len(equity) != 1:
        raise ValueError(f"expected exactly one column marked {EQUITY_SLEEVE!r} in asset_mapping, found {equity}")
    fixed_weights = {
        name: float(value) for name, value in asset_mapping.items() if isinstance(value, int | float)
    }
    split = asset_mapping["fixed_income_split"]
    shares = {
        name[: -len(SHARE_SUFFIX)]: float(value)
        for name, value in split.items()
        if name.endswith(SHARE_SUFFIX) and isinstance(value, int | float)
    }
    if not shares or abs(sum(shares.values()) - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"the fixed-income shares must sum to one, got {shares}")

    mapping = AssetMapping(
        columns=tuple(columns),
        equity=equity[0],
        fixed_weights=fixed_weights,
        fixed_income_shares=shares,
    )

    constant_mix = raw["constant_mix"]
    equity_weight = float(constant_mix["equity_weight"])
    fixed_income_weight = float(constant_mix["fixed_income_weight"])
    if abs(equity_weight + fixed_income_weight + sum(fixed_weights.values()) - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(
            f"the constant mix declares {equity_weight} equity and {fixed_income_weight} fixed income "
            f"beside {fixed_weights}, which do not sum to one"
        )

    merton = raw["merton_crra"]
    glide = raw["glide_path"]
    return BaselinesConfig(
        horizon_months=int(raw["horizon_months"]),
        mapping=mapping,
        glide_equity_start=float(glide["equity_weight_start"]),
        glide_equity_end=float(glide["equity_weight_end"]),
        glide_shape=str(glide["shape"]),
        constant_mix_equity=equity_weight,
        constant_mix_fixed_income=fixed_income_weight,
        equal_weight=float(raw["equal_weight"]["weight_per_asset"]),
        merton_gammas=tuple(float(value) for value in merton["gamma_values"]),
        merton_risky=str(merton["risky_asset"]),
        merton_riskless=str(merton["riskless_asset"]),
    )


# ======================================================================================
# The rules
# ======================================================================================
def _as_policy(weights_at: Callable[[int], np.ndarray]) -> Policy:
    """Turn a function of the month alone into a policy over a batch of observations.

    Three of the four rules ignore the state entirely and the fourth looks only at the clock, which
    is exactly the point of comparing them against an agent that also sees the accumulated capital.
    """

    def policy(observations: np.ndarray, t: int) -> np.ndarray:
        weights = weights_at(t)
        observations = np.asarray(observations)
        if observations.ndim == 1:
            return weights
        return np.broadcast_to(weights, (*observations.shape[:-1], weights.size))

    return policy


def constant_mix_weights(config: BaselinesConfig) -> np.ndarray:
    """The 60/40 portfolio, ``(0.60; 0.40 phi; 0.40 (1 - phi); 0)``.

    Args:
        config: Comparator configuration from ``config/baselines.yml``.

    Returns:
        ``(N,)`` weights on the simplex, in the normative column order.
    """
    return config.mapping.allocate(config.constant_mix_equity)


def constant_mix_policy(config: BaselinesConfig) -> Policy:
    """Hold the same target every month, rebalancing back to it.

    A fixed target is not a strategy without trades. The effective weights move every month because
    the four assets return differently, and pulling the portfolio back to the target incurs exactly
    the turnover the agent is charged for.

    Args:
        config: Comparator configuration from ``config/baselines.yml``.

    Returns:
        A ``policy(observations, t) -> weights`` callable that ignores both arguments.
    """
    return _as_policy(lambda t: constant_mix_weights(config))  # noqa: ARG005 - the clock is not read


def glide_path_equity_weight(config: BaselinesConfig, t: int, horizon: int | None = None) -> float:
    """``w_E(t) = w_0 + (w_T - w_0) t / T``, the declining schedule of thesis 4.2.3.

    Args:
        config: Comparator configuration from ``config/baselines.yml``; only the linear shape is accepted.
        t: Month, from 0 to ``T``.
        horizon: ``T`` in months; ``config.horizon_months`` when omitted.

    Returns:
        Share of the portfolio in the equity column at month ``t``.

    Raises:
        ValueError: If the configuration declares a shape other than ``linear``.
    """
    if config.glide_shape != "linear":
        raise ValueError(f"only the linear shape is implemented, the configuration declares {config.glide_shape!r}")
    horizon = int(horizon if horizon is not None else config.horizon_months)
    return config.glide_equity_start + (config.glide_equity_end - config.glide_equity_start) * (t / horizon)


def glide_path_policy(config: BaselinesConfig, horizon: int | None = None) -> Policy:
    """Reduce equity linearly as the target date approaches.

    The shape comes from two funds of one documented lifecycle series, ten years apart and read on
    the same date, so neither endpoint was chosen here. Its theoretical justification is mean
    reversion, which the block bootstrap destroys beyond the block length: the comparison is
    therefore biased in favour of the agent, and that bias is declared rather than corrected.

    Args:
        config: Comparator configuration from ``config/baselines.yml``.
        horizon: ``T`` in months; ``config.horizon_months`` when omitted.

    Returns:
        A ``policy(observations, t) -> weights`` callable that reads only the month ``t``.
    """
    return _as_policy(lambda t: config.mapping.allocate(glide_path_equity_weight(config, t, horizon)))


def merton_fraction(
    config: BaselinesConfig, gamma: float, statistics: dict[str, Any] | None = None
) -> dict[str, float]:
    """The Merton fraction ``w* = (mu - r) / (gamma sigma^2)`` and its clipped value.

    Computed once from three figures of the frozen matrix, named in thesis 4.2.3 and fixed by D69:
    the annual geometric mean and the annualised volatility of the risky column, and the annual
    geometric mean of the riskless one. The risk-free series of the French file is deliberately not
    used, because the one-month bill is not part of the universe and deciding with one rate while
    investing the remainder in another asset would make the comparator incoherent.

    Both values are returned. At ``gamma = 2`` the unclipped fraction exceeds one: the Merton
    investor would borrow to hold more equity and the no-leverage constraint of Equation
    ``eq:simplex`` leaves it at the corner. That degeneracy is reported rather than avoided, because
    moving the range of gamma to escape it would be choosing the parameter from its consequence
    (D53).

    Args:
        config: Comparator configuration, naming the risky and riskless columns.
        gamma: Coefficient of relative risk aversion.
        statistics: The ``statistics`` block of ``returns_matrix.json``; read from disk when omitted.

    Returns:
        Mapping with ``mu``, ``sigma``, ``r``, ``equity_premium``, ``unclipped`` and ``weight``.
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma must be strictly positive, got {gamma}")
    statistics = statistics if statistics is not None else load_matrix_metadata()["statistics"]
    mu = float(statistics[config.merton_risky]["geometric_mean_annual"])
    sigma = float(statistics[config.merton_risky]["std_annualised"])
    rate = float(statistics[config.merton_riskless]["geometric_mean_annual"])
    unclipped = (mu - rate) / (gamma * sigma**2)
    return {
        "mu": mu,
        "sigma": sigma,
        "r": rate,
        "equity_premium": mu - rate,
        "unclipped": unclipped,
        "weight": float(np.clip(unclipped, 0.0, 1.0)),
    }


def merton_weights(
    config: BaselinesConfig, gamma: float, statistics: dict[str, Any] | None = None
) -> np.ndarray:
    """The Merton portfolio: the clipped fraction in equity, the remainder in the riskless column.

    Args:
        config: Comparator configuration from ``config/baselines.yml``.
        gamma: Coefficient of relative risk aversion.
        statistics: The ``statistics`` block of ``returns_matrix.json``; read from disk when omitted.

    Returns:
        ``(N,)`` weights on the simplex, in the normative column order.
    """
    mapping = config.mapping
    weight = merton_fraction(config, gamma, statistics)["weight"]
    weights = np.zeros(len(mapping.columns), dtype=np.float64)
    weights[mapping.index(config.merton_risky)] = weight
    weights[mapping.index(config.merton_riskless)] += 1.0 - weight
    _assert_on_simplex(weights, f"Merton with gamma = {gamma}")
    return weights


def merton_policy(
    config: BaselinesConfig, gamma: float, statistics: dict[str, Any] | None = None
) -> Policy:
    """Hold the Merton fraction, constant over the horizon.

    Args:
        config: Comparator configuration from ``config/baselines.yml``.
        gamma: Coefficient of relative risk aversion.
        statistics: The ``statistics`` block of ``returns_matrix.json``; read from disk when omitted.

    Returns:
        A ``policy(observations, t) -> weights`` callable returning the same weights every month.
    """
    weights = merton_weights(config, gamma, statistics)
    return _as_policy(lambda t: weights)  # noqa: ARG005 - constant over the horizon


def equal_weight_policy(config: BaselinesConfig) -> Policy:
    """The equal split, 0.25 per column.

    The only comparator with a direct exposure to gold, and the only one estimating nothing at all,
    which is precisely why it is hard to beat.

    Args:
        config: Comparator configuration from ``config/baselines.yml`` (``equal_weight.weight_per_asset``).

    Returns:
        A ``policy(observations, t) -> weights`` callable returning the same weights every month.
    """
    weights = np.full(len(config.mapping.columns), config.equal_weight, dtype=np.float64)
    _assert_on_simplex(weights, "the equal split")
    return _as_policy(lambda t: weights)  # noqa: ARG005 - uses no market information


def dp_table_policy(table: Any) -> Policy:
    """Read a portfolio off the tabulated optimal policy, thesis 4.3.5, fifth strategy.

    The table is defined on a grid of normalised wealth, which is the first component of the
    observation, so the lookup needs no rescaling. For a state between two nodes the nearer one is
    used; the table is the optimum of the problem it was built for, and under the parametric
    generator at zero cost that is the problem being evaluated. Applied to resampled scenarios or
    with costs, it is a strategy computed for a neighbouring problem, and it is reported as such.

    Args:
        table: Anything indexable with ``wealth_grid`` ``(n_wealth,)`` in units of the goal and
            ``policy`` ``(n_wealth, T, N)``, which is the layout ``dp_solver.py`` writes.

    Returns:
        The policy.
    """
    grid = np.asarray(table["wealth_grid"], dtype=np.float64)
    weights = np.asarray(table["policy"], dtype=np.float64)
    if grid.ndim != 1 or weights.ndim != 3 or weights.shape[0] != grid.size:
        raise ValueError(
            f"a wealth grid of {grid.shape} does not match a policy table of {weights.shape}; "
            "expected (n_wealth,) and (n_wealth, T, n_assets)"
        )
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("the wealth grid must be strictly increasing for the nearest-node lookup")
    horizon = weights.shape[1]

    def policy(observations: np.ndarray, t: int) -> np.ndarray:
        observations = np.asarray(observations)
        relative_wealth = np.atleast_1d(observations[..., 0])
        # Nearest node on a sorted grid: the insertion point and its predecessor are the only
        # candidates, so this costs a binary search rather than a scan of the whole grid.
        right = np.clip(np.searchsorted(grid, relative_wealth), 1, grid.size - 1)
        left = right - 1
        nearest = np.where(
            np.abs(relative_wealth - grid[left]) <= np.abs(grid[right] - relative_wealth), left, right
        )
        chosen = weights[nearest, min(t, horizon - 1)]
        return chosen[0] if observations.ndim == 1 else chosen

    return policy


# ======================================================================================
# The comparator set
# ======================================================================================
def build_baselines(
    config: BaselinesConfig | None = None,
    *,
    statistics: dict[str, Any] | None = None,
    horizon: int | None = None,
) -> dict[str, Policy]:
    """The four rules as five concrete portfolios, in the order thesis 4.2.3 presents them.

    Args:
        config: Comparator configuration; read from ``config/baselines.yml`` when omitted.
        statistics: The ``statistics`` block of ``returns_matrix.json``; read from disk when omitted.
        horizon: Overrides the horizon the glide path descends over.

    Returns:
        Mapping from name to policy.
    """
    config = config or load_baselines()
    policies: dict[str, Policy] = {
        "constant_mix": constant_mix_policy(config),
        "glide_path": glide_path_policy(config, horizon),
    }
    for gamma in config.merton_gammas:
        policies[f"merton_gamma_{gamma:g}"] = merton_policy(config, gamma, statistics)
    policies["equal_weight"] = equal_weight_policy(config)
    return policies


def describe_baselines(
    config: BaselinesConfig | None = None,
    *,
    statistics: dict[str, Any] | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    """The parameters every comparator was built from, for the side-car of a results table.

    What the reader of subsection 4.4 needs in order to check that no comparator was tuned: the
    split actually used, the endpoints of the schedule, and both the unclipped and the constrained
    Merton fractions.

    Args:
        config: Comparator configuration; read from ``config/baselines.yml`` when omitted.
        statistics: The ``statistics`` block of ``returns_matrix.json``; read from disk when omitted.
        horizon: ``T`` in months; ``config.horizon_months`` when omitted.

    Returns:
        JSON-serialisable mapping with the columns, horizon, fixed-income split, fixed weights,
        60/40 weights, glide-path endpoints, Merton fractions per ``gamma`` and the equal weight.
    """
    config = config or load_baselines()
    horizon = int(horizon if horizon is not None else config.horizon_months)
    merton = {
        f"gamma_{gamma:g}": merton_fraction(config, gamma, statistics) for gamma in config.merton_gammas
    }
    return {
        "columns": list(config.mapping.columns),
        "horizon_months": horizon,
        "fixed_income_split": dict(config.mapping.fixed_income_shares),
        "fixed_weights": dict(config.mapping.fixed_weights),
        "constant_mix_weights": constant_mix_weights(config).tolist(),
        "glide_path": {
            "equity_weight_start": config.glide_equity_start,
            "equity_weight_end": config.glide_equity_end,
            "shape": config.glide_shape,
            "weights_at_start": config.mapping.allocate(glide_path_equity_weight(config, 0, horizon)).tolist(),
            "weights_at_end": config.mapping.allocate(
                glide_path_equity_weight(config, horizon, horizon)
            ).tolist(),
        },
        "merton": merton,
        "equal_weight": config.equal_weight,
    }
