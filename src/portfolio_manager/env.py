"""Simulation environment: the Markov decision process of thesis subsection 4.3.4.

Contract
--------
Two objects, and the first is the important one.

``step_wealth(wealth, weights, arrival_weights, returns_next, contribution, cost_rate)``
    A pure function returning ``(new_wealth, turnover, new_arrival_weights)``. It is the single
    implementation of Equations ``eq:transicion`` and ``eq:deriva-pesos`` in the whole project:
    ``env.py`` calls it once per step and ``dp_solver.py`` calls it on the entire grid (D74). It
    broadcasts over leading axes, so the solver can evaluate every wealth node against every
    candidate portfolio and every quadrature node in one call. There is no second transition
    anywhere, and the validation against Browne therefore debugs one piece of arithmetic.

``GoalPortfolioEnv(gymnasium.Env)``
    The Gymnasium 1.3.0 wrapper around it. Built either from a scenario tensor, consumed
    sequentially so that episode ``i`` uses row ``i``, which is the evaluation mode that makes the
    comparison across policies paired; or from a generator specification and a seed, producing a
    new scenario at every ``reset``, which is the training mode.

The observation is Equation ``eq:estado``, ``(W_t/G, t/T, C_t/G)``, in ``float32`` because that is
what the network consumes; wealth is carried in ``float64``. The reward is Equation
``eq:recompensa``: zero at every step but the last.

Decisions involved
------------------
D55 (the transition with its exact definitions of ``tau_t`` and ``w~_t``), D56 (the dynamic
programming table is computed at zero cost, so the cost rate is a construction parameter), D57
(``terminated=True`` on the final step, never ``truncated``), D74 (the reward is already in goal
units, so there is no scale parameter; one transition function shared with the solver).

Command line
------------
    uv run python -m portfolio_manager.env --checks
"""

# Imports
# =====================================================================
import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import yaml
from gymnasium import spaces

from portfolio_manager.environment_fingerprint import collect_fingerprint
from portfolio_manager.generator import (
    CHECKS_DIRECTORY,
    GbmParams,
    assert_columns,
    config_path,
    generate,
    generate_from_rng,
    git_commit,
    load_gbm_params,
    load_matrix,
    load_matrix_metadata,
    load_test_set,
)

# Constants
# =====================================================================
OBJECTIVE_LPM2 = "lpm2"
OBJECTIVE_INDICATOR = "indicator"
OBJECTIVES = (OBJECTIVE_LPM2, OBJECTIVE_INDICATOR)

BASIS_POINT = 1e-4

CHECK_SCENARIOS = 2000
"""Scenarios used by check 3 of thesis 4.3.4, the only one that needs a generator."""


# ======================================================================================
# Problem configuration
# ======================================================================================
@dataclass(frozen=True)
class ProblemConfig:
    """The base case: what is being solved, as opposed to how.

    Attributes:
        goal_usd: ``G``, the dated capital goal.
        horizon_months: ``T``, a configuration parameter and not a state variable.
        monthly_contribution_usd: ``c``, paid at the close of each month.
        initial_wealth_usd: ``W_0``.
        cost_bp: ``lambda`` in basis points per unit of turnover.
        cost_sweep_bp: The robustness sweep over the cost.
        objective: ``lpm2`` or ``indicator``.
    """

    goal_usd: float
    horizon_months: int
    monthly_contribution_usd: float
    initial_wealth_usd: float
    cost_bp: float
    cost_sweep_bp: tuple[float, ...]
    objective: str

    @property
    def cost_rate(self) -> float:
        """``lambda`` as a rate per unit of turnover.

        Returns:
            ``cost_bp`` of ``config/problem.yml`` converted from basis points to a fraction
            (10 bp -> 0.001).
        """
        return self.cost_bp * BASIS_POINT

    @property
    def total_contributions_usd(self) -> float:
        """What the investor puts in over the whole horizon, ignoring any return.

        Returns:
            ``c * T`` in USD.
        """
        return self.monthly_contribution_usd * self.horizon_months


def load_problem(path: Path | None = None) -> ProblemConfig:
    """Read ``config/problem.yml``.

    Args:
        path: Path to the file; resolved from the configuration directory when omitted.

    Returns:
        The base case.
    """
    path = path or config_path("problem")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the base case has no defaults in code")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    objective = str(raw["objective"])
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; this module implements {OBJECTIVES}")
    return ProblemConfig(
        goal_usd=float(raw["goal_usd"]),
        horizon_months=int(raw["horizon_months"]),
        monthly_contribution_usd=float(raw["monthly_contribution_usd"]),
        initial_wealth_usd=float(raw["initial_wealth_usd"]),
        cost_bp=float(raw["cost_bp"]),
        cost_sweep_bp=tuple(float(value) for value in raw["cost_sweep_bp"]),
        objective=objective,
    )


# ======================================================================================
# The transition, shared with the solver
# ======================================================================================
def project_to_simplex(action: np.ndarray) -> np.ndarray:
    """Map an action onto the simplex of Equation ``eq:simplex``.

    The rule of thesis 4.3.4: negative components are set to zero and the rest are divided by their
    sum; if the sum is zero the equal split is used. It is not a softmax, which would never return
    an exact zero and would therefore make a corner portfolio unreachable.

    The Dirichlet policy of the agent already produces points of the simplex, so for it this is the
    identity. The projection exists to leave the environment defined for every action of the
    declared space, including the ones a random or untrained policy emits.

    Args:
        action: ``(..., N)`` array.

    Returns:
        ``(..., N)`` array of non-negative rows summing to one.
    """
    clipped = np.clip(np.asarray(action, dtype=np.float64), 0.0, None)
    total = clipped.sum(axis=-1, keepdims=True)
    equal = np.full_like(clipped, 1.0 / clipped.shape[-1])
    return np.where(total > 0.0, clipped / np.where(total > 0.0, total, 1.0), equal)


def drift_weights(weights: np.ndarray, returns_next: np.ndarray) -> np.ndarray:
    """Weights the portfolio arrives with after one month, Equation ``eq:deriva-pesos``.

    The asset that rises most gains weight without anyone buying it. Returning the portfolio to its
    target is what costs ``lambda * tau * W``.

    Args:
        weights: ``(..., N)`` weights held during the month.
        returns_next: ``(..., N)`` simple returns of that month.

    Returns:
        ``(..., N)`` arrival weights, summing to one by construction.
    """
    grown = weights * (1.0 + returns_next)
    portfolio = grown.sum(axis=-1, keepdims=True)
    # A portfolio worth exactly zero has no weights. It cannot arise from the generator, whose
    # simple returns are strictly above -1, but the solver may be handed an arbitrary override.
    safe = np.where(portfolio != 0.0, portfolio, 1.0)
    return np.where(portfolio != 0.0, grown / safe, weights)


def step_wealth(
    wealth: np.ndarray | float,
    weights: np.ndarray,
    arrival_weights: np.ndarray,
    returns_next: np.ndarray,
    contribution: float,
    cost_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One month of Equation ``eq:transicion``, as a pure function.

    Read left to right in the order things happen. At the start of the month the portfolio is
    rebalanced from the weights it arrived with to the weights chosen, which costs
    ``lambda * tau * W``; what remains is invested and earns the portfolio return of the month; at
    the close the contribution arrives, at the arrival weights and free of cost, because every
    strategy must invest it alike.

    Turnover sums purchases and sales with no factor of one half, so a complete switch from one
    asset to another gives ``tau = 2`` and costs twice ``lambda``.

    Every argument broadcasts over leading axes, which is what lets ``dp_solver.py`` evaluate the
    whole wealth grid against every candidate portfolio in one call rather than reimplement this.

    Args:
        wealth: ``(...)`` wealth at the start of the month.
        weights: ``(..., N)`` weights chosen for the month.
        arrival_weights: ``(..., N)`` weights the portfolio arrived with. At ``t = 0`` these equal
            ``weights``, which is the convention ``w~_0 = w_0`` that makes the opening purchase free.
        returns_next: ``(..., N)`` simple returns of the month.
        contribution: Amount added at the close of the month.
        cost_rate: ``lambda``, per unit of turnover.

    Returns:
        Tuple ``(new_wealth, turnover, new_arrival_weights)``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    arrival_weights = np.asarray(arrival_weights, dtype=np.float64)
    returns_next = np.asarray(returns_next, dtype=np.float64)

    turnover = np.abs(weights - arrival_weights).sum(axis=-1)
    portfolio_return = (weights * returns_next).sum(axis=-1)
    new_wealth = np.asarray(wealth, dtype=np.float64) * (1.0 - cost_rate * turnover) * (
        1.0 + portfolio_return
    ) + contribution
    return new_wealth, turnover, drift_weights(weights, returns_next)


# ======================================================================================
# The environment
# ======================================================================================
@dataclass(frozen=True)
class GeneratorSpec:
    """Which generator a training environment draws its episodes from.

    Attributes:
        mode: ``"A"`` or ``"B"``.
        block_length: Mean block length ``L``; required by mode B and rejected by mode A.
        params: Mode-A parameters; read from configuration when omitted.
    """

    mode: str
    block_length: float | None = None
    params: GbmParams | None = None

    @classmethod
    def from_test_set(cls, configuration: str, test_set: dict[str, Any] | None = None) -> "GeneratorSpec":
        """Build the specification named by a configuration of the frozen recipe.

        The training environment takes the mode and the block length of the configuration it will
        later be evaluated on, and never its seed: those live in disjoint bands precisely so that a
        training stream cannot reach the evaluation scenarios (D23).

        Args:
            configuration: Key under ``configurations``, for example ``A_gbm`` or ``B_L12``.
            test_set: Parsed recipe; read from disk when omitted.

        Returns:
            The specification.
        """
        test_set = test_set if test_set is not None else load_test_set()
        spec = test_set["configurations"][configuration]
        block_length = spec.get("mean_block_length_months")
        return cls(mode=str(spec["mode"]), block_length=None if block_length is None else float(block_length))


class GoalPortfolioEnv(gymnasium.Env):
    """The goal-based allocation problem as a Gymnasium environment.

    The episode runs ``T`` months. At each one the agent chooses a portfolio on the simplex, the
    month's returns arrive, the contribution is added, and nothing is rewarded until the end, where
    the terminal reward of Equation ``eq:recompensa`` is paid. That single signal after a hundred
    and twenty decisions is the central difficulty of the problem and the reason the generator must
    be cheap.

    Attributes:
        problem: The base case being solved.
        objective: ``lpm2`` or ``indicator``.
        goal: ``G``.
        horizon: ``T``.
        contribution: ``c``.
        cost_rate: ``lambda`` per unit of turnover.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        *,
        scenarios: np.ndarray | None = None,
        generator_spec: GeneratorSpec | None = None,
        seed: int | None = None,
        problem: ProblemConfig | None = None,
        objective: str | None = None,
        goal: float | None = None,
        horizon: int | None = None,
        contribution: float | None = None,
        initial_wealth: float | None = None,
        cost_rate: float | None = None,
        shaping: bool = False,
        shaping_gamma: float | None = None,
    ) -> None:
        """Build an environment in one of the two construction modes of thesis 4.3.4.

        Args:
            scenarios: ``(n, T, N)`` tensor, consumed sequentially. Evaluation mode.
            generator_spec: Generator to draw a fresh scenario from at every reset. Training mode.
            seed: Seed of the training stream. Required with ``generator_spec``.
            problem: Base case; read from ``config/problem.yml`` when omitted.
            objective: Overrides the objective of the base case.
            goal: Overrides ``G``.
            horizon: Overrides ``T``.
            contribution: Overrides ``c``. The Browne validation configuration sets it to zero.
            initial_wealth: Overrides ``W_0``.
            cost_rate: Overrides ``lambda``. The dynamic programming table is built at zero (D56).
            shaping: Whether to add the potential-based term. Off by default; only the pre-declared
                rule of the unattended batch may turn it on.
            shaping_gamma: Discount of the shaping term, which must equal the one the learning
                algorithm uses. Read from ``config/ppo.yml`` when omitted.
        """
        if (scenarios is None) == (generator_spec is None):
            raise ValueError("supply exactly one of 'scenarios' (evaluation) or 'generator_spec' (training)")

        self.problem = problem or load_problem()
        self.objective = str(objective or self.problem.objective)
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}; this module implements {OBJECTIVES}")

        self.goal = float(goal if goal is not None else self.problem.goal_usd)
        self.horizon = int(horizon if horizon is not None else self.problem.horizon_months)
        self.contribution = float(
            contribution if contribution is not None else self.problem.monthly_contribution_usd
        )
        self.initial_wealth = float(
            initial_wealth if initial_wealth is not None else self.problem.initial_wealth_usd
        )
        self.cost_rate = float(cost_rate if cost_rate is not None else self.problem.cost_rate)
        if self.goal <= 0.0:
            raise ValueError(f"the goal must be strictly positive, got {self.goal}")

        self._scenarios = None if scenarios is None else np.asarray(scenarios, dtype=np.float64)
        self._spec = generator_spec
        self._cursor = 0
        self._matrix: np.ndarray | None = None
        self._gbm: GbmParams | None = None
        self._rng: np.random.Generator | None = None

        if self._scenarios is not None:
            if self._scenarios.ndim != 3 or self._scenarios.shape[1] != self.horizon:
                raise ValueError(
                    f"the tensor has shape {self._scenarios.shape}, expected (n, {self.horizon}, n_assets)"
                )
            self.n_assets = int(self._scenarios.shape[2])
        else:
            if seed is None:
                raise ValueError("training mode needs a seed: the episodes of a run must be reproducible")
            self._rng = np.random.default_rng(seed)
            # Validated once here rather than on every reset: a training run calls the generator
            # hundreds of thousands of times and must not re-read the recipe to do it.
            if self._spec.mode == "B":
                columns, self._matrix = load_matrix()
                assert_columns(columns)
                self.n_assets = int(self._matrix.shape[1])
            else:
                self._gbm = self._spec.params or load_gbm_params()
                if self._spec.params is None:
                    columns, _ = load_matrix()
                    assert_columns(columns)
                    if tuple(self._gbm.assets) != columns:
                        raise ValueError(
                            f"config/gbm_params.yml declares {list(self._gbm.assets)} and the matrix header "
                            f"is {list(columns)}; re-run --fit-gbm"
                        )
                self.n_assets = int(self._gbm.n_assets)

        self.shaping = bool(shaping)
        self._shaping_gamma = float(shaping_gamma) if shaping_gamma is not None else None
        if self.shaping and self._shaping_gamma is None:
            self._shaping_gamma = _ppo_gamma()

        total_contributions = self.contribution * self.horizon
        self.observation_space = spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.array([np.inf, 1.0, max(total_contributions / self.goal, np.finfo(np.float32).tiny)], np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        self._wealth = self.initial_wealth
        self._t = 0
        self._scenario: np.ndarray | None = None
        self._arrival_weights: np.ndarray | None = None

    # --- Gymnasium API ----------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        In evaluation mode the scenario is the next row of the tensor, so that episode ``i`` uses
        row ``i`` and every policy meets the same hundred and twenty months in the same order;
        ``options={"scenario_index": i}`` selects one explicitly. The ``seed`` argument follows the
        Gymnasium contract and reseeds the action space, and in training mode the episode stream as
        well; it does not move the cursor of a tensor, because which scenario comes next is a
        property of the evaluation protocol and not of the random state.

        Args:
            seed: Seed for the Gymnasium random state, and for the training stream when present.
            options: Optionally ``{"scenario_index": int}`` in evaluation mode.

        Returns:
            Tuple ``(observation, info)``.
        """
        super().reset(seed=seed)
        if seed is not None and self._rng is not None:
            self._rng = np.random.default_rng(seed)

        if self._scenarios is not None:
            index = self._cursor if options is None else int(options.get("scenario_index", self._cursor))
            if not 0 <= index < len(self._scenarios):
                raise IndexError(
                    f"scenario {index} requested from a tensor of {len(self._scenarios)}: the evaluation "
                    "loop must run exactly one episode per scenario"
                )
            self._scenario = self._scenarios[index]
            self._cursor = index + 1
        else:
            self._scenario = generate_from_rng(
                self._rng,
                self._spec.mode,
                1,
                self.horizon,
                block_length=self._spec.block_length,
                params=None if self._spec.mode == "B" else self._gbm,
                matrix=self._matrix,
            )[0]

        self._wealth = self.initial_wealth
        self._t = 0
        self._arrival_weights = None
        return self._observation(), {"scenario_index": self._cursor - 1 if self._scenarios is not None else None}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one month.

        Args:
            action: ``(N,)`` action, projected onto the simplex before anything else happens.

        Returns:
            Tuple ``(observation, reward, terminated, truncated, info)``.
        """
        if self._scenario is None:
            raise RuntimeError("step() before reset()")
        if self._t >= self.horizon:
            raise RuntimeError("step() after the episode terminated")

        weights = project_to_simplex(action)
        # w~_0 = w_0: the opening purchase is free, which is the convention of eq:deriva-pesos and
        # the reason the first decision cannot be penalised for existing.
        arrival = weights if self._arrival_weights is None else self._arrival_weights

        potential_before = self._potential(self._wealth, self._t) if self.shaping else 0.0
        wealth, turnover, self._arrival_weights = step_wealth(
            self._wealth, weights, arrival, self._scenario[self._t], self.contribution, self.cost_rate
        )
        self._wealth = float(wealth)
        self._t += 1

        terminated = self._t >= self.horizon
        reward = self._terminal_reward() if terminated else 0.0
        if self.shaping:
            reward += self._shaping_gamma * self._potential(self._wealth, self._t) - potential_before

        return (
            self._observation(),
            float(reward),
            bool(terminated),
            False,  # never truncated: the horizon is the problem, not a time limit (D57)
            {"wealth": self._wealth, "turnover": float(turnover), "weights": weights},
        )

    # --- Internals --------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        """Equation ``eq:estado``, in the precision the network consumes."""
        remaining = self.contribution * (self.horizon - self._t)
        return np.array(
            [self._wealth / self.goal, self._t / self.horizon, remaining / self.goal], dtype=np.float32
        )

    def _terminal_reward(self) -> float:
        """Equation ``eq:recompensa``, already in goal units and therefore bounded (D74)."""
        if self.objective == OBJECTIVE_LPM2:
            return -(max(0.0, 1.0 - self._wealth / self.goal) ** 2)
        return float(self._wealth >= self.goal)

    def _potential(self, wealth: float, t: int) -> float:
        """Potential of the optional shaping term.

        Zero at the terminal state, which is not a detail. Ng, Harada and Russell (1999) leave the
        set of optimal policies unchanged only if the potential vanishes there; the shortfall
        written at ``t = T`` equals the terminal reward itself, so without this the shaping would
        add a second, policy-dependent copy of the reward and would change the optimum it exists to
        preserve. With it the discounted sum telescopes to ``-Phi(s_0)``, the same constant for
        every policy starting from the same state.
        """
        if t >= self.horizon:
            return 0.0
        reachable = wealth + self.contribution * (self.horizon - t)
        return -(max(0.0, self.goal - reachable) ** 2) / (self.goal**2)


def _ppo_gamma() -> float:
    """Read the discount the agent uses, which the shaping term must match.

    One value, read from the agent's own configuration rather than declared a second time here. A
    potential discounted at one rate while the algorithm optimises at another is not
    policy-invariant, and a second declaration is how the two would drift apart.
    """
    path = config_path("ppo")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: shaping must use the same discount as the learning algorithm, so the "
            "value cannot be invented here. Pass shaping_gamma explicitly, or build config/ppo.yml."
        )
    return float((yaml.safe_load(path.read_text(encoding="utf-8")) or {})["ppo"]["gamma"])


# ======================================================================================
# Vectorised rollout
# ======================================================================================
def rollout(
    scenarios: np.ndarray,
    policy: Any,
    *,
    problem: ProblemConfig | None = None,
    objective: str | None = None,
    goal: float | None = None,
    contribution: float | None = None,
    initial_wealth: float | None = None,
    cost_rate: float | None = None,
) -> dict[str, np.ndarray]:
    """Run one policy over every scenario of a tensor at once.

    The same arithmetic the environment performs, applied along a batch axis: twenty thousand
    scenarios stepped a month at a time rather than one scenario stepped to the end and then the
    next. Both paths call :func:`step_wealth`, and a test asserts that they agree exactly, so this
    is a speed-up and not a second model. ``evaluate.py`` consumes it, which is why it takes an
    arbitrary callable and computes nothing specific to any strategy.

    Args:
        scenarios: ``(n, T, N)`` tensor of simple monthly returns.
        policy: Callable ``policy(observations, t) -> weights`` with observations ``(n, 3)`` in
            ``float32`` and weights ``(n, N)``. Weights are projected onto the simplex, so a policy
            that already returns simplex points is unaffected.
        problem: Base case; read from configuration when omitted.
        objective: Overrides the objective of the base case.
        goal: Overrides ``G``.
        contribution: Overrides ``c``.
        initial_wealth: Overrides ``W_0``.
        cost_rate: Overrides ``lambda``.

    Returns:
        Mapping with ``terminal_wealth`` ``(n,)``, ``reward`` ``(n,)``, ``total_turnover`` ``(n,)``
        and ``wealth_path`` ``(n, T + 1)``.
    """
    problem = problem or load_problem()
    objective = str(objective or problem.objective)
    goal = float(goal if goal is not None else problem.goal_usd)
    contribution = float(contribution if contribution is not None else problem.monthly_contribution_usd)
    initial_wealth = float(initial_wealth if initial_wealth is not None else problem.initial_wealth_usd)
    cost_rate = float(cost_rate if cost_rate is not None else problem.cost_rate)

    scenarios = np.asarray(scenarios, dtype=np.float64)
    n_scenarios, horizon, _ = scenarios.shape

    wealth = np.full(n_scenarios, initial_wealth, dtype=np.float64)
    path = np.empty((n_scenarios, horizon + 1), dtype=np.float64)
    path[:, 0] = wealth
    total_turnover = np.zeros(n_scenarios, dtype=np.float64)
    arrival: np.ndarray | None = None

    for t in range(horizon):
        remaining = contribution * (horizon - t)
        observations = np.stack(
            [
                wealth / goal,
                np.full(n_scenarios, t / horizon),
                np.full(n_scenarios, remaining / goal),
            ],
            axis=-1,
        ).astype(np.float32)

        weights = project_to_simplex(policy(observations, t))
        if arrival is None:
            arrival = weights
        wealth, turnover, arrival = step_wealth(
            wealth, weights, arrival, scenarios[:, t], contribution, cost_rate
        )
        total_turnover += turnover
        path[:, t + 1] = wealth

    return {
        "terminal_wealth": wealth,
        "reward": terminal_reward(wealth, goal, objective),
        "total_turnover": total_turnover,
        "wealth_path": path,
    }


def terminal_reward(terminal_wealth: np.ndarray, goal: float, objective: str) -> np.ndarray:
    """Equation ``eq:recompensa``, vectorised over scenarios.

    Args:
        terminal_wealth: ``W_T`` per scenario, in USD.
        goal: ``G``, in USD.
        objective: ``lpm2`` for ``-(max(0, 1 - W_T / G))^2`` or ``indicator`` for ``1{W_T >= G}``.

    Returns:
        Terminal reward per scenario, ``float64``.

    Raises:
        ValueError: If the objective is not one this module implements.
    """
    terminal_wealth = np.asarray(terminal_wealth, dtype=np.float64)
    if objective == OBJECTIVE_LPM2:
        return -(np.clip(1.0 - terminal_wealth / goal, 0.0, None) ** 2)
    if objective == OBJECTIVE_INDICATOR:
        return (terminal_wealth >= goal).astype(np.float64)
    raise ValueError(f"unknown objective {objective!r}; this module implements {OBJECTIVES}")


# ======================================================================================
# The four checks of thesis 4.3.4
# ======================================================================================
def _constant_policy(weights: np.ndarray) -> Any:
    """A policy holding the same target portfolio every month."""

    def policy(observations: np.ndarray, t: int) -> np.ndarray:  # noqa: ARG001 - the contract fixes the signature
        return np.broadcast_to(weights, (observations.shape[0], weights.size))

    return policy


def _alternating_vertex_policy(n_assets: int) -> Any:
    """A policy switching between two vertices of the simplex every month.

    From ``t = 1`` onwards it is the most expensive admissible behaviour: the whole portfolio moves
    from one asset to another, which is a turnover of two.
    """

    def policy(observations: np.ndarray, t: int) -> np.ndarray:
        weights = np.zeros(n_assets)
        weights[t % 2] = 1.0
        return np.broadcast_to(weights, (observations.shape[0], n_assets))

    return policy


def _uniform_simplex_policy(rng: np.random.Generator, n_assets: int) -> Any:
    """A policy drawing a portfolio uniformly from the simplex each month, that is Dirichlet(1)."""

    def policy(observations: np.ndarray, t: int) -> np.ndarray:  # noqa: ARG001
        return rng.dirichlet(np.ones(n_assets), size=observations.shape[0])

    return policy


def _constant_mix_weights(columns: tuple[str, ...]) -> np.ndarray:
    """The 60/40 portfolio projected onto the universe, from ``config/baselines.yml`` (D68)."""
    baselines = yaml.safe_load(config_path("baselines").read_text(encoding="utf-8")) or {}
    equity = float(baselines["constant_mix"]["equity_weight"])
    fixed_income = float(baselines["constant_mix"]["fixed_income_weight"])
    split = baselines["asset_mapping"]["fixed_income_split"]
    weights = dict.fromkeys(columns, 0.0)
    weights["us_equity"] = equity
    weights["ust10y"] = fixed_income * float(split["ust10y_share"])
    weights["ust2y"] = fixed_income * float(split["ust2y_share"])
    return np.array([weights[name] for name in columns], dtype=np.float64)


def environment_checks(problem: ProblemConfig | None = None) -> dict[str, Any]:
    """Run the four checks of thesis 4.3.4 plus the hand-computed path, and return their numbers.

    The same function backs the command line and the tests, so the figures reported in subsection
    4.4 are the figures the suite asserts on.

    Args:
        problem: Base case; read from configuration when omitted.

    Returns:
        A JSON-serialisable document with one entry per check and an overall verdict.
    """
    problem = problem or load_problem()
    columns, _ = load_matrix()
    n_assets = len(columns)
    document: dict[str, Any] = {"problem": _problem_summary(problem), "checks": {}}
    checks = document["checks"]

    zero_returns = np.zeros((4, problem.horizon_months, n_assets), dtype=np.float64)

    # --- 1. Contributions alone, with no return and no cost -------------------------
    # Any policy whatsoever must end at the sum of the contributions, because nothing else in the
    # transition can move the wealth. It is the check that catches an off-by-one in the horizon.
    policies = {
        "equal_weight": _constant_policy(np.full(n_assets, 1.0 / n_assets)),
        "all_equity": _constant_policy(np.eye(n_assets)[0]),
        "alternating_vertices": _alternating_vertex_policy(n_assets),
        "uniform_simplex": _uniform_simplex_policy(np.random.default_rng(1000), n_assets),
    }
    terminal = {
        name: float(rollout(zero_returns, policy, problem=problem, cost_rate=0.0)["terminal_wealth"][0])
        for name, policy in policies.items()
    }
    expected_total = problem.total_contributions_usd
    checks["contributions_only"] = {
        "expected_terminal_wealth": expected_total,
        "terminal_wealth": terminal,
        "max_absolute_error": max(abs(value - expected_total) for value in terminal.values()),
        "passed": all(value == expected_total for value in terminal.values()),
    }

    # --- 2. Cost of the most expensive admissible behaviour -------------------------
    # With tau = 2 every month from t = 1, the recursion W <- W (1 - 2 lambda) + c has a closed
    # form. Matching it to the last cent is what fixes the turnover convention: with a factor of
    # one half in tau the figure would be different and the check would fail.
    cost_rate = problem.cost_rate
    alternating = float(
        rollout(zero_returns, policies["alternating_vertices"], problem=problem)["terminal_wealth"][0]
    )
    closed_form = (
        problem.monthly_contribution_usd
        * (1.0 - (1.0 - 2.0 * cost_rate) ** problem.horizon_months)
        / (2.0 * cost_rate)
    )
    checks["turnover_cost"] = {
        "cost_bp": problem.cost_bp,
        "closed_form": closed_form,
        "simulated": alternating,
        "absolute_error": abs(alternating - closed_form),
        "formula": "c (1 - (1 - 2 lambda)^T) / (2 lambda)",
        "passed": abs(alternating - closed_form) < 1e-6,
    }

    # --- 3. Terminal wealth on resampled history ------------------------------------
    # The median must sit above the contributions, since the universe earns something, and below
    # what those contributions would reach compounding at the best asset of the universe, since no
    # long-only portfolio can beat the best asset in every month.
    test_set = load_test_set()
    seed = int(test_set["seeds"]["validation"]["min"])
    spec = GeneratorSpec.from_test_set("B_L12", test_set)
    scenarios = generate(
        spec.mode, CHECK_SCENARIOS, problem.horizon_months, seed, block_length=spec.block_length
    )
    statistics = load_matrix_metadata()["statistics"]
    best = max(statistics, key=lambda name: statistics[name]["geometric_mean_annual"])
    monthly = (1.0 + statistics[best]["geometric_mean_annual"]) ** (1.0 / 12.0) - 1.0
    upper = problem.monthly_contribution_usd * (((1.0 + monthly) ** problem.horizon_months - 1.0) / monthly)

    bounded: dict[str, Any] = {}
    bootstrap_policies = {
        "uniform_simplex": _uniform_simplex_policy(np.random.default_rng(seed), n_assets),
        "constant_mix_60_40": _constant_policy(_constant_mix_weights(columns)),
    }
    for name, policy in bootstrap_policies.items():
        result = rollout(scenarios, policy, problem=problem)
        wealth = result["terminal_wealth"]
        bounded[name] = {
            "median_terminal_wealth": float(np.median(wealth)),
            "quantiles": {
                str(q): float(np.quantile(wealth, q / 100.0)) for q in (5, 25, 50, 75, 95)
            },
            "probability_of_reaching_goal": float((wealth >= problem.goal_usd).mean()),
            "mean_terminal_reward": float(result["reward"].mean()),
            "annualised_turnover": float(result["total_turnover"].mean() * 12.0 / problem.horizon_months),
            "passed": bool(expected_total < np.median(wealth) < upper),
        }
    checks["bootstrap_bounds"] = {
        "n_scenarios": CHECK_SCENARIOS,
        "seed": seed,
        "block_length": spec.block_length,
        "lower_bound": expected_total,
        "upper_bound": upper,
        "upper_bound_asset": best,
        "policies": bounded,
        "passed": all(entry["passed"] for entry in bounded.values()),
    }

    # --- 4. Interface compliance and the bound on the reward ------------------------
    from gymnasium.utils.env_checker import check_env  # local: only this check needs it

    rewards: list[float] = []
    env = GoalPortfolioEnv(scenarios=scenarios[:64], problem=problem)
    check_env(env, skip_render_check=True)
    env = GoalPortfolioEnv(scenarios=scenarios[:64], problem=problem)
    # The random actions come from the action space, whose stream is separate from the scenario
    # one and is seeded here. Without this the bound reported below moves between runs, and a
    # figure that goes into subsection 4.4 has to reproduce.
    env.action_space.seed(seed)
    for _ in range(len(scenarios[:64])):
        env.reset()
        terminated = False
        while not terminated:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert not truncated
            rewards.append(reward)
    checks["interface"] = {
        "check_env": "passed",
        "episodes": 64,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "passed": bool(min(rewards) >= -1.0 and max(rewards) <= 0.0),
    }

    # --- 5. A path short enough to verify by hand -----------------------------------
    checks["toy_path"] = _toy_path_check()

    document["all_passed"] = all(bool(entry["passed"]) for entry in checks.values())
    return document


def _problem_summary(problem: ProblemConfig) -> dict[str, Any]:
    """The base case, as written into every result file."""
    return {
        "goal_usd": problem.goal_usd,
        "horizon_months": problem.horizon_months,
        "monthly_contribution_usd": problem.monthly_contribution_usd,
        "initial_wealth_usd": problem.initial_wealth_usd,
        "cost_bp": problem.cost_bp,
        "objective": problem.objective,
    }


def _toy_path_check() -> dict[str, Any]:
    """Three months, two assets, every figure computable with a pencil.

    The long checks bound aggregates; this one pins the arithmetic itself, including the one place
    where an implementation can plausibly go wrong without any aggregate noticing: the arrival
    weights, which drift with the returns and decide the turnover of the following month.
    """
    returns = np.array([[[0.10, 0.00], [-0.20, 0.00], [0.50, 0.00]]], dtype=np.float64)
    env = GoalPortfolioEnv(
        scenarios=returns,
        goal=1000.0,
        horizon=3,
        contribution=100.0,
        initial_wealth=1000.0,
        cost_rate=0.01,
        objective=OBJECTIVE_LPM2,
    )
    env.reset()
    observed: list[dict[str, float]] = []
    for weights in ([1.0, 0.0], [0.5, 0.5], [0.0, 1.0]):
        _, reward, terminated, _, info = env.step(np.array(weights))
        observed.append({"wealth": info["wealth"], "turnover": info["turnover"], "reward": reward})

    # Month 1. w = (1, 0) and w~_0 = w by convention, so tau = 0. Wealth 1000 * 1.10 = 1100, plus
    #          the contribution: 1200. The portfolio is all in asset 1, so it arrives at (1, 0).
    # Month 2. w = (0.5, 0.5) against arrival (1, 0): tau = |0.5-1| + |0.5-0| = 1, costing 1 %.
    #          1200 * 0.99 = 1188, which returns 0.5*(-0.20) + 0.5*0 = -0.10, giving 1069.2, plus
    #          the contribution: 1169.2. Arrival weights: asset 1 is 0.5*0.8 = 0.4 of the pre-trade
    #          portfolio against a total of 0.9, so (4/9, 5/9).
    # Month 3. w = (0, 1) against (4/9, 5/9): tau = 4/9 + 4/9 = 8/9, costing 8/900.
    #          1169.2 * (1 - 8/900) = 1158.8086..., return 0, plus the contribution: 1258.8086...
    expected = [
        {"wealth": 1200.0, "turnover": 0.0},
        {"wealth": 1169.2, "turnover": 1.0},
        {"wealth": 1169.2 * (1.0 - 8.0 / 900.0) + 100.0, "turnover": 8.0 / 9.0},
    ]
    errors = [
        max(abs(observed[i]["wealth"] - expected[i]["wealth"]), abs(observed[i]["turnover"] - expected[i]["turnover"]))
        for i in range(3)
    ]
    return {
        "expected": expected,
        "observed": observed,
        "max_absolute_error": max(errors),
        "terminal_reward": observed[-1]["reward"],
        "terminated_on_last_step": bool(terminated),
        "passed": bool(max(errors) < 1e-9 and terminated),
    }


def write_environment_checks(document: dict[str, Any], directory: Path = CHECKS_DIRECTORY) -> Path:
    """Write the numbers where subsection 4.4 reads them from.

    Args:
        document: Output of :func:`environment_checks`.
        directory: Destination folder; ``results/checks/`` by default.

    Returns:
        Path of the written ``env_checks.json``, which also carries the command line, UTC time,
        git commit and environment fingerprint.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "env_checks.json"
    payload = {
        **document,
        "command_line": sys.argv,
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment_fingerprint": collect_fingerprint(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    return path


# ======================================================================================
# Command line
# ======================================================================================
def main(argv: list[str] | None = None) -> int:
    """Command line: ``python -m portfolio_manager.env --checks``.

    Args:
        argv: Arguments to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        Exit status: 0 when every check passes (or with no flag, after printing the help), 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Simulation environment: the four checks of thesis 4.3.4.")
    parser.add_argument("--checks", action="store_true", help="run the four checks and write their numbers")
    parser.add_argument("--seed", type=int, default=None, help="unused by --checks, whose seeds are declared")
    args = parser.parse_args(argv)

    if not args.checks:
        parser.print_help()
        return 0

    try:
        document = environment_checks()
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    path = write_environment_checks(document)
    for name, check in document["checks"].items():
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {name}")
    print(f"\nwrote {path}")
    return 0 if document["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
