"""The agent: a Dirichlet policy trained with PPO.

Thesis subsection 4.3.8 is not yet written; this module implements what D57 and the build brief
specify, and is the document the subsection will be written from.

Contract
--------
``DirichletDistribution``
    A ``stable_baselines3.common.distributions.Distribution`` over the simplex. The network emits
    four reals, the concentration is ``alpha = softplus(logits) + alpha_min`` with ``alpha_min > 1``
    from configuration, and a sample is therefore a portfolio by construction rather than by
    projection. This is the algorithmic contribution of the work (D27): composing a Gaussian with a
    softmax deforms the density and miscalibrates the entropy, because the transformation is neither
    injective onto the simplex nor volume-preserving, whereas the Dirichlet is the natural
    distribution on that set.

``DirichletActorCriticPolicy``
    The wiring. ``ActorCriticPolicy`` dispatches on the concrete distribution type in two places and
    raises on anything it does not recognise, so both hooks are overridden. **The PPO algorithm
    itself is untouched** (D26): no hypothesis of this work depends on the optimiser, and
    reimplementing it would add days of debugging for nothing.

``train``
    One run: a vectorised environment in training mode, PPO with the library defaults, a held-out
    evaluation callback, and a directory holding the policy, the metrics and the fingerprint.

Seed discipline
---------------
Three disjoint bands (D23). Training streams are ``seed * 100 + env_index``, inside 0-999. The
held-out tensor uses ``1000 + seed``, inside the validation band 1000-1999. The frozen test set
uses ``20260809 + offset`` and is never touched here: no figure this module produces is measured on
it, and no decision is taken from one.

Decisions involved
------------------
D26 (SB3 is not reimplemented), D27 (the Dirichlet policy), D57 (library defaults written out,
``alpha_min > 1``), D74 and A40 (shaping is a diagnostic read only under a pre-declared rule).

Command line
------------
    uv run python -m portfolio_manager.train --config A_gbm --seed 0 --total-timesteps 50000
"""

# Imports
# =====================================================================
import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from torch import nn
from torch.distributions import Dirichlet

from portfolio_manager.baselines import Policy
from portfolio_manager.env import GeneratorSpec, GoalPortfolioEnv, ProblemConfig, load_problem, rollout
from portfolio_manager.environment_fingerprint import collect_fingerprint
from portfolio_manager.generator import config_path, generate, git_commit, load_test_set

# Constants
# =====================================================================
RUNS_DIRECTORY = Path("results/runs")


def run_directory_name(configuration: str, cost_bp: float) -> str:
    """Directory of one training configuration, naming the cost it was trained under.

    The transaction cost changes the optimal policy and not merely the score of a given one, so an
    agent trained at ten basis points is a different agent from one trained at zero. Two such runs
    under one name would be indistinguishable once the terminal that launched them is gone, and the
    sweep over the cost exists precisely to put them side by side.
    """
    return f"{configuration}_{cost_bp:g}bp"


TRAINING_BAND = (0, 999)
"""Seeds reserved for training streams, disjoint from validation and test (D23)."""

ENV_SEED_STRIDE = 100
"""Sub-environment ``i`` of a run seeded ``seed`` owns the stream ``seed * stride + i``. With at
most this many sub-environments the streams of two runs cannot collide."""

STEPS_PER_RATE_UNIT = 100_000
"""The throughput figure is reported per hundred thousand steps, which is what sizes the batch."""


# ======================================================================================
# Configuration
# ======================================================================================
@dataclass(frozen=True)
class AgentConfig:
    """Everything the agent needs that is not code.

    Attributes:
        ppo: The Stable-Baselines3 defaults, passed through untouched (D57).
        alpha_min: Lower bound of the Dirichlet concentration, strictly above one.
        log_prob_epsilon: Clamp applied to actions before evaluating their log-probability.
        device: Compute device; CPU for networks this size.
        n_envs: Sub-environments gathering transitions in parallel.
        total_timesteps: Default length of a run; the command line overrides it.
        eval_freq: Total environment steps between held-out evaluations.
        n_eval_scenarios: Scenarios in the held-out tensor.
    """

    ppo: dict[str, Any]
    alpha_min: float
    log_prob_epsilon: float
    device: str
    n_envs: int
    total_timesteps: int
    eval_freq: int
    n_eval_scenarios: int

    @property
    def gamma(self) -> float:
        """The discount the algorithm optimises with, which the shaping term must match."""
        return float(self.ppo["gamma"])


def load_agent_config(path: Path | None = None) -> AgentConfig:
    """Read ``config/ppo.yml``."""
    path = path or config_path("ppo")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: no hyperparameter has a default in code")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    policy = raw["policy"]
    alpha_min = float(policy["alpha_min"])
    if alpha_min <= 1.0:
        raise ValueError(
            f"alpha_min must be strictly above one (D57), got {alpha_min}. At or below one the "
            "Dirichlet density is unbounded at the faces of the simplex and the ratio PPO clips "
            "stops being meaningful."
        )
    return AgentConfig(
        ppo=dict(raw["ppo"]),
        alpha_min=alpha_min,
        log_prob_epsilon=float(policy["log_prob_epsilon"]),
        device=str(raw["device"]),
        n_envs=int(raw["n_envs"]),
        total_timesteps=int(raw["total_timesteps"]),
        eval_freq=int(raw["eval_freq"]),
        n_eval_scenarios=int(raw["n_eval_scenarios"]),
    )


# ======================================================================================
# The Dirichlet policy
# ======================================================================================
class DirichletDistribution(Distribution):
    """A Dirichlet distribution over the simplex, as a Stable-Baselines3 distribution.

    The natural distribution on the set the action must belong to. Every sample satisfies the two
    constraints of Equation ``eq:simplex`` exactly, so the projection the environment applies is
    the identity for this policy and no probability mass is spent on actions outside the feasible
    set.

    Attributes:
        action_dim: Number of assets.
        alpha_min: Lower bound of the concentration, strictly above one.
        epsilon: Clamp applied before evaluating a log-probability.
    """

    def __init__(self, action_dim: int, alpha_min: float, epsilon: float = 1e-6) -> None:
        super().__init__()
        if alpha_min <= 1.0:
            raise ValueError(f"alpha_min must be strictly above one (D57), got {alpha_min}")
        self.action_dim = int(action_dim)
        self.alpha_min = float(alpha_min)
        self.epsilon = float(epsilon)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """The layer producing the logits behind the concentration."""
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, action_logits: th.Tensor) -> "DirichletDistribution":
        """Set the concentration from the network output.

        ``alpha = softplus(logits) + alpha_min`` is positive by construction and bounded below by
        ``alpha_min``, so no state can drive the policy into the regime where the density diverges
        at the boundary. Validation is disabled on the underlying distribution because the actions
        handed back by the rollout buffer are stored in single precision and their components sum
        to one only to within that precision; :meth:`log_prob` renormalises them instead.
        """
        self.distribution = Dirichlet(nn.functional.softplus(action_logits) + self.alpha_min, validate_args=False)
        return self

    @property
    def concentration(self) -> th.Tensor:
        """The concentration vector currently parameterising the distribution."""
        return self.distribution.concentration

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """Log-density of an action, after making it a strictly interior simplex point.

        A Dirichlet assigns no density to a point with a zero component, and a sampled weight small
        enough to round to zero in single precision is common at the concentrations this policy
        reaches. Left alone it would return minus infinity, which propagates through the ratio into
        the gradient and ends the run. Clamping to ``[eps, 1 - eps]`` and renormalising moves the
        action by at most a few parts in a million and keeps the quantity finite.
        """
        clamped = th.clamp(actions, self.epsilon, 1.0 - self.epsilon)
        return self.distribution.log_prob(clamped / clamped.sum(dim=-1, keepdim=True))

    def entropy(self) -> th.Tensor:
        """Shannon entropy, available in closed form for the Dirichlet.

        Having it exactly is one of the reasons for this distribution: under a Gaussian composed
        with a softmax the entropy of the actions is not the entropy of the Gaussian, so the
        entropy bonus of PPO would regularise a quantity other than the one it appears to.
        """
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        """A portfolio drawn from the policy, on the simplex by construction."""
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        """The deterministic action: the Dirichlet **mean**, ``alpha / sum(alpha)``.

        The mean rather than the density mode. With ``alpha_min > 1`` the mode exists too, at
        ``(alpha - 1) / (sum(alpha) - N)``, but it is more extreme than the mean and moves
        discontinuously as a component approaches one, whereas the mean is the barycentre of what
        the policy would actually play. Declared in ``config/ppo.yml`` as ``deterministic_action``.
        """
        return self.distribution.mean

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Parameters to action, in one call."""
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Parameters to a sampled action and its log-probability, in one call."""
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)


class DirichletActorCriticPolicy(ActorCriticPolicy):
    """``ActorCriticPolicy`` with the Dirichlet in place of the diagonal Gaussian.

    ``ActorCriticPolicy`` dispatches on the concrete distribution type when it builds the heads and
    again when it turns a latent code into a distribution, and raises on anything it does not
    recognise. Both hooks are overridden here and nothing else is: the algorithm, the rollout
    buffer and the update are the library's (D26).
    """

    def __init__(self, *args: Any, alpha_min: float, log_prob_epsilon: float = 1e-6, **kwargs: Any) -> None:
        self._alpha_min = float(alpha_min)
        self._log_prob_epsilon = float(log_prob_epsilon)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        """Install the Dirichlet, then let the base class build the heads around it."""
        self.action_dist = DirichletDistribution(
            int(np.prod(self.action_space.shape)), self._alpha_min, self._log_prob_epsilon
        )
        self._build_mlp_extractor()
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            # The gains the library uses, including the small one on the action head that keeps the
            # initial policy close to uniform over the simplex rather than at a corner.
            gains = {self.mlp_extractor: np.sqrt(2), self.action_net: 0.01, self.value_net: 1.0}
            if self.share_features_extractor:
                gains[self.features_extractor] = np.sqrt(2)
            else:
                gains[self.pi_features_extractor] = np.sqrt(2)
                gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in gains.items():
                module.apply(lambda layer, gain=gain: self.init_weights(layer, gain))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """Latent code to distribution; the base class has no branch for this type."""
        return self.action_dist.proba_distribution(self.action_net(latent_pi))


# ======================================================================================
# Environments
# ======================================================================================
def make_training_env(
    spec: GeneratorSpec,
    seed: int,
    n_envs: int,
    *,
    problem: ProblemConfig | None = None,
    cost_rate: float | None = None,
    shaping: bool = False,
    shaping_gamma: float | None = None,
) -> VecEnv:
    """A vectorised environment in training mode, one generator stream per sub-environment.

    ``make_vec_env`` is deliberately not used: it has no hook for per-rank construction arguments,
    and the per-rank generator seed is the whole point. The sub-environments are built explicitly
    and wrapped exactly as that helper would wrap them.

    Args:
        spec: Mode and block length, taken from a named configuration of the frozen recipe.
        seed: Seed of the run, inside the training band.
        n_envs: Number of sub-environments.
        problem: Base case; read from configuration when omitted.
        cost_rate: Overrides ``lambda``. The agent must be trained under the cost it will face,
            because the cost changes the optimal policy and not merely the score.
        shaping: Whether to add the potential-based term.
        shaping_gamma: Discount of that term; must equal the one PPO uses.

    Returns:
        The vectorised environment.
    """
    if not TRAINING_BAND[0] <= seed <= TRAINING_BAND[1]:
        raise ValueError(
            f"the training seed must lie in the band {TRAINING_BAND} (D23), got {seed}. The validation "
            "and test bands are reserved so that a training stream cannot reach the evaluation scenarios."
        )
    if n_envs > ENV_SEED_STRIDE:
        raise ValueError(f"at most {ENV_SEED_STRIDE} sub-environments, or the streams of two runs collide")

    def make(index: int):
        def _init() -> Monitor:
            return Monitor(
                GoalPortfolioEnv(
                    generator_spec=spec,
                    seed=seed * ENV_SEED_STRIDE + index,
                    problem=problem,
                    cost_rate=cost_rate,
                    shaping=shaping,
                    shaping_gamma=shaping_gamma,
                )
            )

        return _init

    return DummyVecEnv([make(index) for index in range(n_envs)])


def policy_from_model(model: PPO) -> Policy:
    """Expose a trained model through the comparator interface.

    Section 2b: ``evaluate.py`` scores any ``policy(observations, t) -> weights``, so the agent has
    to look like one. The deterministic action is the Dirichlet mean.
    """

    def policy(observations: np.ndarray, t: int) -> np.ndarray:  # noqa: ARG001 - the clock is in the state
        actions, _ = model.predict(np.asarray(observations, dtype=np.float32), deterministic=True)
        return actions

    return policy


def uniform_simplex_policy(rng: np.random.Generator, n_assets: int) -> Policy:
    """The random policy the agent is compared against at the start of a run.

    Uniform on the simplex, which is Dirichlet with every concentration at one. This is the
    reference the pre-declared shaping rule of D74 and A40 is stated against, so it has to be a
    fixed, documented strategy rather than the untrained network: an untrained Dirichlet head emits
    near-zero logits, whose mean is the equal split, and comparing against that would be comparing
    against 1/N under another name.
    """

    def policy(observations: np.ndarray, t: int) -> np.ndarray:  # noqa: ARG001
        return rng.dirichlet(np.ones(n_assets), size=len(observations))

    return policy


# ======================================================================================
# Held-out evaluation
# ======================================================================================
def probability_of_reaching_goal(
    scenarios: np.ndarray, policy: Policy, problem: ProblemConfig, cost_rate: float | None = None
) -> float:
    """The primary metric of thesis 4.2.4, on a tensor held out from training."""
    wealth = rollout(scenarios, policy, problem=problem, cost_rate=cost_rate)["terminal_wealth"]
    return float((wealth >= problem.goal_usd).mean())


class HeldOutEvaluation(BaseCallback):
    """Score the deterministic policy on a fixed held-out tensor during training.

    Not a model-selection device. Nothing is restored from the best checkpoint and no
    hyperparameter is chosen from this curve; it exists so that the question "is the agent learning
    at all" has an answer before the frozen test set is touched, and so that the degradation
    criterion has something to read.

    Attributes:
        scenarios: The held-out tensor.
        problem: Base case.
        eval_freq: Total environment steps between evaluations.
        random_value: The random-policy value, computed once before training.
        history: One record per evaluation.
    """

    def __init__(
        self,
        scenarios: np.ndarray,
        problem: ProblemConfig,
        eval_freq: int,
        n_envs: int,
        random_value: float,
        verbose: int = 0,
        *,
        cost_rate: float | None = None,
    ) -> None:
        super().__init__(verbose)
        self.scenarios = scenarios
        self.problem = problem
        self.cost_rate = cost_rate
        # The callback fires once per vectorised step, which advances n_envs environments.
        self.every_n_calls = max(1, eval_freq // max(n_envs, 1))
        self.random_value = random_value
        self.history: list[dict[str, float]] = []

    def _evaluate(self) -> float:
        value = probability_of_reaching_goal(
            self.scenarios, policy_from_model(self.model), self.problem, self.cost_rate
        )
        self.history.append({"timesteps": int(self.num_timesteps), "probability_of_reaching_goal": value})
        self.logger.record("held_out/probability_of_reaching_goal", value)
        self.logger.record("held_out/random_policy", self.random_value)
        self.logger.record("held_out/improvement_over_random", value - self.random_value)
        return value

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_calls == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        self._evaluate()


# ======================================================================================
# One run
# ======================================================================================
def train(
    configuration: str,
    seed: int,
    total_timesteps: int | None = None,
    *,
    cost_bp: float | None = None,
    shaping: bool = False,
    agent: AgentConfig | None = None,
    problem: ProblemConfig | None = None,
    runs_directory: Path = RUNS_DIRECTORY,
    argv: list[str] | None = None,
    progress_bar: bool = False,
) -> dict[str, Any]:
    """Train one agent and write everything needed to read the result on Monday.

    Args:
        configuration: Key of the frozen recipe supplying the generator mode and block length.
        seed: Seed of the run, inside the training band.
        total_timesteps: Length of the run; the configured default when omitted.
        cost_bp: Transaction cost the agent is trained under; the base case when omitted.
        shaping: Whether to add the potential-based term. Diagnostic only (D74, A40).
        agent: Agent configuration; read from ``config/ppo.yml`` when omitted.
        problem: Base case; read from configuration when omitted.
        runs_directory: Root of the output tree.
        argv: Command line recorded in the metrics file.
        progress_bar: Whether to show a progress bar.

    Returns:
        The metrics document, which is also written to disk.
    """
    agent = agent or load_agent_config()
    problem = problem or load_problem()
    total_timesteps = int(total_timesteps if total_timesteps is not None else agent.total_timesteps)
    effective_bp = problem.cost_bp if cost_bp is None else float(cost_bp)
    cost_rate = effective_bp * 1e-4
    recipe = load_test_set()
    spec = GeneratorSpec.from_test_set(configuration, recipe)

    # The held-out tensor: validation band, disjoint from both the training streams and the test set.
    validation_band = recipe["seeds"]["validation"]
    validation_seed = int(validation_band["min"]) + seed
    if not int(validation_band["min"]) <= validation_seed <= int(validation_band["max"]):
        raise ValueError(f"the held-out seed {validation_seed} falls outside the validation band")
    held_out = generate(
        spec.mode,
        agent.n_eval_scenarios,
        problem.horizon_months,
        validation_seed,
        block_length=spec.block_length,
    )
    random_value = probability_of_reaching_goal(
        held_out,
        uniform_simplex_policy(np.random.default_rng(validation_seed), int(held_out.shape[2])),
        problem,
        cost_rate,
    )

    output = runs_directory / run_directory_name(configuration, effective_bp) / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)

    environment = make_training_env(
        spec,
        seed,
        agent.n_envs,
        problem=problem,
        cost_rate=cost_rate,
        shaping=shaping,
        shaping_gamma=agent.gamma if shaping else None,
    )
    model = PPO(
        DirichletActorCriticPolicy,
        environment,
        seed=seed,
        device=agent.device,
        tensorboard_log=str(output / "tensorboard"),
        policy_kwargs={"alpha_min": agent.alpha_min, "log_prob_epsilon": agent.log_prob_epsilon},
        verbose=0,
        **agent.ppo,
    )

    callback = HeldOutEvaluation(
        held_out, problem, agent.eval_freq, agent.n_envs, random_value, cost_rate=cost_rate
    )
    started = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=progress_bar)
    elapsed = time.perf_counter() - started
    environment.close()

    model.save(output / "policy.zip")
    final_value = callback.history[-1]["probability_of_reaching_goal"]
    metrics = {
        "configuration": configuration,
        "run_directory": run_directory_name(configuration, effective_bp),
        "seed": seed,
        "cost_bp": effective_bp,
        "shaping": shaping,
        "total_timesteps": int(model.num_timesteps),
        "seconds_elapsed": elapsed,
        "seconds_per_100k_steps": elapsed * STEPS_PER_RATE_UNIT / max(model.num_timesteps, 1),
        "final_held_out_probability": final_value,
        "random_policy_probability": random_value,
        "improvement_over_random": final_value - random_value,
        "held_out": {
            "seed": validation_seed,
            "n_scenarios": agent.n_eval_scenarios,
            "mode": spec.mode,
            "block_length": spec.block_length,
            "history": callback.history,
        },
        "training_env_seeds": [seed * ENV_SEED_STRIDE + index for index in range(agent.n_envs)],
        "agent": {
            "ppo": agent.ppo,
            "alpha_min": agent.alpha_min,
            "log_prob_epsilon": agent.log_prob_epsilon,
            "n_envs": agent.n_envs,
            "device": agent.device,
            "policy": "DirichletActorCriticPolicy",
        },
        "problem": {
            "goal_usd": problem.goal_usd,
            "horizon_months": problem.horizon_months,
            "monthly_contribution_usd": problem.monthly_contribution_usd,
            "cost_bp": effective_bp,
            "objective": problem.objective,
        },
        "command_line": argv if argv is not None else sys.argv,
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment_fingerprint": collect_fingerprint(),
        "status": "preliminary until the figures freeze",
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


# ======================================================================================
# Command line
# ======================================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Dirichlet policy with PPO.")
    parser.add_argument("--config", required=True, help="configuration of the frozen recipe, e.g. A_gbm or B_L12")
    parser.add_argument("--seed", type=int, required=True, help=f"training seed, inside the band {TRAINING_BAND}")
    parser.add_argument("--total-timesteps", type=int, default=None, help="length of the run")
    parser.add_argument(
        "--cost-bp", type=float, default=None, help="transaction cost to train under; the base case when omitted"
    )
    parser.add_argument("--shaping", action="store_true", help="add the potential-based term (diagnostic, D74)")
    parser.add_argument("--progress-bar", action="store_true", help="show a progress bar")
    args = parser.parse_args(argv)

    try:
        metrics = train(
            args.config,
            args.seed,
            args.total_timesteps,
            cost_bp=args.cost_bp,
            shaping=args.shaping,
            argv=argv if argv is not None else sys.argv,
            progress_bar=args.progress_bar,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(
        f"config {metrics['configuration']}  cost {metrics['cost_bp']:g} bp  "
        f"seed {metrics['seed']}  shaping {metrics['shaping']}"
    )
    print(f"{metrics['total_timesteps']:,} steps in {metrics['seconds_elapsed']:.1f} s")
    print(f"seconds per 100k steps        {metrics['seconds_per_100k_steps']:.1f}")
    print(f"held-out P(W_T >= G)          {metrics['final_held_out_probability']:.4f}")
    print(f"random policy, same scenarios {metrics['random_policy_probability']:.4f}")
    print(f"improvement over random       {metrics['improvement_over_random']:+.4f}")
    print(f"\nwrote {RUNS_DIRECTORY / metrics['run_directory'] / f'seed_{metrics['seed']}'}")
    print("PRELIMINARY: not frozen until the code freeze of 10 October.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
