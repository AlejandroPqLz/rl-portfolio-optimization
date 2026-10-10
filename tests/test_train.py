"""Tests of the agent: the Dirichlet policy and the training run.

The distribution is the algorithmic contribution of the work (D27), and it is the one component
not taken from a library, so it is tested as a probability distribution and not merely as code
that runs: the density integrates to one, every sample is a portfolio, the entropy is the closed
form, and the log-probability survives the action that would otherwise end a run, a weight that
rounds to zero in single precision.

The second thing tested here is the seed discipline. Three disjoint bands are what separate
training from evaluation (D23), and nothing downstream can detect a violation.
"""

# Imports
# =====================================================================
import inspect
import json

import numpy as np
import pytest
import torch as th
import yaml
from stable_baselines3 import PPO
from torch.distributions import Dirichlet

from portfolio_manager.env import GeneratorSpec, load_problem
from portfolio_manager.generator import config_path
from portfolio_manager.train import (
    ENV_SEED_STRIDE,
    TRAINING_BAND,
    DirichletActorCriticPolicy,
    DirichletDistribution,
    load_agent_config,
    make_training_env,
    policy_from_model,
    probability_of_reaching_goal,
    run_directory_name,
    train,
    uniform_simplex_policy,
)

# Constants
# =====================================================================
SEED = 0
ALPHA_MIN = 1.1
QUADRATURE_NODES = 200_001
"""Nodes of the trapezoidal rule behind the normalisation check. Odd, so the midpoint is a node."""


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def agent():
    """The agent configuration as it stands in the tree."""
    return load_agent_config()


@pytest.fixture(scope="module")
def problem():
    """The base case."""
    return load_problem()


def _distribution(logits: th.Tensor, alpha_min: float = ALPHA_MIN) -> DirichletDistribution:
    """A Dirichlet parameterised by the given logits."""
    return DirichletDistribution(logits.shape[-1], alpha_min).proba_distribution(logits)


# ======================================================================================
# The configuration
# ======================================================================================
def test_the_declared_hyperparameters_are_the_library_defaults(agent):
    """D57: the claim that nothing was tuned, checked against the installed signature.

    The file was written by reading this signature rather than by transcription. Comparing it back
    turns "these are the Stable-Baselines3 2.9.0 defaults" from an assertion in prose into a
    property of the repository, and catches a library upgrade that moves a default underneath the
    reported results.
    """
    defaults = inspect.signature(PPO.__init__).parameters
    for name, value in agent.ppo.items():
        assert name in defaults, f"{name} is not a PPO argument"
        assert value == defaults[name].default, f"{name} is not the library default"


def test_no_hyperparameter_outside_the_defaults_is_passed(agent):
    """Anything PPO takes that is absent from the block is left at its default by omission."""
    managed = {"policy", "env", "seed", "device", "tensorboard_log", "policy_kwargs", "verbose"}
    for name in agent.ppo:
        assert name not in managed, f"{name} is set twice: in the config and by train.py"


def test_the_concentration_floor_is_above_one(agent):
    """D57. Below one the density diverges at the faces of the simplex."""
    assert agent.alpha_min > 1.0


def test_a_floor_at_or_below_one_is_rejected(tmp_path):
    """The constraint is structural, so a configuration violating it must not load."""
    raw = yaml.safe_load(config_path("ppo").read_text(encoding="utf-8"))
    raw["policy"]["alpha_min"] = 1.0
    path = tmp_path / "ppo.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly above one"):
        load_agent_config(path)
    with pytest.raises(ValueError, match="strictly above one"):
        DirichletDistribution(4, 0.5)


def test_the_device_is_the_declared_one(agent):
    """CPU: MPS is slower for networks this size and is not bit-reproducible across versions."""
    assert agent.device == "cpu"


# ======================================================================================
# The Dirichlet as a probability distribution
# ======================================================================================
def test_the_density_integrates_to_one():
    """Numerical quadrature on a two-asset case, which is the whole claim about ``log_prob``.

    With two assets the simplex is the segment ``x, 1 - x``, so the Dirichlet density is a Beta
    density in one variable and the integral is computable directly. If the log-probability were
    off by a normalising constant, or were a density with respect to the wrong measure, the policy
    gradient would be scaled by a factor the clipping of PPO is not designed to absorb, and nothing
    downstream would reveal it.
    """
    for logits in (th.tensor([0.0, 0.0]), th.tensor([1.5, -0.5]), th.tensor([-2.0, 2.0])):
        distribution = _distribution(logits)
        x = th.linspace(0.0, 1.0, QUADRATURE_NODES, dtype=th.float64)[1:-1]
        actions = th.stack([x, 1.0 - x], dim=-1).to(th.float32)
        density = th.exp(distribution.log_prob(actions)).to(th.float64)
        integral = th.trapezoid(density, x)
        assert integral.item() == pytest.approx(1.0, abs=2e-3), logits.tolist()


def test_the_concentration_is_softplus_plus_the_floor():
    """``alpha = softplus(logits) + alpha_min``: positive always, and bounded below by the floor."""
    logits = th.tensor([[-10.0, 0.0, 2.0, 50.0]])
    distribution = _distribution(logits)
    expected = th.nn.functional.softplus(logits) + ALPHA_MIN
    th.testing.assert_close(distribution.concentration, expected)
    assert distribution.concentration.min().item() > 1.0


def test_every_sample_is_a_portfolio():
    """On the simplex by construction, so the projection of the environment is the identity.

    This is the point of the Dirichlet. A Gaussian would need a transformation to reach the
    simplex, and that transformation is what deforms the density and miscalibrates the entropy.
    """
    distribution = _distribution(th.randn(4096, 4, generator=th.Generator().manual_seed(SEED)))
    samples = distribution.sample()
    assert samples.shape == (4096, 4)
    assert samples.min().item() >= 0.0
    th.testing.assert_close(samples.sum(dim=-1), th.ones(4096), atol=1e-6, rtol=0.0)


def test_the_deterministic_action_is_the_dirichlet_mean():
    """Declared in ``config/ppo.yml`` as ``deterministic_action: mean``."""
    logits = th.tensor([[0.3, -1.0, 2.0, 0.0]])
    distribution = _distribution(logits)
    alpha = distribution.concentration
    th.testing.assert_close(distribution.mode(), alpha / alpha.sum(dim=-1, keepdim=True))
    th.testing.assert_close(distribution.mode().sum(dim=-1), th.ones(1))


def test_the_deterministic_action_is_not_the_density_mode():
    """A deliberate choice, so it is pinned rather than left to drift.

    With ``alpha_min > 1`` the density mode exists, at ``(alpha - 1) / (sum(alpha) - N)``. It is
    more extreme than the mean and moves discontinuously as a component approaches one; the mean is
    the barycentre of what the policy would actually play.
    """
    distribution = _distribution(th.tensor([[3.0, -1.0, -1.0, -1.0]]))
    alpha = distribution.concentration
    density_mode = (alpha - 1.0) / (alpha.sum(dim=-1, keepdim=True) - alpha.shape[-1])
    assert not th.allclose(distribution.mode(), density_mode)
    assert density_mode.max() > distribution.mode().max()


def test_the_entropy_is_the_closed_form():
    """Available exactly, which is why the entropy bonus of PPO regularises what it appears to."""
    logits = th.tensor([[0.5, -0.5, 1.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    distribution = _distribution(logits)
    reference = Dirichlet(th.nn.functional.softplus(logits) + ALPHA_MIN)
    th.testing.assert_close(distribution.entropy(), reference.entropy())
    assert th.isfinite(distribution.entropy()).all()


def test_a_more_concentrated_policy_has_less_entropy():
    """The sign of the quantity, which an implementation can get backwards and still run."""
    diffuse = _distribution(th.tensor([[-5.0, -5.0, -5.0, -5.0]])).entropy()
    peaked = _distribution(th.tensor([[8.0, -5.0, -5.0, -5.0]])).entropy()
    assert peaked.item() < diffuse.item()


def test_the_log_probability_survives_a_weight_that_rounds_to_zero():
    """The failure this clamp exists to prevent.

    A Dirichlet assigns no density to a point with a zero component, and at the concentrations this
    policy reaches a sampled weight small enough to round to zero in single precision is ordinary.
    Left alone the log-probability is minus infinity, which propagates through the ratio into the
    gradient and ends the run with no indication of the cause.
    """
    distribution = _distribution(th.tensor([[1.0, 1.0, 1.0, 1.0]]))
    degenerate = th.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert not th.isfinite(Dirichlet(distribution.concentration, validate_args=False).log_prob(degenerate)).all()
    assert th.isfinite(distribution.log_prob(degenerate)).all()


def test_the_clamp_barely_moves_an_ordinary_action():
    """It must rescue the degenerate case without perturbing the common one."""
    distribution = _distribution(th.tensor([[0.0, 0.0, 0.0, 0.0]]))
    action = th.tensor([[0.25, 0.25, 0.25, 0.25]])
    reference = Dirichlet(distribution.concentration).log_prob(action)
    th.testing.assert_close(distribution.log_prob(action), reference, atol=1e-5, rtol=0.0)


def test_actions_and_log_probabilities_come_back_together():
    """``log_prob_from_params`` is what the rollout collection calls."""
    distribution = DirichletDistribution(4, ALPHA_MIN)
    actions, log_prob = distribution.log_prob_from_params(th.zeros(16, 4))
    assert actions.shape == (16, 4)
    assert log_prob.shape == (16,)
    assert th.isfinite(log_prob).all()
    th.testing.assert_close(actions.sum(dim=-1), th.ones(16), atol=1e-6, rtol=0.0)


def test_parameters_give_a_deterministic_action_on_request():
    """``actions_from_params`` with ``deterministic`` must not sample."""
    distribution = DirichletDistribution(4, ALPHA_MIN)
    logits = th.tensor([[1.0, 0.0, -1.0, 0.5]])
    first = distribution.actions_from_params(logits, deterministic=True)
    again = distribution.actions_from_params(logits, deterministic=True)
    th.testing.assert_close(first, again)


# ======================================================================================
# The policy wiring
# ======================================================================================
@pytest.fixture(scope="module")
def model(agent, problem) -> PPO:
    """An untrained model, enough to exercise the wiring without learning anything."""
    environment = make_training_env(GeneratorSpec(mode="A"), SEED, 2, problem=problem)
    return PPO(
        DirichletActorCriticPolicy,
        environment,
        seed=SEED,
        device=agent.device,
        policy_kwargs={"alpha_min": agent.alpha_min, "log_prob_epsilon": agent.log_prob_epsilon},
        n_steps=64,
        batch_size=32,
    )


def test_the_policy_carries_the_dirichlet(model, agent):
    """``ActorCriticPolicy`` raises on an unrecognised distribution, so both hooks are overridden."""
    assert isinstance(model.policy.action_dist, DirichletDistribution)
    assert model.policy.action_dist.alpha_min == agent.alpha_min
    assert model.policy.action_net.out_features == 4


def test_the_algorithm_itself_is_untouched(model):
    """D26: the customisation is a distribution and a policy, never the optimiser.

    No hypothesis of this work depends on the optimiser, so the library's PPO is used unmodified
    and unsubclassed. The update rule, the rollout buffer and the advantage estimation are theirs.
    """
    import stable_baselines3.common.on_policy_algorithm as on_policy

    assert type(model) is PPO
    assert PPO.train.__module__ == "stable_baselines3.ppo.ppo"
    assert PPO.collect_rollouts is on_policy.OnPolicyAlgorithm.collect_rollouts


def test_predicted_actions_are_portfolios(model):
    """What the environment receives, and what ``evaluate.py`` will score."""
    observations = np.array([[0.0, 0.0, 0.8], [0.5, 0.5, 0.4], [2.0, 0.99, 0.01]], dtype=np.float32)
    for deterministic in (True, False):
        actions, _ = model.predict(observations, deterministic=deterministic)
        assert actions.shape == (3, 4)
        assert actions.min() >= 0.0
        np.testing.assert_allclose(actions.sum(axis=-1), 1.0, atol=1e-5)


def test_the_deterministic_prediction_is_reproducible(model):
    """A deterministic action must not depend on the state of any random stream."""
    observations = np.array([[0.3, 0.2, 0.6]], dtype=np.float32)
    first, _ = model.predict(observations, deterministic=True)
    again, _ = model.predict(observations, deterministic=True)
    np.testing.assert_array_equal(first, again)


def test_the_model_exposes_the_comparator_interface(model, problem):
    """Section 2b: the agent is scored by the same code as every comparator."""
    policy = policy_from_model(model)
    weights = policy(np.zeros((8, 3), dtype=np.float32), 0)
    assert weights.shape == (8, 4)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


def test_evaluating_actions_gives_finite_gradients(model):
    """The quantity PPO differentiates, on observations spanning the state space."""
    observations = th.tensor([[0.0, 0.0, 0.8], [1.5, 0.5, 0.4], [4.0, 1.0, 0.0]])
    actions = th.tensor([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    values, log_prob, entropy = model.policy.evaluate_actions(observations, actions)
    assert th.isfinite(values).all()
    assert th.isfinite(log_prob).all()
    assert th.isfinite(entropy).all()
    log_prob.sum().backward()
    gradients = [p.grad for p in model.policy.parameters() if p.grad is not None]
    assert gradients and all(th.isfinite(g).all() for g in gradients)


# ======================================================================================
# Seed discipline
# ======================================================================================
def test_sub_environments_own_disjoint_streams(problem):
    """``seed * 100 + index``, so two runs cannot share a stream."""
    environment = make_training_env(GeneratorSpec(mode="A"), 3, 4, problem=problem)
    seeds = [sub.unwrapped._rng for sub in environment.envs]
    assert len({id(stream) for stream in seeds}) == 4
    scenarios = []
    for sub in environment.envs:
        sub.reset()
        scenarios.append(sub.unwrapped._scenario.copy())
    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            assert not np.array_equal(scenarios[i], scenarios[j])
    environment.close()


def test_a_seed_outside_the_training_band_is_refused(problem):
    """D23. A training stream reaching the evaluation scenarios is undetectable downstream."""
    for seed in (TRAINING_BAND[1] + 1, 20260809, -1):
        with pytest.raises(ValueError, match="training seed must lie in the band"):
            make_training_env(GeneratorSpec(mode="A"), seed, 2, problem=problem)


def test_too_many_sub_environments_are_refused(problem):
    """Beyond the stride the streams of consecutive runs would overlap."""
    with pytest.raises(ValueError, match="collide"):
        make_training_env(GeneratorSpec(mode="A"), 0, ENV_SEED_STRIDE + 1, problem=problem)


def test_the_random_policy_is_uniform_on_the_simplex(problem):
    """The reference the pre-declared shaping rule is stated against (D74, A40)."""
    policy = uniform_simplex_policy(np.random.default_rng(1000), 4)
    weights = policy(np.zeros((4096, 3), dtype=np.float32), 0)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
    # Uniform on the simplex means every asset averages 1/N, with no column favoured.
    np.testing.assert_allclose(weights.mean(axis=0), 0.25, atol=0.02)
    assert weights.min() < 0.01 and weights.max() > 0.9


def test_the_held_out_probability_is_a_probability(problem):
    """The primary metric, on a tensor the agent never trains on."""
    from portfolio_manager.generator import generate

    scenarios = generate("A", 64, problem.horizon_months, 1000)
    value = probability_of_reaching_goal(
        scenarios, uniform_simplex_policy(np.random.default_rng(1000), 4), problem
    )
    assert 0.0 <= value <= 1.0


# ======================================================================================
# One run, end to end
# ======================================================================================
@pytest.mark.slow
def test_a_short_run_writes_everything_needed_to_read_it(tmp_path):
    """The smoke run: the whole pipeline, short enough to be a test and complete enough to be one.

    Monday morning is read from ``metrics.json`` alone, so the fields that make a run interpretable
    without the terminal it was launched from are what this asserts.
    """
    metrics = train("A_gbm", SEED, 2048, runs_directory=tmp_path, argv=["test"])
    # The directory names the cost the agent was trained under, so two runs at different costs
    # cannot be confused once the terminal that launched them is gone.
    output = tmp_path / metrics["run_directory"] / f"seed_{SEED}"
    assert metrics["run_directory"] == run_directory_name("A_gbm", metrics["cost_bp"])

    assert (output / "policy.zip").is_file()
    assert (output / "tensorboard").is_dir()
    written = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert written == metrics

    assert metrics["total_timesteps"] >= 2048
    assert metrics["seconds_per_100k_steps"] > 0.0
    assert 0.0 <= metrics["final_held_out_probability"] <= 1.0
    assert 0.0 <= metrics["random_policy_probability"] <= 1.0
    assert metrics["improvement_over_random"] == pytest.approx(
        metrics["final_held_out_probability"] - metrics["random_policy_probability"]
    )
    # The held-out tensor is in the validation band and the training streams are in the training one.
    assert metrics["held_out"]["seed"] == 1000 + SEED
    assert metrics["training_env_seeds"][0] == SEED * ENV_SEED_STRIDE
    assert max(metrics["training_env_seeds"]) <= TRAINING_BAND[1]
    assert metrics["agent"]["policy"] == "DirichletActorCriticPolicy"
    assert metrics["environment_fingerprint"]["device"] == "cpu"
    assert metrics["status"].startswith("preliminary")
    assert metrics["shaping"] is False


@pytest.mark.slow
def test_a_saved_policy_can_be_reloaded_and_scored(tmp_path):
    """The batch saves policies on one night and they are read on another."""
    metrics = train("A_gbm", SEED, 2048, runs_directory=tmp_path, argv=["test"])
    reloaded = PPO.load(
        tmp_path / metrics["run_directory"] / f"seed_{SEED}" / "policy.zip",
        device="cpu",
        custom_objects={"policy_class": DirichletActorCriticPolicy},
    )
    weights = policy_from_model(reloaded)(np.zeros((4, 3), dtype=np.float32), 0)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


@pytest.mark.slow
def test_the_shaping_run_leaves_the_wealth_dynamics_alone(tmp_path):
    """The diagnostic of D74 must change the signal and not the problem."""
    metrics = train("A_gbm", SEED, 2048, shaping=True, runs_directory=tmp_path, argv=["test"])
    assert metrics["shaping"] is True
    assert metrics["cost_bp"] == load_problem().cost_bp
    assert 0.0 <= metrics["final_held_out_probability"] <= 1.0
