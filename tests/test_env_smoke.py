"""Smoke test for the reinforcement-learning stack.

Trains PPO on CartPole-v1 with an explicit seed. This does not exercise the project's own environment or reward: it
checks that NumPy, PyTorch, Gymnasium and Stable-Baselines3 are mutually consistent, which is the failure mode that
otherwise surfaces hours later while writing ``env.py``.
"""

# Imports
# =====================================================================
import gymnasium as gym
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# TODO: change to import from config values when created and delete DEVICE constant from environment_fingerprint.py
from portfolio_manager.environment_fingerprint import DEVICE

# Constants
# =====================================================================
ENV_ID = "CartPole-v1"
SEED = 0
TOTAL_TIMESTEPS = 10_000
N_EVAL_EPISODES = 20
TRAINED_FLOOR = 100.0  # Weak lower bound; PPO clears it comfortably after 10k steps.


@pytest.mark.slow
def test_ppo_training_updates_the_policy(capsys: pytest.CaptureFixture[str]) -> None:
    """Train PPO on CartPole-v1 and check that the policy changed and that the mean return is above a weak floor.
    This is a smoke test for the RL stack, not a benchmark for PPO.

    Args:
        capsys: Pytest fixture to capture stdout/stderr.
    Returns:
        None. Raises AssertionError if the test fails.
    """
    eval_env = Monitor(gym.make(ENV_ID))
    eval_env.reset(seed=SEED + 1_000)  # to avoid contaminating the training seed with evaluation noise

    model = PPO("MlpPolicy", ENV_ID, seed=SEED, device=DEVICE, verbose=0)

    before_train = {name: tensor.detach().clone() for name, tensor in model.policy.state_dict().items()}
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    after = model.policy.state_dict()

    # Check that at least one parameter changed and that the mean return is above a weak floor.
    largest_shift = max(
        float(torch.max(torch.abs(after[name] - tensor)).item()) for name, tensor in before_train.items()
    )
    trained_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES)
    eval_env.close()

    with capsys.disabled():
        print(f"\nlargest parameter shift: {largest_shift:.3e}")
        print(f"mean return after training: {trained_reward:.1f}")

    assert model.num_timesteps >= TOTAL_TIMESTEPS, f"training stopped at {model.num_timesteps} of {TOTAL_TIMESTEPS}"
    assert largest_shift > 0.0, (
        "no parameter changed during training: gradients are null, which points to a broken PyTorch or "
        "linear-algebra backend rather than to a reinforcement-learning problem"
    )
    assert trained_reward >= TRAINED_FLOOR, f"mean return is {trained_reward:.1f}, below {TRAINED_FLOOR}"
