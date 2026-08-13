"""Environment fingerprint: capture and comparison.

The scenario tensors produced by the generator are bit-for-bit reproducible only inside a documented environment.
Mode B (block bootstrap) draws integers and copies rows, so it is portable; mode A (geometric Brownian motion)
requires a Cholesky factorisation whose last bit depends on the BLAS implementation. This module is the single
source of truth for what "the documented environment" means, and is consumed from three places:

* the stack smoke test, which records the fingerprint that fills ``evaluation.freeze_environment`` in the frozen
  test-set recipe;
* the generator, when it registers a produced tensor;
* the regression test, which compares tensor hashes exactly when the current environment matches the frozen one
  and skips with an explanatory message otherwise.

Usage:
-----
    uv run python -m portfolio_manager.environment_fingerprint > results/environment/environment_fingerprint.json
"""

# Imports
# =====================================================================
import json
import platform
import sys
from collections.abc import Mapping
from typing import Any

import gymnasium
import numpy as np
import stable_baselines3
import torch

# Constants
# =====================================================================
HASH_CRITICAL_KEYS: tuple[str, ...] = ("python_version", "numpy_version", "blas_backend", "machine", "device")
DEVICE = "cpu"  # MPS is slower for small MLP policies and not bit-reproducible across PyTorch versions.


def collect_fingerprint() -> dict[str, Any]:
    """Return the environment descriptors that condition numerical output.

    Every field is retrieved defensively: an unavailable descriptor degrades to ``None`` instead of raising, so that
    recording a fingerprint never aborts a run that would otherwise have succeeded.
    """

    try:
        blas_backend = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:  # pylint: disable=broad-except
        print("WARNING: could not determine BLAS backend")
        blas_backend = None

    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "blas_backend": blas_backend,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "gymnasium_version": gymnasium.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "device": DEVICE,
    }


def compare_with_frozen(frozen: Mapping[str, Any]) -> list[str]:
    """Return the hash-critical keys that differ from the frozen environment.

    Args:
        frozen: The ``evaluation.freeze_environment`` mapping read from the frozen test-set recipe.

    Returns:
        list of str: Empty when an exact hash comparison is legitimate. Otherwise, the keys that differ,
        so that the caller can explain the mismatch instead of reporting an anonymous failure.
    """
    current = collect_fingerprint()
    return [key for key in HASH_CRITICAL_KEYS if str(frozen.get(key)) != str(current.get(key))]


def main() -> None:
    """Print the fingerprint as JSON on stdout."""
    print(json.dumps(collect_fingerprint(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
