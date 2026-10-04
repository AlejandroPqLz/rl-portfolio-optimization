"""Scenario generator: from the consolidated historical matrix to synthetic return tensors.

Contract
--------
``generate(mode, n_scenarios, horizon, seed, *, block_length=None, params=None) -> np.ndarray``
returns ``(n_scenarios, horizon, 4)`` of **simple** monthly returns in ``float64``, with the columns
in the order of ``data/returns_matrix.csv``. All randomness of one call comes from a single
``numpy.random.default_rng(seed)`` stream, so one seed fixes one tensor inside the documented
environment.

Two modes, both specified in thesis subsection 4.3.3 (``subsec:desarrollo-generador``):

Mode A (parametric)
    Monthly log gross returns are multivariate normal with the mean vector and covariance stored in
    ``config/gbm_params.yml``; the simple return is ``exp(x) - 1``. Months are independent by
    construction, which is what makes the state of ``eq:estado`` exactly Markovian and the dynamic
    programming table the optimal reference of H1.

Mode B (stationary block bootstrap, Politis and Romano 1994)
    Whole rows of the historical matrix are copied in contiguous runs of geometric length with mean
    ``L``, from a uniformly drawn start month, with circular indexing and replacement, until the
    horizon is filled. Copying whole rows keeps the four assets synchronised, so the contemporaneous
    correlation structure is preserved exactly; the dependence at lag ``k`` is the historical one
    attenuated by ``(1 - 1/L)**k`` (``eq:bootstrap-autocorr``).

Decisions involved
------------------
D23 (frozen test set, independent seed), D39 (the matrix digest is written but never compared; only
the scenario tensor digest is a frozen reference, and only inside the frozen environment), D74
(single build block, 4 October 2026).

Command line
------------
    uv run python -m portfolio_manager.generator --fit-gbm
    uv run python -m portfolio_manager.generator --test-set
    uv run python -m portfolio_manager.generator --stylized-facts --mode B --block-length 12
    uv run python -m portfolio_manager.generator --stylized-facts --mode A
"""

# Imports
# =====================================================================
import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from portfolio_manager.environment_fingerprint import collect_fingerprint, compare_with_frozen

# Constants
# =====================================================================
CONFIG_DIR = Path("config")
MATRIX_PATH = Path("data/returns_matrix.csv")
MATRIX_METADATA_PATH = Path("data/returns_matrix.json")
TEST_SET_DIRECTORY = Path("results/test_set")
CHECKS_DIRECTORY = Path("results/checks")

MODE_PARAMETRIC = "A"
MODE_BOOTSTRAP = "B"

SCENARIO_CHUNK = 2048
"""Scenarios drawn per batch in mode A. Bounds peak memory without touching the random stream:
``Generator.standard_normal`` fills in C order, so a chunked draw consumes the same variates in the
same order as a single draw of the full tensor."""

INDEX_CHUNK = 1024
"""Scenarios per batch when resolving bootstrap block membership. Pure memory bound: the membership
test is ``(chunk, horizon, horizon)`` booleans."""

CHECK_SCENARIOS = 2000
"""Scenarios used by ``--stylized-facts``. Fixed by thesis 4.3.3 as the size of the verification,
not as a parameter of the generator; overridable from the command line."""

CORRELATION_TOLERANCE = 0.05
"""Thesis 4.3.3, check 2: maximum absolute deviation of a synthetic contemporaneous correlation."""

AUTOCORRELATION_TOLERANCE = 0.05
"""Thesis 4.3.3, check 3: maximum absolute deviation from ``(1 - 1/L) * rho_1``. The mode-A arm of
the same check uses it as the band around zero, so both arms are measured on one scale."""

ZERO_KURTOSIS_TOLERANCE = 0.10
"""Band around zero for the excess kurtosis of equity in mode A: under a tenth of the historical
1.02, which is the claim being made, namely that the lognormal does not reproduce the heavy tail.

Why an absolute band and not two standard errors. The thesis fixes two standard errors for the
marginal mean and standard deviation, where the comparison is against a historical value estimated
from 420 months and the historical error dominates, leaving the check with a wide margin. It says
only "compatible with zero" for the two quantities that are genuinely zero under mode A, and there
the two-standard-error reading would turn an acceptance check into a significance test on a true
null: it rejects five per cent of the time per statistic by construction, and with six such
statistics the suite would fail about a quarter of the time on nothing but the seed. The band is
therefore declared in advance, as a size that matters rather than a size that is detectable, and
the standard-error reading is still reported next to it as ``z_score`` for subsection 4.4."""

SIGMA_EIGENVALUE_FLOOR = 0.0
"""Eigenvalues below this are clipped when the covariance is singular and Cholesky is unavailable."""


# ======================================================================================
# Configuration and file helpers
# ======================================================================================
def config_path(stem: str) -> Path:
    """Return the configuration file for ``stem``.

    ``.yml`` is the extension this repository uses and the one new files are created with; ``.yaml``
    is still accepted on read so that a file carried over under the other spelling is found rather
    than silently ignored, which is the failure mode this helper exists to prevent.

    Args:
        stem: File name without extension, for example ``test_set``.

    Returns:
        Path to the existing file, or the ``.yml`` path when neither exists yet.
    """
    for suffix in (".yml", ".yaml"):
        candidate = CONFIG_DIR / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return CONFIG_DIR / f"{stem}.yml"


def sha256_of_file(path: Path) -> str:
    """SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor: np.ndarray) -> str:
    """Digest of a scenario tensor, exactly as ``evaluation.hash_procedure`` states it.

    Args:
        tensor: Scenario tensor.

    Returns:
        ``sha256`` of ``arr.astype('float64', order='C').tobytes()``.
    """
    return hashlib.sha256(tensor.astype("float64", order="C").tobytes()).hexdigest()


def git_commit() -> str | None:
    """Current commit hash, or ``None`` outside a repository. Never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def write_sidecar(result_path: Path, payload: dict[str, Any], argv: list[str] | None = None) -> Path:
    """Write the JSON side-car that every result file carries.

    Args:
        result_path: Path of the result file the side-car documents.
        payload: Fields specific to the result.
        argv: Command line that produced it; defaults to the running one.

    Returns:
        Path of the side-car.
    """
    sidecar = result_path.with_suffix(".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "result_file": str(result_path),
        "command_line": argv if argv is not None else sys.argv,
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment_fingerprint": collect_fingerprint(),
        **payload,
    }
    sidecar.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


# ======================================================================================
# Historical matrix
# ======================================================================================
def load_matrix(path: Path = MATRIX_PATH) -> tuple[tuple[str, ...], np.ndarray]:
    """Read the consolidated matrix of monthly simple returns.

    Args:
        path: Path to ``data/returns_matrix.csv``.

    Returns:
        Tuple ``(columns, values)`` with the column names in file order and a ``(D, 4)``
        ``float64`` array.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run the consolidation stage first")
    frame = pd.read_csv(path, index_col=0)
    values = frame.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"{path}: expected a two-dimensional matrix, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: the matrix contains non-finite entries")
    return tuple(str(name) for name in frame.columns), values


def load_matrix_metadata(path: Path = MATRIX_METADATA_PATH) -> dict[str, Any]:
    """Read ``data/returns_matrix.json``, the recorded description of the matrix."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run the consolidation stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_test_set(path: Path | None = None) -> dict[str, Any]:
    """Read the frozen test-set recipe."""
    path = path or config_path("test_set")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def assert_columns(columns: tuple[str, ...], test_set: dict[str, Any] | None = None) -> None:
    """Fail unless the CSV header equals the normative asset order of the frozen recipe.

    The order indexes portfolio weights, correlation matrices, action components and figure legends.
    A silent reordering corrupts every result without raising anywhere else, so it is checked at the
    single point where history enters the system.

    Args:
        columns: Column names as read from the CSV header.
        test_set: Parsed recipe; read from disk when omitted.
    """
    test_set = test_set if test_set is not None else load_test_set()
    declared = tuple(str(name) for name in test_set["data"]["assets"])
    if columns != declared:
        raise ValueError(
            f"column order mismatch: the matrix header is {list(columns)} and the frozen recipe "
            f"declares {list(declared)}. Align 'data.assets' with the header without reordering it."
        )


# ======================================================================================
# Mode A: parametric
# ======================================================================================
@dataclass(frozen=True)
class GbmParams:
    """Mean vector and covariance of the monthly log gross returns.

    The parameters are estimated once by ``--fit-gbm`` and read back from
    ``config/gbm_params.yml``; the generator never re-estimates them (thesis 4.3.3). An arbitrary
    ``mu`` and ``sigma`` are accepted, including a degenerate asset with zero variance, which is
    what the Browne validation configuration needs (D44).

    Attributes:
        assets: Asset names in the normative order.
        mu: ``(N,)`` mean of the monthly log gross returns.
        sigma: ``(N, N)`` covariance of the monthly log gross returns.
    """

    assets: tuple[str, ...]
    mu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=np.float64).reshape(-1)
        sigma = np.asarray(self.sigma, dtype=np.float64)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)
        n_assets = mu.size
        if sigma.shape != (n_assets, n_assets):
            raise ValueError(f"sigma has shape {sigma.shape}, expected ({n_assets}, {n_assets})")
        if len(self.assets) != n_assets:
            raise ValueError(f"{len(self.assets)} asset names for {n_assets} parameters")
        if not np.allclose(sigma, sigma.T, atol=1e-15, rtol=0.0):
            raise ValueError("sigma is not symmetric")

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.mu.size

    def factor(self) -> np.ndarray:
        """Return ``F`` with ``F @ F.T == sigma``, applied to independent standard normals.

        Cholesky is used whenever it succeeds, which is the factorisation the thesis declares and
        the one the frozen digests were produced with. It raises on a singular covariance, which
        happens as soon as an asset is given zero variance; the fallback is an eigendecomposition
        with the eigenvalues clipped at zero, so a degenerate asset simply contributes a zero row.

        Returns:
            ``(N, N)`` factor of the covariance.
        """
        try:
            return np.linalg.cholesky(self.sigma)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(self.sigma)
            eigenvalues = np.clip(eigenvalues, SIGMA_EIGENVALUE_FLOOR, None)
            return eigenvectors * np.sqrt(eigenvalues)

    def to_mapping(self) -> dict[str, Any]:
        """Serialisable form, with plain Python floats."""
        return {
            "assets": list(self.assets),
            "mu": [float(value) for value in self.mu],
            "sigma": [[float(value) for value in row] for row in self.sigma],
        }

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "GbmParams":
        """Rebuild from the parsed YAML document."""
        return cls(
            assets=tuple(str(name) for name in mapping["assets"]),
            mu=np.asarray(mapping["mu"], dtype=np.float64),
            sigma=np.asarray(mapping["sigma"], dtype=np.float64),
        )


def fit_gbm(matrix: np.ndarray, columns: tuple[str, ...]) -> GbmParams:
    """Estimate the parametric mode from the historical matrix.

    Args:
        matrix: ``(D, N)`` matrix of monthly **simple** returns.
        columns: Asset names in the normative order.

    Returns:
        Parameters of the monthly log gross returns, ``log(1 + r)``.
    """
    log_gross = np.log1p(matrix)
    return GbmParams(assets=columns, mu=log_gross.mean(axis=0), sigma=np.cov(log_gross, rowvar=False))


def load_gbm_params(path: Path | None = None) -> GbmParams:
    """Read ``config/gbm_params.yml``."""
    path = path or config_path("gbm_params")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: run 'python -m portfolio_manager.generator --fit-gbm' first"
        )
    return GbmParams.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def write_gbm_params(params: GbmParams, metadata: dict[str, Any], path: Path | None = None) -> tuple[Path, bool]:
    """Write ``config/gbm_params.yml`` with its provenance header.

    Re-running the estimate on an unchanged matrix must not disturb the file. The digest of this
    file is recorded in the frozen recipe, so letting ``estimated_on`` advance with the calendar
    would change the digest and force a recipe edit on a day when nothing was actually re-estimated.
    The stored date is therefore kept whenever the parameters and the source digest are identical.

    Args:
        params: Estimated parameters.
        metadata: Parsed ``data/returns_matrix.json``, source of the digest and the window.
        path: Destination; resolved from the configuration directory when omitted.

    Returns:
        Tuple ``(path, rewritten)``.
    """
    path = path or config_path("gbm_params")
    window = metadata["window"]
    document = {
        **params.to_mapping(),
        "source_matrix_sha256": metadata["sha256"],
        "window": f"{window['start']} / {window['end']}",
        "estimated_on": datetime.now(UTC).date().isoformat(),
    }

    if path.is_file():
        stored = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        unchanged = all(stored.get(key) == document[key] for key in document if key != "estimated_on")
        if unchanged:
            document["estimated_on"] = stored.get("estimated_on", document["estimated_on"])
    header = (
        "# Parameters of the parametric generator (mode A).\n"
        "# =============================================================================\n"
        "# Mean vector and covariance of the monthly LOG GROSS returns, log(1 + r), estimated once\n"
        "# from the consolidated historical matrix and never re-estimated at generation time\n"
        "# (thesis 4.3.3). The simple return handed to the environment is exp(x) - 1.\n"
        "#\n"
        "# Written by:\n"
        "#   uv run python -m portfolio_manager.generator --fit-gbm\n"
        "#\n"
        "# 'assets' repeats the normative column order of config/test_set.yml:data.assets. The\n"
        "# digest below identifies the matrix the estimate came from; per D39 it is informational\n"
        "# and is never compared against a frozen reference.\n"
        "# =============================================================================\n\n"
    )
    rendered = header + yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
    if path.is_file() and path.read_text(encoding="utf-8") == rendered:
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path, True


def _generate_parametric(
    rng: np.random.Generator, n_scenarios: int, horizon: int, params: GbmParams
) -> np.ndarray:
    """Draw mode-A scenarios. Months are independent by construction."""
    factor = params.factor()
    out = np.empty((n_scenarios, horizon, params.n_assets), dtype=np.float64)
    for low in range(0, n_scenarios, SCENARIO_CHUNK):
        high = min(low + SCENARIO_CHUNK, n_scenarios)
        normals = rng.standard_normal((high - low, horizon, params.n_assets))
        out[low:high] = np.expm1(params.mu + normals @ factor.T)
    return out


# ======================================================================================
# Mode B: stationary block bootstrap
# ======================================================================================
def bootstrap_indices(
    rng: np.random.Generator, n_scenarios: int, horizon: int, n_rows: int, block_length: float
) -> np.ndarray:
    """Row indices of the historical matrix drawn by the stationary block bootstrap.

    One scenario is built by repeating three steps until the horizon is filled: draw a start month
    uniformly over the ``n_rows`` available, draw a block length from a geometric distribution of
    mean ``block_length``, and take that many consecutive months. The last block is truncated,
    indexing is circular and sampling is with replacement. Both draws are taken in one call per
    kind, so each scenario consumes exactly ``2 * horizon`` variates whatever its block structure;
    the surplus is discarded, which keeps the stream independent of the realised lengths.

    Args:
        rng: The single stream of the call.
        n_scenarios: Number of scenarios.
        horizon: Months per scenario.
        n_rows: Rows of the historical matrix.
        block_length: Mean block length ``L``; the geometric parameter is ``p = 1 / L``.

    Returns:
        ``(n_scenarios, horizon)`` array of row indices into the historical matrix.
    """
    if not block_length or block_length <= 0.0:
        raise ValueError(f"block_length must be strictly positive, got {block_length!r}")
    if block_length < 1.0:
        raise ValueError(f"block_length is a mean number of months and cannot be below 1, got {block_length}")

    starts = rng.integers(0, n_rows, size=(n_scenarios, horizon))
    lengths = rng.geometric(1.0 / block_length, size=(n_scenarios, horizon))

    # Episode-time boundaries of each block. Lengths are at least one, so the cumulative sum always
    # reaches the horizon within the horizon blocks drawn and the last block is simply truncated.
    ends = np.cumsum(lengths, axis=1)
    block_starts = ends - lengths

    months = np.arange(horizon)
    indices = np.empty((n_scenarios, horizon), dtype=np.int64)
    for low in range(0, n_scenarios, INDEX_CHUNK):
        high = min(low + INDEX_CHUNK, n_scenarios)
        block_of_month = (ends[low:high, :, None] <= months[None, None, :]).sum(axis=1)
        rows = np.arange(high - low)[:, None]
        offset = months[None, :] - block_starts[low:high][rows, block_of_month]
        indices[low:high] = (starts[low:high][rows, block_of_month] + offset) % n_rows
    return indices


def _generate_bootstrap(
    rng: np.random.Generator, n_scenarios: int, horizon: int, matrix: np.ndarray, block_length: float
) -> np.ndarray:
    """Draw mode-B scenarios by copying whole rows, so the four assets stay synchronised."""
    indices = bootstrap_indices(rng, n_scenarios, horizon, matrix.shape[0], block_length)
    return matrix[indices]


# ======================================================================================
# Public entry point
# ======================================================================================
def generate(
    mode: str,
    n_scenarios: int,
    horizon: int,
    seed: int,
    *,
    block_length: float | None = None,
    params: GbmParams | None = None,
    matrix: np.ndarray | None = None,
    columns: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Produce a tensor of synthetic monthly simple returns.

    Args:
        mode: ``"A"`` for the parametric generator, ``"B"`` for the stationary block bootstrap.
        n_scenarios: Number of independent scenarios.
        horizon: Months per scenario.
        seed: Seed of the single ``default_rng`` stream of this call.
        block_length: Mean block length ``L``, required by mode B and rejected by mode A.
        params: Parameters for mode A; read from ``config/gbm_params.yml`` when omitted.
        matrix: Historical matrix for mode B; read from disk when omitted.
        columns: Column names matching ``matrix``; read from disk when omitted.

    Returns:
        ``(n_scenarios, horizon, N)`` array of simple monthly returns in ``float64``.
    """
    return generate_from_rng(
        np.random.default_rng(seed),
        mode,
        n_scenarios,
        horizon,
        block_length=block_length,
        params=params,
        matrix=matrix,
        columns=columns,
    )


def generate_from_rng(
    rng: np.random.Generator,
    mode: str,
    n_scenarios: int,
    horizon: int,
    *,
    block_length: float | None = None,
    params: GbmParams | None = None,
    matrix: np.ndarray | None = None,
    columns: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Same as :func:`generate`, drawing from a stream the caller owns and keeps.

    :func:`generate` is the entry point for a tensor built once from a declared seed, which is what
    the frozen test set needs. Training is the other case: a sub-environment produces a new scenario
    at every ``reset`` and must advance a single stream across the whole run rather than reseed per
    episode, because reseeding would make the episodes a function of a derived seed sequence instead
    of one declared seed (thesis 4.3.4).

    Args:
        rng: The stream to draw from, advanced in place.
        mode: ``"A"`` or ``"B"``.
        n_scenarios: Number of independent scenarios.
        horizon: Months per scenario.
        block_length: Mean block length ``L``, required by mode B and rejected by mode A.
        params: Parameters for mode A; read from ``config/gbm_params.yml`` when omitted.
        matrix: Historical matrix for mode B; read from disk when omitted.
        columns: Column names matching ``matrix``; read from disk when omitted.

    Returns:
        ``(n_scenarios, horizon, N)`` array of simple monthly returns in ``float64``.
    """
    if n_scenarios <= 0 or horizon <= 0:
        raise ValueError(f"n_scenarios and horizon must be positive, got {n_scenarios} and {horizon}")

    if mode == MODE_PARAMETRIC:
        if block_length is not None:
            raise ValueError("mode A has no blocks: block_length must be None")
        if params is None:
            params = load_gbm_params()
            if columns is None:
                columns, _ = load_matrix()
            assert_columns(columns)
            if tuple(params.assets) != columns:
                raise ValueError(
                    f"config/gbm_params.yml declares {list(params.assets)} and the matrix header is "
                    f"{list(columns)}; re-run --fit-gbm"
                )
        return _generate_parametric(rng, n_scenarios, horizon, params)

    if mode == MODE_BOOTSTRAP:
        if block_length is None:
            raise ValueError("mode B requires block_length, the mean of the geometric distribution")
        if matrix is None:
            columns, matrix = load_matrix()
        if columns is not None:
            assert_columns(columns)
        return _generate_bootstrap(rng, n_scenarios, horizon, matrix, float(block_length))

    raise ValueError(f"unknown mode {mode!r}: this module implements {MODE_PARAMETRIC!r} and {MODE_BOOTSTRAP!r}")


def generate_configuration(
    name: str, test_set: dict[str, Any], *, matrix: np.ndarray | None = None, columns: tuple[str, ...] | None = None
) -> tuple[np.ndarray, int]:
    """Build one configuration of the frozen recipe with its normative effective seed.

    Args:
        name: Configuration key under ``configurations``.
        test_set: Parsed recipe.
        matrix: Historical matrix, to avoid re-reading it per configuration.
        columns: Column names matching ``matrix``.

    Returns:
        Tuple ``(tensor, effective_seed)``.
    """
    spec = test_set["configurations"][name]
    effective_seed = int(test_set["seeds"]["test_base"]) + int(spec["offset"])
    return (
        generate(
            str(spec["mode"]),
            int(spec["n_scenarios"]),
            int(spec["horizon_months"]),
            effective_seed,
            block_length=spec.get("mean_block_length_months"),
            matrix=matrix,
            columns=columns,
        ),
        effective_seed,
    )


# ======================================================================================
# Surgical YAML editing
# ======================================================================================
# The configuration files carry the audit trail of the project in their comments: re-dumping them
# through PyYAML would delete it. Every write below therefore patches the exact lines involved and
# leaves the rest of the text, comments included, byte for byte identical.
def _set_scalar(text: str, key: str, indent: int, value: str) -> str:
    """Replace the value of the single ``key`` appearing at the given indentation.

    Args:
        text: Full file text.
        key: Mapping key.
        indent: Number of leading spaces identifying the nesting level.
        value: Replacement text, which may carry its own inline comment.

    Returns:
        The patched text.
    """
    pattern = re.compile(rf"^{' ' * indent}{re.escape(key)}:[^\S\n]*(?P<old>[^\n]*)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one '{key}:' at indent {indent}, found {len(matches)}")
    match = matches[0]
    return f"{text[: match.start()]}{' ' * indent}{key}: {value}{text[match.end():]}"


def _set_block_map(text: str, key: str, indent: int, entries: dict[str, str]) -> str:
    """Replace a mapping written either as ``key: {}`` or as an indented block.

    Args:
        text: Full file text.
        key: Mapping key.
        indent: Number of leading spaces identifying the nesting level.
        entries: Entries to render, in insertion order.

    Returns:
        The patched text.
    """
    pattern = re.compile(rf"^{' ' * indent}{re.escape(key)}:[^\S\n]*[^\n]*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one '{key}:' at indent {indent}, found {len(matches)}")
    match = matches[0]

    # Absorb the lines belonging to the mapping: anything indented deeper than the key itself.
    end = match.end()
    while end < len(text):
        line_end = text.find("\n", end + 1)
        line_end = len(text) if line_end == -1 else line_end
        line = text[end + 1 : line_end]
        if line.strip() and not line.startswith(" " * (indent + 1)):
            break
        if not line.strip():
            break
        end = line_end

    pad = " " * indent
    if entries:
        body = "\n".join(f"{pad}  {name}: {value}" for name, value in entries.items())
        rendered = f"{pad}{key}:\n{body}"
    else:
        rendered = f"{pad}{key}: {{}}"
    return f"{text[: match.start()]}{rendered}{text[end:]}"


def apply_test_set_corrections(text: str, metadata: dict[str, Any], gbm_path: Path) -> tuple[str, list[str]]:
    """Apply the three corrections the frozen recipe needs before the first generation.

    The recipe was written in August, before any data existed. Three of its fields still point at
    paths and labels that the consolidation stage later replaced; aligning them is the documented
    precondition of the first build, not a reopening of the freeze. No configuration is added,
    removed, renamed or reordered.

    Args:
        text: Full text of the recipe.
        metadata: Parsed ``data/returns_matrix.json``.
        gbm_path: Path of ``config/gbm_params.yml``, whose digest fills the ``A_gbm`` slot.

    Returns:
        Tuple ``(text, applied)`` where ``applied`` lists the corrections actually written.
    """
    applied: list[str] = []
    current = yaml.safe_load(text) or {}
    data = current.get("data", {})

    if str(data.get("historical_matrix")) != str(MATRIX_PATH):
        text = _set_scalar(text, "historical_matrix", 2, str(MATRIX_PATH))
        applied.append("data.historical_matrix")

    if str(data.get("sha256")) != metadata["sha256"]:
        text = _set_scalar(text, "sha256", 2, metadata["sha256"])
        applied.append("data.sha256")

    columns = list(metadata["columns"])
    if list(data.get("assets", [])) != columns:
        text = _set_scalar(text, "assets", 2, f"[{', '.join(columns)}]")
        applied.append("data.assets")

    if gbm_path.is_file():
        digest = sha256_of_file(gbm_path)
        declared = current["configurations"]["A_gbm"]["gbm_params"].get("sha256")
        if str(declared) != digest:
            text = _set_scalar(text, "sha256", 6, digest)
            applied.append("configurations.A_gbm.gbm_params.sha256")

    return text, applied


def correct_test_set(test_set_path: Path | None = None) -> tuple[str, dict[str, Any], list[str]]:
    """Apply the recipe corrections in place and return the corrected text and document.

    Both ``--fit-gbm`` and ``--test-set`` go through here, because the asset-label alignment is a
    precondition of the column assertion that guards every generation.

    Args:
        test_set_path: Path of the recipe; resolved from the configuration directory when omitted.

    Returns:
        Tuple ``(text, document, applied)``.
    """
    test_set_path = test_set_path or config_path("test_set")
    text = test_set_path.read_text(encoding="utf-8")
    text, applied = apply_test_set_corrections(text, load_matrix_metadata(), config_path("gbm_params"))
    if applied:
        test_set_path.write_text(text, encoding="utf-8")
    return text, yaml.safe_load(text) or {}, applied


# ======================================================================================
# Frozen test set
# ======================================================================================
def build_test_set(
    test_set_path: Path | None = None,
    output_directory: Path = TEST_SET_DIRECTORY,
    *,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Build every active configuration of the frozen recipe and reconcile its digests.

    On the first build the digest and the date of each configuration are written into the recipe.
    On later builds the digest is compared, but only when ``compare_with_frozen`` reports the five
    hash-critical descriptors equal to the running environment: mode A needs a matrix factorisation
    whose last bit depends on the BLAS build, so a mismatch across environments is expected rather
    than a defect (thesis 4.3.1, D39). Outside the frozen environment the comparison is skipped and
    the differing descriptors are reported.

    Args:
        test_set_path: Path of the recipe; resolved from the configuration directory when omitted.
        output_directory: Where the tensors are written.
        argv: Command line recorded in the side-cars.

    Returns:
        Summary per configuration, with the digest, the verdict and the effective seed.
    """
    test_set_path = test_set_path or config_path("test_set")
    metadata = load_matrix_metadata()
    text, test_set, applied = correct_test_set(test_set_path)

    columns, matrix = load_matrix()
    assert_columns(columns, test_set)

    evaluation = test_set["evaluation"]
    differing = compare_with_frozen(evaluation["freeze_environment"])
    frozen_digests = dict(evaluation.get("tensor_sha256") or {})
    frozen_dates = dict(evaluation.get("first_generation_date") or {})
    today = datetime.now(UTC).date().isoformat()

    output_directory.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "corrections_applied": applied,
        "environment_differences": differing,
        "configurations": {},
    }

    for name, spec in test_set["configurations"].items():
        if str(spec.get("status")) != "active":
            continue
        tensor, effective_seed = generate_configuration(name, test_set, matrix=matrix, columns=columns)
        digest = tensor_digest(tensor)

        if name not in frozen_digests:
            verdict = "recorded"
        elif differing:
            verdict = "skipped"
        elif frozen_digests[name] == digest:
            verdict = "match"
        else:
            verdict = "MISMATCH"

        tensor_path = output_directory / f"{name}.npy"
        np.save(tensor_path, tensor)
        write_sidecar(
            tensor_path,
            {
                "configuration": name,
                "mode": spec["mode"],
                "n_scenarios": int(spec["n_scenarios"]),
                "horizon_months": int(spec["horizon_months"]),
                "mean_block_length_months": spec.get("mean_block_length_months"),
                "effective_seed": effective_seed,
                "seed": effective_seed,
                "tensor_sha256": digest,
                "hash_procedure": evaluation["hash_procedure"],
                "source_matrix": str(MATRIX_PATH),
                "source_matrix_sha256": metadata["sha256"],
                "frozen_digest_verdict": verdict,
                "environment_differences": differing,
            },
            argv=argv,
        )

        frozen_digests.setdefault(name, digest)
        frozen_dates.setdefault(name, today)
        summary["configurations"][name] = {
            "effective_seed": effective_seed,
            "tensor_sha256": digest,
            "verdict": verdict,
            "path": str(tensor_path),
        }

    text = _set_block_map(text, "tensor_sha256", 2, {k: f'"{v}"' for k, v in frozen_digests.items()})
    text = _set_block_map(text, "first_generation_date", 2, {k: f'"{v}"' for k, v in frozen_dates.items()})
    test_set_path.write_text(text, encoding="utf-8")
    return summary


# ======================================================================================
# Stylized facts
# ======================================================================================
def pooled_autocorrelation(series: np.ndarray, lag: int) -> float:
    """Lag-``lag`` autocorrelation pooled over the leading axis.

    Pairs are formed inside each scenario and then pooled, so no pair ever straddles two
    independent scenarios.

    Args:
        series: ``(n_scenarios, horizon)`` array, or a one-dimensional historical series.
        lag: Positive lag in months.

    Returns:
        The pooled Pearson correlation between the series and its lagged copy.
    """
    values = np.atleast_2d(series)
    first = values[:, :-lag].ravel()
    second = values[:, lag:].ravel()
    return float(np.corrcoef(first, second)[0, 1])


def _excess_kurtosis(values: np.ndarray) -> float:
    """Excess kurtosis of a flattened sample."""
    flat = values.ravel()
    centred = flat - flat.mean()
    return float((centred**4).mean() / (centred**2).mean() ** 2 - 3.0)


def stylized_facts(
    mode: str,
    *,
    n_scenarios: int = CHECK_SCENARIOS,
    horizon: int | None = None,
    seed: int | None = None,
    block_length: float | None = None,
    test_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the five checks of thesis 4.3.3 and return their statistics.

    The same function backs the command line and the tests, so the numbers reported in 4.4 are the
    numbers the test suite asserts on.

    1. Marginals: per asset, the scenario mean and monthly standard deviation lie within two
       standard errors of the historical values; the excess kurtosis of ``us_equity`` is positive in
       mode B and compatible with zero in mode A.
    2. Correlations: every contemporaneous correlation is within 0.05 of the historical one.
    3. Temporal dependence: in mode B the first-order autocorrelation of gold and of the two-year
       Treasury is within 0.05 of ``(1 - 1/L) * rho_1`` (``eq:bootstrap-autocorr``), with ``rho_1``
       read from ``returns_matrix.json``; in mode A it is within the same 0.05 band of zero.

    The two criteria the thesis states as "compatible with zero", this one in mode A and the excess
    kurtosis below, are gated on a declared absolute band rather than on a multiple of the standard
    error, for the reason given with :data:`ZERO_KURTOSIS_TOLERANCE`. The standard-error reading is
    reported alongside as ``z_score``.
    4. Volatility clustering: in mode B the autocorrelation of the absolute equity return at lags
       one to three is positive and no larger than the historical value.
    5. Reproducibility: the same seed reproduces the digest and a different seed does not.

    Args:
        mode: ``"A"`` or ``"B"``.
        n_scenarios: Scenarios drawn for the checks.
        horizon: Months per scenario; taken from the matching frozen configuration when omitted.
        seed: Seed; the lower bound of the validation band when omitted.
        block_length: Mean block length for mode B; taken from the matching configuration.
        test_set: Parsed recipe; read from disk when omitted.

    Returns:
        A JSON-serialisable document with one entry per check and an overall verdict.
    """
    test_set = test_set if test_set is not None else load_test_set()
    columns, matrix = load_matrix()
    assert_columns(columns, test_set)
    metadata = load_matrix_metadata()
    statistics = metadata["statistics"]

    configuration = _default_configuration(mode, block_length, test_set)
    horizon = int(horizon if horizon is not None else configuration["horizon_months"])
    seed = int(seed if seed is not None else test_set["seeds"]["validation"]["min"])
    if mode == MODE_BOOTSTRAP and block_length is None:
        block_length = float(configuration["mean_block_length_months"])

    tensor = generate(mode, n_scenarios, horizon, seed, block_length=block_length, matrix=matrix, columns=columns)
    n_observations = n_scenarios * horizon
    n_history = matrix.shape[0]

    document: dict[str, Any] = {
        "mode": mode,
        "block_length": block_length,
        "n_scenarios": n_scenarios,
        "horizon": horizon,
        "seed": seed,
        "n_synthetic_observations": n_observations,
        "n_historical_observations": n_history,
        "source_matrix_sha256": metadata["sha256"],
        "checks": {},
    }
    checks = document["checks"]

    # --- 1. Marginals ---------------------------------------------------------------
    marginals: dict[str, Any] = {"assets": {}, "passed": True}
    for index, name in enumerate(columns):
        flat = tensor[:, :, index].ravel()
        historical = statistics[name]
        synthetic_mean, synthetic_std = float(flat.mean()), float(flat.std(ddof=1))
        # Two standard errors of the difference. The historical term dominates: 420 months against
        # hundreds of thousands of synthetic ones.
        mean_error = 2.0 * float(
            np.hypot(historical["std_monthly"] / np.sqrt(n_history), synthetic_std / np.sqrt(n_observations))
        )
        std_error = 2.0 * float(
            np.hypot(
                historical["std_monthly"] / np.sqrt(2.0 * (n_history - 1)),
                synthetic_std / np.sqrt(2.0 * (n_observations - 1)),
            )
        )
        mean_ok = abs(synthetic_mean - historical["mean_monthly"]) <= mean_error
        std_ok = abs(synthetic_std - historical["std_monthly"]) <= std_error
        marginals["assets"][name] = {
            "synthetic_mean": synthetic_mean,
            "historical_mean": historical["mean_monthly"],
            "mean_tolerance": mean_error,
            "mean_passed": bool(mean_ok),
            "synthetic_std": synthetic_std,
            "historical_std": historical["std_monthly"],
            "std_tolerance": std_error,
            "std_passed": bool(std_ok),
        }
        marginals["passed"] = marginals["passed"] and mean_ok and std_ok

    equity = columns[0]
    excess_kurtosis = _excess_kurtosis(tensor[:, :, 0])
    kurtosis_standard_error = float(np.sqrt(24.0 / n_observations))
    kurtosis_ok = (
        excess_kurtosis > 0.0 if mode == MODE_BOOTSTRAP else abs(excess_kurtosis) <= ZERO_KURTOSIS_TOLERANCE
    )
    marginals["excess_kurtosis"] = {
        "asset": equity,
        "synthetic": excess_kurtosis,
        "historical": _excess_kurtosis(matrix[:, 0]),
        "criterion": "positive" if mode == MODE_BOOTSTRAP else "compatible with zero",
        "tolerance": None if mode == MODE_BOOTSTRAP else ZERO_KURTOSIS_TOLERANCE,
        "standard_error": kurtosis_standard_error,
        "z_score": excess_kurtosis / kurtosis_standard_error,
        "passed": bool(kurtosis_ok),
    }
    marginals["passed"] = bool(marginals["passed"] and kurtosis_ok)
    checks["marginals"] = marginals

    # --- 2. Correlations ------------------------------------------------------------
    historical_correlation = np.corrcoef(matrix, rowvar=False)
    synthetic_correlation = np.corrcoef(tensor.reshape(-1, len(columns)), rowvar=False)
    pairs: dict[str, Any] = {}
    correlations_passed = True
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            deviation = abs(float(synthetic_correlation[i, j] - historical_correlation[i, j]))
            passed = deviation <= CORRELATION_TOLERANCE
            correlations_passed = correlations_passed and passed
            pairs[f"{columns[i]}|{columns[j]}"] = {
                "synthetic": float(synthetic_correlation[i, j]),
                "historical": float(historical_correlation[i, j]),
                "absolute_deviation": deviation,
                "passed": bool(passed),
            }
    checks["correlations"] = {"tolerance": CORRELATION_TOLERANCE, "pairs": pairs, "passed": bool(correlations_passed)}

    # --- 3. Temporal dependence -----------------------------------------------------
    n_pairs = n_scenarios * (horizon - 1)
    autocorrelation_standard_error = 1.0 / float(np.sqrt(n_pairs))
    temporal: dict[str, Any] = {"assets": {}, "passed": True}
    for name in ("gold", "ust2y"):
        index = columns.index(name)
        measured = pooled_autocorrelation(tensor[:, :, index], 1)
        historical_rho = float(statistics[name]["autocorrelation_lag1"])
        expected = (1.0 - 1.0 / float(block_length)) * historical_rho if mode == MODE_BOOTSTRAP else 0.0
        passed = abs(measured - expected) <= AUTOCORRELATION_TOLERANCE
        temporal["passed"] = temporal["passed"] and passed
        temporal["assets"][name] = {
            "synthetic_lag1": measured,
            "historical_lag1": historical_rho,
            "expected": expected,
            "tolerance": AUTOCORRELATION_TOLERANCE,
            "standard_error": autocorrelation_standard_error,
            "z_score": (measured - expected) / autocorrelation_standard_error,
            "passed": bool(passed),
        }
    temporal["passed"] = bool(temporal["passed"])
    checks["temporal_dependence"] = temporal

    # --- 4. Volatility clustering ---------------------------------------------------
    absolute_synthetic = np.abs(tensor[:, :, 0])
    absolute_historical = np.abs(matrix[:, 0])
    clustering: dict[str, Any] = {"asset": equity, "lags": {}, "applicable": mode == MODE_BOOTSTRAP}
    clustering_passed = True
    for lag in (1, 2, 3):
        measured = pooled_autocorrelation(absolute_synthetic, lag)
        historical_value = pooled_autocorrelation(absolute_historical[None, :], lag)
        passed = 0.0 < measured <= historical_value
        clustering_passed = clustering_passed and passed
        clustering["lags"][str(lag)] = {
            "synthetic": measured,
            "historical": historical_value,
            "passed": bool(passed),
        }
    clustering["passed"] = bool(clustering_passed) if mode == MODE_BOOTSTRAP else True
    checks["volatility_clustering"] = clustering

    # --- 5. Reproducibility ---------------------------------------------------------
    digest = tensor_digest(tensor)
    same = tensor_digest(
        generate(mode, n_scenarios, horizon, seed, block_length=block_length, matrix=matrix, columns=columns)
    )
    other = tensor_digest(
        generate(mode, n_scenarios, horizon, seed + 1, block_length=block_length, matrix=matrix, columns=columns)
    )
    checks["reproducibility"] = {
        "digest": digest,
        "same_seed_digest": same,
        "other_seed": seed + 1,
        "other_seed_digest": other,
        "passed": bool(same == digest and other != digest),
    }

    document["all_passed"] = all(bool(check["passed"]) for check in checks.values())
    return document


def _default_configuration(mode: str, block_length: float | None, test_set: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen configuration whose horizon and block length the checks adopt."""
    for spec in test_set["configurations"].values():
        if str(spec["mode"]) != mode:
            continue
        if mode == MODE_BOOTSTRAP and block_length is not None:
            if float(spec["mean_block_length_months"] or 0.0) != float(block_length):
                continue
        return spec
    raise ValueError(f"no configuration in the frozen recipe has mode {mode!r} and block length {block_length!r}")


def write_stylized_facts(document: dict[str, Any], directory: Path = CHECKS_DIRECTORY) -> Path:
    """Write the statistics where subsection 4.4 reads them from."""
    block_length = document["block_length"]
    suffix = f"_{int(block_length)}" if block_length else ""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"stylized_facts_{document['mode']}{suffix}.json"
    payload = {
        **document,
        "command_line": sys.argv,
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment_fingerprint": collect_fingerprint(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ======================================================================================
# Command line
# ======================================================================================
def _print_fit_gbm() -> None:
    _, test_set, applied = correct_test_set()
    if applied:
        print("corrections applied to the recipe: " + ", ".join(applied))
    columns, matrix = load_matrix()
    assert_columns(columns, test_set)
    metadata = load_matrix_metadata()
    params = fit_gbm(matrix, columns)
    path, rewritten = write_gbm_params(params, metadata)
    print(f"{'wrote' if rewritten else 'already current in'} {path}  sha256 {sha256_of_file(path)}")
    print(f"source matrix {MATRIX_PATH}  sha256 {metadata['sha256']}\n")
    print(f"{'asset':12} {'mu (log)':>12} {'sd (log)':>12}")
    for name, mean, variance in zip(params.assets, params.mu, np.diag(params.sigma), strict=True):
        print(f"{name:12} {mean:12.8f} {np.sqrt(variance):12.8f}")


def _print_test_set(argv: list[str]) -> int:
    summary = build_test_set(argv=argv)
    if summary["corrections_applied"]:
        print("corrections applied to the recipe: " + ", ".join(summary["corrections_applied"]))
    if summary["environment_differences"]:
        print(
            "environment differs from the frozen one in "
            f"{summary['environment_differences']}: digest comparison skipped (D39)"
        )
    print(f"\n{'configuration':16} {'seed':>10} {'verdict':10} sha256")
    for name, entry in summary["configurations"].items():
        print(f"{name:16} {entry['effective_seed']:10d} {entry['verdict']:10} {entry['tensor_sha256']}")
    return 1 if any(entry["verdict"] == "MISMATCH" for entry in summary["configurations"].values()) else 0


def _print_stylized_facts(mode: str, block_length: float | None, n_scenarios: int, seed: int | None) -> int:
    document = stylized_facts(mode, n_scenarios=n_scenarios, seed=seed, block_length=block_length)
    path = write_stylized_facts(document)
    print(
        f"mode {document['mode']}  L {document['block_length']}  "
        f"{document['n_scenarios']} scenarios x {document['horizon']} months  seed {document['seed']}\n"
    )
    for name, check in document["checks"].items():
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {name}")
    print(f"\nwrote {path}")
    return 0 if document["all_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scenario generator: parametric and block bootstrap modes.")
    parser.add_argument("--fit-gbm", action="store_true", help="estimate and write config/gbm_params.yml")
    parser.add_argument("--test-set", action="store_true", help="build every active frozen configuration")
    parser.add_argument("--stylized-facts", action="store_true", help="run the five checks of thesis 4.3.3")
    parser.add_argument("--mode", choices=(MODE_PARAMETRIC, MODE_BOOTSTRAP), help="generator mode for the checks")
    parser.add_argument("--block-length", type=float, default=None, help="mean block length L, mode B only")
    parser.add_argument("--n-scenarios", type=int, default=CHECK_SCENARIOS, help="scenarios used by the checks")
    parser.add_argument("--seed", type=int, default=None, help="seed; defaults to the validation band lower bound")
    args = parser.parse_args(argv)

    try:
        if args.fit_gbm:
            _print_fit_gbm()
        if args.test_set:
            if args.seed is not None:
                raise ValueError("--test-set derives its seeds from the frozen recipe; --seed is not accepted")
            return _print_test_set(argv if argv is not None else sys.argv)
        if args.stylized_facts:
            if args.mode is None:
                raise ValueError("--stylized-facts requires --mode A or --mode B")
            return _print_stylized_facts(args.mode, args.block_length, args.n_scenarios, args.seed)
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    if not (args.fit_gbm or args.test_set or args.stylized_facts):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
