"""Tests of the scenario generator, thesis subsection 4.3.3.

The five stylized-fact checks are not reimplemented here: the tests call
:func:`portfolio_manager.generator.stylized_facts`, which is the same function the command line
runs, and assert on its output. The numbers reported in subsection 4.4 are therefore the numbers
the suite asserts on, and a check cannot pass on the report while failing in the tests.
"""

# Imports
# =====================================================================
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from portfolio_manager.environment_fingerprint import compare_with_frozen
from portfolio_manager.generator import (
    GbmParams,
    _set_block_map,
    _set_scalar,
    assert_columns,
    bootstrap_indices,
    config_path,
    fit_gbm,
    generate,
    generate_configuration,
    load_gbm_params,
    load_matrix,
    load_matrix_metadata,
    load_test_set,
    pooled_autocorrelation,
    sha256_of_file,
    stylized_facts,
    tensor_digest,
    write_gbm_params,
)

# Constants
# =====================================================================
SMALL_SCENARIOS = 64
SMALL_HORIZON = 24
SMALL_SEED = 1000  # validation band, never the training band 0-999 nor the test base


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def matrix() -> tuple[tuple[str, ...], np.ndarray]:
    """Historical matrix and its column order, read once for the whole module."""
    return load_matrix()


@pytest.fixture(scope="module")
def test_set() -> dict:
    """Parsed frozen recipe."""
    return load_test_set()


@pytest.fixture(scope="module")
def params() -> GbmParams:
    """Parameters of the parametric mode, as stored by ``--fit-gbm``."""
    return load_gbm_params()


# ======================================================================================
# Column order: the single point where history enters the system
# ======================================================================================
def test_matrix_header_equals_frozen_asset_order(matrix, test_set):
    """The CSV header is the normative order declared by the frozen recipe."""
    columns, _ = matrix
    assert columns == ("us_equity", "ust10y", "ust2y", "gold")
    assert_columns(columns, test_set)


def test_assert_columns_rejects_a_reordering(matrix, test_set):
    """A silent reordering corrupts every weight vector, so it must raise here."""
    columns, _ = matrix
    with pytest.raises(ValueError, match="column order mismatch"):
        assert_columns((columns[1], columns[0], columns[2], columns[3]), test_set)


def test_matrix_metadata_agrees_with_the_header(matrix):
    """``returns_matrix.json`` documents the same matrix the generator resamples."""
    columns, values = matrix
    metadata = load_matrix_metadata()
    assert tuple(metadata["columns"]) == columns
    assert metadata["n_rows"] == values.shape[0] == 420


# ======================================================================================
# Mode A: parametric
# ======================================================================================
def test_gbm_params_file_matches_the_matrix(params, matrix):
    """The stored estimate belongs to the matrix in the tree and to its column order."""
    columns, values = matrix
    assert tuple(params.assets) == columns
    stored = yaml.safe_load(config_path("gbm_params").read_text(encoding="utf-8"))
    assert stored["source_matrix_sha256"] == load_matrix_metadata()["sha256"]
    refitted = fit_gbm(values, columns)
    np.testing.assert_allclose(params.mu, refitted.mu, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(params.sigma, refitted.sigma, rtol=0.0, atol=1e-15)


def test_gbm_params_digest_is_recorded_in_the_recipe(test_set):
    """``A_gbm`` points at the parameter file actually on disk, by digest and by path."""
    declared = test_set["configurations"]["A_gbm"]["gbm_params"]
    assert declared["sha256"] == sha256_of_file(config_path("gbm_params"))
    assert declared["file"] == str(config_path("gbm_params"))


def test_configuration_files_use_one_extension():
    """``.yml`` throughout: a second spelling is how a configuration stops being read at all."""
    assert config_path("gbm_params").suffix == ".yml"
    for stem in ("dataset", "test_set", "baselines", "gbm_params"):
        assert config_path(stem).is_file(), f"config/{stem}.yml is missing"
    assert not list(Path("config").glob("*.yaml"))


def test_refitting_an_unchanged_matrix_rewrites_nothing(matrix, tmp_path):
    """``--fit-gbm`` is idempotent, so the digest recorded in the frozen recipe cannot drift.

    Only ``estimated_on`` would otherwise move with the calendar, and that alone would change the
    digest and force an edit of the frozen recipe on a day when nothing was re-estimated.
    """
    columns, values = matrix
    metadata = load_matrix_metadata()
    copy = tmp_path / "gbm_params.yml"
    copy.write_text(config_path("gbm_params").read_text(encoding="utf-8"), encoding="utf-8")

    _, rewritten = write_gbm_params(fit_gbm(values, columns), metadata, copy)
    assert not rewritten
    assert copy.read_text(encoding="utf-8") == config_path("gbm_params").read_text(encoding="utf-8")


def test_refitting_a_changed_matrix_does_rewrite(matrix, tmp_path):
    """The guard must not suppress a real change of the estimate."""
    columns, values = matrix
    copy = tmp_path / "gbm_params.yml"
    copy.write_text(config_path("gbm_params").read_text(encoding="utf-8"), encoding="utf-8")

    _, rewritten = write_gbm_params(fit_gbm(values * 1.01, columns), load_matrix_metadata(), copy)
    assert rewritten


def test_mode_a_shape_and_dtype(params):
    """Shape, dtype and the lower bound a simple return cannot cross."""
    tensor = generate("A", SMALL_SCENARIOS, SMALL_HORIZON, SMALL_SEED, params=params)
    assert tensor.shape == (SMALL_SCENARIOS, SMALL_HORIZON, 4)
    assert tensor.dtype == np.float64
    assert np.isfinite(tensor).all()
    assert (tensor > -1.0).all(), "a simple return of -1 or below would allow negative wealth"


def test_mode_a_months_are_independent(params):
    """Independence across months is what makes the state exactly Markovian in this mode."""
    tensor = generate("A", 4000, 120, SMALL_SEED, params=params)
    for index in range(4):
        autocorrelation = pooled_autocorrelation(tensor[:, :, index], 1)
        assert abs(autocorrelation) < 4.0 / np.sqrt(4000 * 119)


def test_mode_a_rejects_a_block_length(params):
    """Mode A has no blocks; passing one means the caller confused the modes."""
    with pytest.raises(ValueError, match="no blocks"):
        generate("A", 2, 2, SMALL_SEED, block_length=12.0, params=params)


def test_gbm_factor_reproduces_the_covariance(params):
    """``F @ F.T`` is the covariance, whichever factorisation was used."""
    factor = params.factor()
    np.testing.assert_allclose(factor @ factor.T, params.sigma, rtol=0.0, atol=1e-18)
    assert np.allclose(np.triu(factor, 1), 0.0), "the non-degenerate case must take the Cholesky path"


def test_gbm_factor_handles_a_degenerate_asset():
    """A zero-variance asset makes the covariance singular, which Cholesky cannot factor.

    This is the Browne validation configuration of D44: equity plus a riskless asset with drift
    ``r``. The factorisation must degrade to an eigendecomposition instead of raising, and the
    degenerate column must come out exactly constant rather than merely small.
    """
    degenerate = GbmParams(
        assets=("us_equity", "riskless"),
        mu=np.array([0.008, 0.003]),
        sigma=np.array([[0.002, 0.0], [0.0, 0.0]]),
    )
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(degenerate.sigma)

    factor = degenerate.factor()
    np.testing.assert_allclose(factor @ factor.T, degenerate.sigma, rtol=0.0, atol=1e-18)

    tensor = generate("A", 32, 12, SMALL_SEED, params=degenerate)
    assert tensor[:, :, 1].std() == 0.0
    np.testing.assert_allclose(tensor[:, :, 1], np.expm1(0.003), rtol=0.0, atol=1e-15)
    assert tensor[:, :, 0].std() > 0.0


def test_gbm_params_rejects_an_asymmetric_covariance():
    """An asymmetric covariance is a construction error, not a model."""
    with pytest.raises(ValueError, match="not symmetric"):
        GbmParams(assets=("a", "b"), mu=np.zeros(2), sigma=np.array([[1.0, 0.5], [0.4, 1.0]]))


def test_gbm_params_rejects_a_name_count_mismatch():
    """The asset names index the columns, so their count is part of the contract."""
    with pytest.raises(ValueError, match="asset names"):
        GbmParams(assets=("a",), mu=np.zeros(2), sigma=np.eye(2))


# ======================================================================================
# Mode B: stationary block bootstrap
# ======================================================================================
def test_mode_b_shape_and_dtype(matrix):
    """Shape, dtype and the lower bound a simple return cannot cross."""
    columns, values = matrix
    tensor = generate("B", SMALL_SCENARIOS, SMALL_HORIZON, SMALL_SEED, block_length=12.0, matrix=values)
    assert tensor.shape == (SMALL_SCENARIOS, SMALL_HORIZON, 4)
    assert tensor.dtype == np.float64
    assert (tensor > -1.0).all()


def test_mode_b_copies_whole_historical_rows(matrix):
    """Every synthetic month is a real month, for the four assets at once.

    This is what preserves the marginals exactly and the contemporaneous correlations exactly; an
    independent draw per asset would pass a marginal check and destroy the diversification problem.
    """
    _, values = matrix
    tensor = generate("B", 128, 60, SMALL_SEED, block_length=12.0, matrix=values)
    historical_rows = {row.tobytes() for row in values}
    synthetic_rows = {row.tobytes() for row in tensor.reshape(-1, values.shape[1])}
    assert synthetic_rows <= historical_rows


def test_mode_b_requires_a_block_length(matrix):
    """``L`` has no default: it is the parameter of the sweep."""
    _, values = matrix
    with pytest.raises(ValueError, match="requires block_length"):
        generate("B", 2, 2, SMALL_SEED, matrix=values)


def test_mode_b_rejects_a_block_length_below_one(matrix):
    """``L`` is a mean number of months, so values below one are a unit error."""
    _, values = matrix
    with pytest.raises(ValueError, match="cannot be below 1"):
        generate("B", 2, 2, SMALL_SEED, block_length=0.5, matrix=values)


def test_bootstrap_blocks_are_contiguous_and_circular():
    """Inside a block the index advances by one month, wrapping at the end of the sample.

    The share of month-to-month transitions that advance by one is the complement of the block
    break probability, ``1 - 1/L``, which is the mechanism behind ``eq:bootstrap-autocorr``.
    """
    n_rows, block_length = 420, 12.0
    rng = np.random.default_rng(SMALL_SEED)
    indices = bootstrap_indices(rng, 2000, 120, n_rows, block_length)

    assert indices.min() >= 0 and indices.max() < n_rows
    advances = indices[:, 1:] == (indices[:, :-1] + 1) % n_rows
    # Coincidental advances across a block break happen with probability 1/n_rows.
    expected = (1.0 - 1.0 / block_length) + (1.0 / block_length) / n_rows
    assert abs(advances.mean() - expected) < 0.01

    wrapped = (indices[:, :-1] == n_rows - 1) & (indices[:, 1:] == 0)
    assert wrapped.any(), "the sample must be treated as circular"


def test_bootstrap_block_lengths_are_geometric():
    """Run lengths average ``L``, which is the Politis and Romano parameterisation."""
    n_rows, block_length = 420, 12.0
    rng = np.random.default_rng(SMALL_SEED)
    indices = bootstrap_indices(rng, 4000, 120, n_rows, block_length)
    breaks = indices[:, 1:] != (indices[:, :-1] + 1) % n_rows
    observed_mean_run = indices[:, 1:].size / max(breaks.sum(), 1)
    assert abs(observed_mean_run - block_length) < 1.0


# ======================================================================================
# Reproducibility
# ======================================================================================
@pytest.mark.parametrize(("mode", "block_length"), [("A", None), ("B", 12.0)])
def test_same_seed_reproduces_and_another_seed_does_not(mode, block_length, matrix, params):
    """Check 5 of thesis 4.3.3, on both modes."""
    _, values = matrix
    kwargs = {"block_length": block_length, "matrix": values, "params": params if mode == "A" else None}
    first = tensor_digest(generate(mode, SMALL_SCENARIOS, SMALL_HORIZON, SMALL_SEED, **kwargs))
    again = tensor_digest(generate(mode, SMALL_SCENARIOS, SMALL_HORIZON, SMALL_SEED, **kwargs))
    other = tensor_digest(generate(mode, SMALL_SCENARIOS, SMALL_HORIZON, SMALL_SEED + 1, **kwargs))
    assert first == again
    assert first != other


def test_tensor_digest_follows_the_declared_hash_procedure(test_set, matrix):
    """The digest is exactly what ``evaluation.hash_procedure`` says it is."""
    _, values = matrix
    tensor = generate("B", 8, 8, SMALL_SEED, block_length=12.0, matrix=values)
    expected = hashlib.sha256(tensor.astype("float64", order="C").tobytes()).hexdigest()
    assert tensor_digest(tensor) == expected
    assert test_set["evaluation"]["hash_procedure"] == "sha256 of arr.astype('float64', order='C').tobytes()"


def test_effective_seed_follows_the_normative_derivation(test_set, matrix):
    """``effective_seed = seeds.test_base + offset``: one base value plus declared offsets.

    Keeping the whole set reproducible from a single number is what makes a collision with the
    training band 0-999 or the validation band 1000-1999 impossible by construction.
    """
    columns, values = matrix
    base = int(test_set["seeds"]["test_base"])
    assert base > test_set["seeds"]["validation"]["max"]

    for name, spec in test_set["configurations"].items():
        if str(spec.get("status")) != "active":
            continue
        # A reduced copy of the recipe: the seed derivation is the subject, not the tensor size.
        reduced = {**test_set, "configurations": {name: {**spec, "n_scenarios": 8, "horizon_months": 8}}}
        tensor, seed = generate_configuration(name, reduced, matrix=values, columns=columns)
        assert seed == base + int(spec["offset"])
        assert tensor.shape == (8, 8, 4)


def test_frozen_test_set_digests_reproduce(test_set, matrix):
    """The regression test of thesis 4.3.1: inside the frozen environment the digests must match.

    Outside it the comparison is skipped with the list of differing descriptors, because mode A
    needs a matrix factorisation whose last bit depends on the BLAS build (D39).
    """
    recorded = test_set["evaluation"].get("tensor_sha256") or {}
    if not recorded:
        pytest.skip("no digest recorded yet: run --test-set once")
    differing = compare_with_frozen(test_set["evaluation"]["freeze_environment"])
    if differing:
        pytest.skip(f"environment differs from the frozen one in {differing}")

    columns, values = matrix
    for name in recorded:
        tensor, _ = generate_configuration(name, test_set, matrix=values, columns=columns)
        assert tensor_digest(tensor) == recorded[name], f"{name} no longer reproduces its frozen digest"


# ======================================================================================
# Stylized facts, thesis 4.3.3
# ======================================================================================
@pytest.fixture(scope="module")
def facts_bootstrap() -> dict:
    """The five checks on mode B with the base block length of the sweep."""
    return stylized_facts("B", block_length=12.0)


@pytest.fixture(scope="module")
def facts_parametric() -> dict:
    """The five checks on mode A."""
    return stylized_facts("A")


@pytest.mark.parametrize(
    "check", ["marginals", "correlations", "temporal_dependence", "volatility_clustering", "reproducibility"]
)
def test_bootstrap_passes_every_stylized_fact(facts_bootstrap, check):
    """Each of the five checks, reported separately so a failure names itself."""
    assert facts_bootstrap["checks"][check]["passed"], facts_bootstrap["checks"][check]


@pytest.mark.parametrize(
    "check", ["marginals", "correlations", "temporal_dependence", "volatility_clustering", "reproducibility"]
)
def test_parametric_passes_every_stylized_fact(facts_parametric, check):
    """The same five checks under the parametric mode, with its own criteria."""
    assert facts_parametric["checks"][check]["passed"], facts_parametric["checks"][check]


def test_bootstrap_keeps_the_heavy_tail_of_equity(facts_bootstrap):
    """Each synthetic month is a real month, so the excess kurtosis survives resampling."""
    kurtosis = facts_bootstrap["checks"]["marginals"]["excess_kurtosis"]
    assert kurtosis["synthetic"] > 0.0
    assert abs(kurtosis["synthetic"] - kurtosis["historical"]) < 0.5


def test_parametric_has_no_heavy_tail(facts_parametric):
    """The lognormal is not meant to reproduce heavy tails; its role is tractability."""
    kurtosis = facts_parametric["checks"]["marginals"]["excess_kurtosis"]
    assert abs(kurtosis["synthetic"]) <= kurtosis["tolerance"]
    assert kurtosis["synthetic"] < kurtosis["historical"] / 10.0


def test_the_zero_criteria_are_not_significance_tests(facts_parametric):
    """The bands around zero are declared sizes, comfortably wider than the sampling noise.

    A two-standard-error gate on a statistic that is genuinely zero rejects five per cent of the
    time by construction; across the three such statistics of mode A the suite would fail on
    nothing but the seed. Each band must therefore stay several standard errors wide, and this test
    is what stops a later edit from quietly turning the criterion back into a significance test.
    """
    marginals = facts_parametric["checks"]["marginals"]["excess_kurtosis"]
    entries = [marginals, *facts_parametric["checks"]["temporal_dependence"]["assets"].values()]
    for entry in entries:
        assert entry["tolerance"] >= 4.0 * entry["standard_error"]
        assert abs(entry["z_score"]) < entry["tolerance"] / entry["standard_error"]


def test_bootstrap_attenuates_autocorrelation_as_predicted(facts_bootstrap):
    """``eq:bootstrap-autocorr``: the historical dependence survives multiplied by ``1 - 1/L``.

    Gold and the two-year Treasury are the two columns with dependence worth measuring, and both
    carry it by construction rather than as a market fact: monthly averaging in one case, the
    accrual term of the duration reconstruction in the other. Comparing against the attenuated
    historical value rather than against zero is what makes the check meaningful.
    """
    for entry in facts_bootstrap["checks"]["temporal_dependence"]["assets"].values():
        assert abs(entry["synthetic_lag1"] - entry["expected"]) <= entry["tolerance"]
        assert entry["expected"] < entry["historical_lag1"]


def test_bootstrap_keeps_volatility_clustering_below_the_historical_level(facts_bootstrap):
    """Clustering survives attenuated at every lag from one to three."""
    for entry in facts_bootstrap["checks"]["volatility_clustering"]["lags"].values():
        assert 0.0 < entry["synthetic"] <= entry["historical"]


def test_longer_blocks_retain_more_dependence():
    """The sweep must move the quantity it exists to move: ``L = 24`` attenuates less than 12.

    Measured on the two-year Treasury, the column with by far the strongest historical dependence,
    so the gap the sweep opens is several standard errors wide rather than a coin flip.
    """
    short = stylized_facts("B", block_length=12.0)
    long = stylized_facts("B", block_length=24.0)
    short_entry = short["checks"]["temporal_dependence"]["assets"]["ust2y"]
    long_entry = long["checks"]["temporal_dependence"]["assets"]["ust2y"]
    assert long_entry["expected"] > short_entry["expected"]
    assert long_entry["synthetic_lag1"] > short_entry["synthetic_lag1"]


def test_stylized_facts_rejects_an_unknown_configuration():
    """The horizon and the block length come from the frozen recipe, not from a literal."""
    with pytest.raises(ValueError, match="no configuration"):
        stylized_facts("B", block_length=7.0)


def test_unknown_mode_raises(matrix):
    """Modes C and D are declared in the recipe but not implemented here."""
    _, values = matrix
    with pytest.raises(ValueError, match="unknown mode"):
        generate("C", 2, 2, SMALL_SEED, matrix=values)


# ======================================================================================
# Configuration editing
# ======================================================================================
def test_scalar_patch_preserves_the_surrounding_comments():
    """The configuration files carry the audit trail in their comments; patches must not touch it."""
    text = "# provenance comment\ndata:\n  sha256: TBD\n  window: keep me\n"
    patched = _set_scalar(text, "sha256", 2, "abc123")
    assert patched == "# provenance comment\ndata:\n  sha256: abc123\n  window: keep me\n"


def test_scalar_patch_refuses_an_ambiguous_key():
    """Two keys at the same indentation would make the target ambiguous, so it must raise."""
    text = "a:\n  sha256: one\nb:\n  sha256: two\n"
    with pytest.raises(ValueError, match="exactly one"):
        _set_scalar(text, "sha256", 2, "x")


def test_block_map_patch_replaces_an_empty_mapping():
    """``tensor_sha256: {}`` becomes an indented block without disturbing the next key."""
    text = "evaluation:\n  tensor_sha256: {}\n  first_generation_date: {}\n"
    patched = _set_block_map(text, "tensor_sha256", 2, {"A_gbm": '"d1"'})
    assert patched == 'evaluation:\n  tensor_sha256:\n    A_gbm: "d1"\n  first_generation_date: {}\n'


def test_block_map_patch_rewrites_an_existing_block():
    """A later build updates the block in place rather than appending a second one."""
    text = 'evaluation:\n  tensor_sha256:\n    A_gbm: "old"\n    B_L12: "old"\n  dtype: float64\n'
    patched = _set_block_map(text, "tensor_sha256", 2, {"A_gbm": '"new"'})
    assert patched == 'evaluation:\n  tensor_sha256:\n    A_gbm: "new"\n  dtype: float64\n'
