"""Action A28: the fixed-income split phi, solved from the project's own reconstruction.

The constant-mix and glide-path comparators of thesis 4.3.5 split their bond sleeve ``phi`` to the
ten-year leg and ``1 - phi`` to the two-year one, chosen so that the sleeve carries the six-year
duration of the aggregate index the reference fund tracks. The specification fixed ``phi`` from
reference durations of 8.5 and 1.9 years; A28 replaces them by the sample-mean modified durations
that this project's own reconstruction actually produced over the window, under D36.

The value must be settled before the first evaluation and may not move afterwards, so these tests
check the stored configuration and not only the arithmetic.
"""

# Imports
# =====================================================================
import re
from pathlib import Path

import pytest
import yaml

from portfolio_manager.etl.dataset.consolidate import (
    RAW_DIRECTORY,
    DatasetConfig,
    baselines_path,
    ensure_comparator_blocks,
    load_config,
    sample_mean_modified_durations,
    solve_phi,
    write_fixed_income_split,
)

# Constants
# =====================================================================
NOMINAL_PHI = 0.621
"""The value of the specification, solved from reference durations of 8.5 and 1.9 years."""


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def baselines() -> dict:
    """Parsed comparator configuration."""
    return yaml.safe_load(baselines_path().read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def split(baselines) -> dict:
    """The fixed-income split block."""
    return baselines["asset_mapping"]["fixed_income_split"]


@pytest.fixture(scope="module")
def dataset_config() -> DatasetConfig:
    """Window and bond specifications."""
    return load_config()


# ======================================================================================
# The arithmetic
# ======================================================================================
def test_solve_phi_reproduces_the_target_duration():
    """``phi`` is defined by the equation it solves, so the round trip must be exact."""
    phi = solve_phi(8.5, 1.9, 6.0)
    assert phi * 8.5 + (1.0 - phi) * 1.9 == pytest.approx(6.0)
    assert phi == pytest.approx(NOMINAL_PHI, abs=5e-4), "the nominal value of the specification"


def test_solve_phi_rejects_two_identical_legs():
    """With both legs at the same duration the split is undetermined, not arbitrary."""
    with pytest.raises(ValueError, match="undetermined"):
        solve_phi(4.0, 4.0, 6.0)


def test_solve_phi_does_not_hide_an_unreachable_target():
    """A target outside the two durations must surface as a weight outside the simplex."""
    assert solve_phi(8.5, 1.9, 10.0) > 1.0


# ======================================================================================
# The sample-mean durations
# ======================================================================================
@pytest.fixture(scope="module")
def durations(dataset_config) -> dict[str, float]:
    """Mean modified duration of each leg over the window, from the raw yield files."""
    for spec in dataset_config.bonds:
        if not (Path(RAW_DIRECTORY) / spec.filename).is_file():
            pytest.skip(f"{spec.filename} absent: run the ingestion module first")
    return sample_mean_modified_durations(RAW_DIRECTORY, dataset_config)


def test_sample_durations_are_ordered_and_plausible(durations, dataset_config):
    """The ten-year leg must be the longer one, and each must sit near its maturity.

    An inverted order or a duration far from the maturity would mean the par-bond sensitivities were
    evaluated on the wrong series or in the wrong units, which is exactly the failure the scale
    check of D37 exists to catch.
    """
    assert durations["ust10y"] > durations["ust2y"]
    assert 6.0 < durations["ust10y"] < 10.0
    assert 1.5 < durations["ust2y"] < 2.0


def test_sample_durations_bracket_the_target(durations, split):
    """The six-year target must lie between the two legs, or no valid split exists."""
    target = float(split["target_duration_years"])
    assert durations["ust2y"] < target < durations["ust10y"]


def test_sample_based_phi_differs_from_the_nominal_one(durations, split):
    """A28 exists because the reconstruction's own durations are not the reference ones."""
    phi = solve_phi(durations["ust10y"], durations["ust2y"], float(split["target_duration_years"]))
    assert 0.0 < phi < 1.0
    assert abs(phi - NOMINAL_PHI) > 1e-3


# ======================================================================================
# The stored configuration
# ======================================================================================
def test_configuration_holds_the_sample_based_split(split):
    """The file on disk carries the solved value and the two durations it came from."""
    assert "sample_durations_used" in split, "A28 has not been run: see --fixed-income-report"
    sample = split["sample_durations_used"]
    phi = float(split["ust10y_share"])
    solved = solve_phi(float(sample["D10"]), float(sample["D2"]), float(split["target_duration_years"]))

    assert phi == pytest.approx(solved, abs=1e-6)
    assert phi != NOMINAL_PHI, "the nominal value has not been replaced"
    assert float(split["ust2y_share"]) == pytest.approx(1.0 - phi, abs=1e-6)
    assert sample["n_months"] == 420
    assert sample["window"] == "1990-01 / 2024-12"


def test_stored_durations_match_the_reconstruction(split, durations):
    """The recorded durations are the ones the pipeline produces today, not a transcription."""
    sample = split["sample_durations_used"]
    assert float(sample["D10"]) == pytest.approx(durations["ust10y"], abs=1e-6)
    assert float(sample["D2"]) == pytest.approx(durations["ust2y"], abs=1e-6)


def test_external_provenance_of_the_target_is_preserved(split):
    """A28 replaces the durations, never the externally sourced six-year target."""
    assert float(split["target_duration_years"]) == 6.0
    assert split["target_duration_source"]["quoted"] == "average duration of the Index was 6 years"
    assert split["reference_durations_used"]["D10"] == 8.5
    assert split["reference_durations_used"]["D2"] == 1.9


def test_writing_the_split_is_idempotent(durations, dataset_config, tmp_path):
    """Re-running the report must not rewrite the file, so a frozen value cannot drift."""
    copy = tmp_path / "baselines.yml"
    copy.write_text(baselines_path().read_text(encoding="utf-8"), encoding="utf-8")

    phi, rewritten = write_fixed_income_split(durations, dataset_config, copy)
    assert not rewritten, "the configuration in the tree is already the sample-based one"
    assert copy.read_text(encoding="utf-8") == baselines_path().read_text(encoding="utf-8")
    assert phi == pytest.approx(float(yaml.safe_load(copy.read_text())["asset_mapping"]
                                      ["fixed_income_split"]["ust10y_share"]), abs=1e-9)


def test_writing_the_split_preserves_the_provenance_comments(durations, dataset_config, tmp_path):
    """The file is patched line by line: the comments are the audit trail and must survive."""
    copy = tmp_path / "baselines.yml"
    stale = re.sub(
        r"^    ust10y_share:.*$", "    ust10y_share: 0.5", baselines_path().read_text(encoding="utf-8"), flags=re.M
    )
    copy.write_text(stale, encoding="utf-8")

    _, rewritten = write_fixed_income_split(durations, dataset_config, copy)
    assert rewritten
    for marker in ("RULE: no value here may be changed", "Vanguard Target Retirement 2035", "phi_constant_note"):
        assert marker in copy.read_text(encoding="utf-8")


# ======================================================================================
# The comparator blocks of D68
# ======================================================================================
def test_constant_mix_and_equal_weight_blocks_exist(baselines):
    """Both were written before ``baselines.py``, so no weight is an implicit choice in code."""
    assert baselines["constant_mix"]["equity_weight"] == 0.60
    assert baselines["constant_mix"]["fixed_income_weight"] == 0.40
    assert baselines["constant_mix"]["equity_weight"] + baselines["constant_mix"]["fixed_income_weight"] == 1.0
    assert baselines["equal_weight"]["weight_per_asset"] == 0.25


def test_every_comparator_block_declares_that_it_was_not_calibrated(baselines):
    """``calibrated: false`` is the claim the file exists to make checkable in one place."""
    for block in ("constant_mix", "equal_weight", "merton_crra", "transaction_cost"):
        assert baselines[block]["calibrated"] is False


def test_constant_mix_shares_the_glide_path_projection(baselines):
    """D68: with the mapping shared, the two heuristics differ only in their temporal shape."""
    assert baselines["constant_mix"]["uses_asset_mapping"] is True
    assert baselines["equal_weight"]["uses_asset_mapping"] is False
    assert baselines["asset_mapping"]["gold"] == 0.0


def test_adding_the_comparator_blocks_is_idempotent(tmp_path):
    """A second run must not append a duplicate block."""
    copy = tmp_path / "baselines.yml"
    copy.write_text(baselines_path().read_text(encoding="utf-8"), encoding="utf-8")
    assert ensure_comparator_blocks(copy) == []
