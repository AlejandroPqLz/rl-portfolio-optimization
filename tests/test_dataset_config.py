"""The dataset configuration is read, not merely present.

``load_config`` falls back to hard-coded defaults when the file is missing, which is a reasonable
thing for it to do and a dangerous thing to leave unchecked: the constant naming the file once
pointed at an extension the repository does not use, so the fallback ran on every call and the
declared window, filenames and bond maturities were decoration. Nothing failed, because the
defaults happened to agree with the file.

These tests close that gap from both sides. The values loaded must equal the values written, so a
silent fallback is a failure rather than a coincidence; and the matrix rebuilt from them must carry
the digest already recorded, so the correction cannot have moved the data underneath every frozen
digest in the project.
"""

# Imports
# =====================================================================
import json

import pytest
import yaml

from portfolio_manager.etl.dataset.consolidate import (
    CONFIG_PATH,
    MATRIX_METADATA_PATH,
    RAW_DIRECTORY,
    DatasetConfig,
    build_matrix,
    load_config,
    write_matrix,
)


# Fixtures
# =====================================================================
@pytest.fixture(scope="module")
def declared() -> dict:
    """The configuration file as written."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def loaded() -> DatasetConfig:
    """The configuration as the consolidation stage sees it."""
    return load_config()


# ======================================================================================
# The file is found
# ======================================================================================
def test_the_configured_path_exists():
    """The constant must name a file that is there, or every call silently takes the fallback."""
    assert CONFIG_PATH.is_file(), f"{CONFIG_PATH} does not exist, so load_config falls back to defaults"
    assert CONFIG_PATH.suffix == ".yml"


def test_the_loaded_window_is_the_declared_one(declared, loaded):
    """The window is what makes the dataset immutable, so it has to come from the file."""
    assert loaded.window_start == declared["window"]["start"]
    assert loaded.window_end == declared["window"]["end"]
    assert loaded.expected_rows == declared["window"]["expected_rows"]


def test_the_loaded_filenames_are_the_declared_ones(declared, loaded):
    """A provider file renamed in the configuration must change what is read."""
    assert loaded.equity_filename == declared["equity_filename"]
    assert loaded.gold_filename == declared["gold_filename"]


def test_the_loaded_bonds_are_the_declared_ones(declared, loaded):
    """Names, files and maturities: the maturity fixes the sensitivities of the reconstruction."""
    assert [spec.name for spec in loaded.bonds] == list(declared["bonds"])
    for spec in loaded.bonds:
        assert spec.filename == declared["bonds"][spec.name]["filename"]
        assert spec.maturity_years == declared["bonds"][spec.name]["maturity_years"]


def test_a_changed_file_changes_the_configuration(tmp_path):
    """The assertion that distinguishes reading the file from agreeing with it by accident.

    Every value above also happens to equal the hard-coded default, so on their own they would pass
    against a loader that ignored the file entirely. Altering the file must alter the result.
    """
    altered = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    altered["window"]["start"] = "1995-01"
    altered["window"]["expected_rows"] = 360
    altered["gold_filename"] = "somewhere_else.xlsx"
    altered["bonds"]["ust10y"]["maturity_years"] = 7
    path = tmp_path / "dataset.yml"
    path.write_text(yaml.safe_dump(altered), encoding="utf-8")

    config = load_config(path)
    assert config.window_start == "1995-01"
    assert config.expected_rows == 360
    assert config.gold_filename == "somewhere_else.xlsx"
    assert next(spec for spec in config.bonds if spec.name == "ust10y").maturity_years == 7


def test_a_missing_file_still_falls_back(tmp_path):
    """The fallback is deliberate and stays; what was wrong was taking it without noticing."""
    assert load_config(tmp_path / "absent.yml") == DatasetConfig()


# ======================================================================================
# The data behind every frozen digest is unchanged
# ======================================================================================
def test_the_column_order_is_the_normative_one(loaded):
    """Equity, the two Treasury legs in declaration order, then gold."""
    assert loaded.columns == ("us_equity", "ust10y", "ust2y", "gold")


@pytest.mark.slow
def test_the_rebuilt_matrix_carries_the_recorded_digest(loaded, tmp_path):
    """Reading the configuration instead of the defaults must not have moved the data.

    Every frozen scenario digest, the comparator parameters and the reference run all descend from
    this one file. If correcting the path had changed any figure of it, the correction would have
    silently invalidated all of them, and nothing else in the suite compares the matrix against
    what was recorded when it was built.
    """
    rebuilt = build_matrix(RAW_DIRECTORY, loaded)
    metadata = write_matrix(rebuilt, loaded, tmp_path / "matrix.csv", tmp_path / "matrix.json")
    recorded = json.loads(MATRIX_METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["sha256"] == recorded["sha256"]
    assert metadata["n_rows"] == recorded["n_rows"] == 420
    assert metadata["columns"] == recorded["columns"]
    for column, statistics in recorded["statistics"].items():
        for name, value in statistics.items():
            assert metadata["statistics"][column][name] == pytest.approx(value, rel=0.0, abs=1e-15)
