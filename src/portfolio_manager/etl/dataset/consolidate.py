"""Consolidation: the four raw provider files to the D x 4 matrix of monthly returns.

This module orchestrates. The provider-specific parsing lives in one module per source:

:mod:`portfolio_manager.equity`
    Kenneth French research factors. Already returns; the total return is recovered as ``Mkt-RF + RF``.

:mod:`portfolio_manager.fixed_income`
    Treasury constant maturity yields. These are yields, not returns, and the duration reconstruction is what turns
    one into the other.

:mod:`portfolio_manager.gold`
    World Bank monthly commodity prices, in monthly-average convention.

What stays here is what belongs to the matrix as a whole: the configuration, the alignment of four indices into one,
the window, the verifications, the recorded statistics and the digest. ``--fixed-income-report`` additionally closes
A28: it reports the sample-mean modified durations of the reconstruction over the window, solves
``phi * D_long + (1 - phi) * D_short = target_duration_years`` and writes ``phi`` into ``config/baselines.yml``,
which is the split that the constant-mix and glide-path comparators of thesis 4.3.5 read (D67, D68). After the first
run of ``evaluate.py`` the value may no longer change.

Command line, from the repository root:
    uv run python -m portfolio_manager.etl.dataset.consolidate --scale-check
    uv run python -m portfolio_manager.etl.dataset.consolidate --fixed-income-report
    uv run python -m portfolio_manager.etl.dataset.consolidate --build
"""

# Imports
# =====================================================================
import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from portfolio_manager.etl.dataset.equity import load_equity_returns
from portfolio_manager.etl.dataset.fixed_income import (
    UST2Y,
    UST10Y,
    BondSpec,
    load_bond_returns,
    month_end_observations,
    read_fred_daily,
    return_components,
    scale_check,
)
from portfolio_manager.etl.dataset.gold import load_gold_returns

# Constants
# =====================================================================
RAW_DIRECTORY = Path("data/raw")
CONFIG_DIR = Path("config")
CONFIG_PATH = Path("config/dataset.yml")
MATRIX_PATH = Path("data/returns_matrix.csv")
MATRIX_METADATA_PATH = Path("data/returns_matrix.json")
PHI_DECIMALS = 6
"""Decimals written for the fixed-income split. The stored value is compared at the same precision,
so re-running the report on an already-updated file rewrites nothing."""

PHI_COMMENT_COLUMN = 34
"""Column at which the inline comments of the split are aligned, matching the rest of the file."""


# =====================================================================================
# Configuration
# =====================================================================================
@dataclass(frozen=True)
class DatasetConfig:
    """Everything the consolidation needs that is not code.

    The window is declared here and never hard-coded: cutting at a fixed past date is what
    makes the dataset immutable, and an immutable dataset is what makes the frozen test set
    and the September code freeze mean anything.
    """

    window_start: str = "1990-01"
    window_end: str = "2024-12"
    expected_rows: int = 420
    equity_filename: str = "us_equity_ff_factors.zip"
    gold_filename: str = "gold_worldbank_pinksheet.xlsx"
    equity_column: str = "us_equity"
    gold_column: str = "gold"
    bonds: tuple[BondSpec, ...] = field(default_factory=lambda: (UST10Y, UST2Y))

    @property
    def columns(self) -> tuple[str, ...]:
        """Column order of the output matrix."""
        return (self.equity_column, *(spec.name for spec in self.bonds), self.gold_column)


def load_config(path: Path = CONFIG_PATH) -> DatasetConfig:
    """Read the dataset configuration, falling back to the declared defaults.

    Args:
        path: Path to the YAML file.

    Returns:
        Configuration object. If the file is absent the defaults are used.
    """
    if not path.is_file():
        return DatasetConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bonds = raw.get("bonds", {})
    specs = tuple(BondSpec.from_mapping(name, values) for name, values in bonds.items()) or (UST10Y, UST2Y)
    window = raw.get("window", {})
    return DatasetConfig(
        window_start=str(window.get("start", "1990-01")),
        window_end=str(window.get("end", "2024-12")),
        expected_rows=int(window.get("expected_rows", 420)),
        equity_filename=str(raw.get("equity_filename", "us_equity_ff_factors.zip")),
        gold_filename=str(raw.get("gold_filename", "gold_worldbank_pinksheet.xlsx")),
        equity_column=str(raw.get("equity_column", "us_equity")),
        gold_column=str(raw.get("gold_column", "gold")),
        bonds=specs,
    )


# ======================================================================================
# Assembly
# ======================================================================================
def column_statistics(returns: pd.Series) -> dict[str, float | int]:
    """Descriptive statistics recorded for every column.

    Args:
        returns: Series of monthly simple returns, indexed by month.

    Returns:
        Dictionary of statistics. Every value is a float except ``n_months``, which is a count.
    """
    values = returns.to_numpy(dtype=np.float64)
    return {
        "n_months": int(values.size),
        "mean_monthly": float(values.mean()),
        "std_monthly": float(values.std(ddof=1)),
        "std_annualised": float(values.std(ddof=1) * np.sqrt(12.0)),
        "geometric_mean_annual": float((1.0 + values).prod() ** (12.0 / values.size) - 1.0),
        "autocorrelation_lag1": float(returns.autocorr(1)),
        "min_month": float(values.min()),
        "max_month": float(values.max()),
    }


def _sha256_of(path: Path) -> str:
    """SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_matrix(raw_dir: Path = RAW_DIRECTORY, config: DatasetConfig | None = None) -> pd.DataFrame:
    """Assemble the D x 4 matrix of monthly simple returns.

    Order of operations matters in one place and only one: returns are computed on the full
    raw series and the window is applied afterwards.

    Args:
        raw_dir: Directory holding the four raw files.
        config: Window, column names and bond specifications. Defaults come from ``config/dataset.yml`` if present.

    Returns:
        Frame of 420 rows by 4 columns, ``float64``, indexed by month.
    """
    config = config or load_config()

    for filename in (config.equity_filename, config.gold_filename, *(spec.filename for spec in config.bonds)):
        if not (raw_dir / filename).is_file():
            raise FileNotFoundError(f"{raw_dir / filename} not found: run the ingestion module first")

    columns: dict[str, pd.Series] = {
        config.equity_column: load_equity_returns(raw_dir / config.equity_filename),
        config.gold_column: load_gold_returns(raw_dir / config.gold_filename),
    }
    for spec in config.bonds:
        columns[spec.name] = load_bond_returns(raw_dir / spec.filename, spec)

    matrix = pd.concat([columns[name] for name in config.columns], axis=1, join="outer")
    matrix = matrix.loc[config.window_start : config.window_end].astype(np.float64)
    matrix.index.name = "month"

    incomplete = matrix.columns[matrix.isna().any()].tolist()
    if incomplete:
        raise ValueError(
            f"columns with gaps inside the window: {incomplete}. If the gap is the first month, the raw "
            "file does not reach the month before the window start, and a return needs the level of the "
            "preceding month."
        )
    constant = [name for name in matrix.columns if matrix[name].std(ddof=1) == 0.0]
    if constant:
        raise ValueError(f"columns with zero variance: {constant}. A parsing failure, not a market fact.")
    if matrix.index.has_duplicates:
        raise ValueError("duplicated months in the consolidated index")
    if len(matrix) != config.expected_rows:
        raise ValueError(
            f"{len(matrix)} rows between {config.window_start} and {config.window_end}, expected "
            f"{config.expected_rows}. A short count usually means a raw file does not reach the month "
            "before the window start."
        )

    expected_index = pd.period_range(config.window_start, config.window_end, freq="M")
    if not matrix.index.equals(expected_index):
        raise ValueError(
            f"the consolidated index is not the contiguous run of months from {config.window_start} to "
            f"{config.window_end}"
        )
    return matrix


def write_matrix(
    matrix: pd.DataFrame,
    config: DatasetConfig,
    matrix_path: Path = MATRIX_PATH,
    metadata_path: Path = MATRIX_METADATA_PATH,
) -> dict[str, object]:
    """Write the matrix as text and its metadata as JSON, and return the metadata.

    Args:
        matrix: The D x 4 frame of monthly simple returns.
        config: The configuration used to build the matrix.
        matrix_path: Path to the CSV file to write.
        metadata_path: Path to the JSON file to write.

    Returns:
        Metadata dictionary, which is also written to the JSON file.
    """
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_path, index_label="month", lineterminator="\n")

    metadata: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": {"start": config.window_start, "end": config.window_end},
        "duration_model": "par_bond",
        "columns": list(matrix.columns),
        "n_rows": int(len(matrix)),
        "sha256": _sha256_of(matrix_path),
        "statistics": {name: column_statistics(matrix[name]) for name in matrix.columns},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


# ======================================================================================
# A28: sample-mean modified durations and the fixed-income split phi
# ======================================================================================
def baselines_path() -> Path:
    """Return ``config/baselines.yml``.

    ``.yml`` is the extension this repository uses; ``.yaml`` is still accepted on read so that a
    file carried over under the other spelling is found rather than silently ignored.
    """
    for suffix in (".yml", ".yaml"):
        candidate = CONFIG_DIR / f"baselines{suffix}"
        if candidate.is_file():
            return candidate
    return CONFIG_DIR / "baselines.yml"


def sample_mean_modified_durations(raw_dir: Path, config: DatasetConfig) -> dict[str, float]:
    """Mean modified duration of each reconstructed bond leg over the window.

    The sensitivities are the ones the reconstruction actually used: recomputed every month from the
    start-of-period yield under the par-bond assumption (D36). Averaging them over the window gives
    the duration the leg carried on average, which is what the fixed-income split must reproduce.

    Args:
        raw_dir: Directory holding the raw yield files.
        config: Window and bond specifications.

    Returns:
        Mapping from column name to mean modified duration, in years.
    """
    durations: dict[str, float] = {}
    for spec in config.bonds:
        monthly = month_end_observations(read_fred_daily(raw_dir / spec.filename))
        components = return_components(monthly["yield_pct"], spec)
        components.index = components.index.to_period("M")
        window = components.loc[config.window_start : config.window_end, "modified_duration"]
        if len(window) != config.expected_rows:
            raise ValueError(
                f"{spec.name}: {len(window)} monthly durations between {config.window_start} and "
                f"{config.window_end}, expected {config.expected_rows}"
            )
        durations[spec.name] = float(window.mean())
    return durations


def solve_phi(duration_long: float, duration_short: float, target: float) -> float:
    """Solve ``phi * D_long + (1 - phi) * D_short = target`` for ``phi``.

    Args:
        duration_long: Mean modified duration of the long leg.
        duration_short: Mean modified duration of the short leg.
        target: Target duration of the aggregate index being reproduced.

    Returns:
        The share of the long leg. Not clipped: a target outside the two durations is a
        specification error and must surface as a weight outside ``[0, 1]`` rather than be hidden.
    """
    if duration_long == duration_short:
        raise ValueError("the two legs have the same mean duration; the split is undetermined")
    return (target - duration_short) / (duration_long - duration_short)


def _set_yaml_scalar(text: str, key: str, indent: int, value: str) -> str:
    """Replace the value of the single ``key`` at the given indentation, comments elsewhere intact.

    The file carries the provenance of every externally sourced parameter in its comments, so it is
    patched line by line rather than re-dumped.
    """
    pattern = re.compile(rf"^{' ' * indent}{re.escape(key)}:[^\S\n]*(?P<old>[^\n]*)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one '{key}:' at indent {indent}, found {len(matches)}")
    match = matches[0]
    return f"{text[: match.start()]}{' ' * indent}{key}: {value}{text[match.end():]}"


def _commented(key: str, indent: int, value: float, comment: str) -> str:
    """Render a numeric value with its inline comment aligned to the column used by the file."""
    rendered = f"{value:.{PHI_DECIMALS}f}"
    padding = max(1, PHI_COMMENT_COLUMN - (indent + len(key) + 2 + len(rendered)))
    return f"{rendered}{' ' * padding}# {comment}"


def _insert_yaml_block(text: str, anchor_key: str, indent: int, block: str) -> str:
    """Insert ``block`` after the mapping headed by ``anchor_key``, not inside it.

    The anchor's own entries sit at a deeper indentation and must be skipped: inserting between the
    header and its entries would reparent them to the new block and silently empty the anchor.
    """
    pattern = re.compile(rf"^{' ' * indent}{re.escape(anchor_key)}:[^\S\n]*[^\n]*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one '{anchor_key}:' at indent {indent}, found {len(matches)}")

    end = matches[0].end()
    while end < len(text):
        line_end = text.find("\n", end + 1)
        line_end = len(text) if line_end == -1 else line_end
        line = text[end + 1 : line_end]
        if not line.strip() or not line.startswith(" " * (indent + 1)):
            break
        end = line_end
    return f"{text[:end]}\n{block.rstrip()}{text[end:]}"


def write_fixed_income_split(
    durations: dict[str, float], config: DatasetConfig, path: Path | None = None
) -> tuple[float, bool]:
    """Write the sample-based ``phi`` into the baselines configuration.

    Closes A28. The nominal 0.621 of the specification came from reference durations of 8.5 and 1.9
    years; this replaces it with the value solved from the project's own reconstruction, and records
    the two durations next to it. The external provenance of the six-year target is left untouched.

    Args:
        durations: Mean modified duration per bond column.
        config: Bond specifications, used to tell the long leg from the short one.
        path: Destination; resolved from the configuration directory when omitted.

    Returns:
        Tuple ``(phi, rewritten)``.
    """
    path = path or baselines_path()
    text = path.read_text(encoding="utf-8")
    document = yaml.safe_load(text) or {}
    split = document["asset_mapping"]["fixed_income_split"]
    target = float(split["target_duration_years"])

    ordered = sorted(config.bonds, key=lambda spec: spec.maturity_years)
    short_leg, long_leg = ordered[0].name, ordered[-1].name
    phi = round(solve_phi(durations[long_leg], durations[short_leg], target), PHI_DECIMALS)

    if float(split[f"{long_leg}_share"]) == phi:
        return phi, False

    long_key, short_key = f"{long_leg}_share", f"{short_leg}_share"
    text = _set_yaml_scalar(
        text, long_key, 4, _commented(long_key, 4, phi, "phi, solved from the sample-mean durations (A28)")
    )
    text = _set_yaml_scalar(text, short_key, 4, _commented(short_key, 4, 1.0 - phi, "1 - phi"))

    if "sample_durations_used" not in document["asset_mapping"]["fixed_income_split"]:
        block = (
            "    # A28, closed on {date}. The durations above are the nominal values of the\n"
            "    # specification; the ones below are the sample means of the project's own\n"
            "    # reconstruction over the window, under D36, and they are what phi now solves from.\n"
            "    sample_durations_used:\n"
            "      {long_key}: {d_long:.6f}\n"
            "      {short_key}: {d_short:.6f}\n"
            "      window: \"{start} / {end}\"\n"
            "      n_months: {n_months}\n"
            "      definition: >\n"
            "        Mean over the window of the monthly modified duration of the notional par bond,\n"
            "        evaluated at the start-of-period yield, which is the sensitivity the\n"
            "        reconstruction of Equation eq:renta-fija actually applied.\n"
            "      command: \"uv run python -m portfolio_manager.etl.dataset.consolidate --fixed-income-report\"\n"
            "      computed_on: \"{date}\"\n"
        ).format(
            date=datetime.now(UTC).date().isoformat(),
            long_key=f"D{int(max(spec.maturity_years for spec in config.bonds))}",
            short_key=f"D{int(min(spec.maturity_years for spec in config.bonds))}",
            d_long=durations[long_leg],
            d_short=durations[short_leg],
            start=config.window_start,
            end=config.window_end,
            n_months=config.expected_rows,
        )
        text = _insert_yaml_block(text, "reference_durations_used", 4, "\n" + block)

    path.write_text(text, encoding="utf-8")
    return phi, True


def ensure_comparator_blocks(path: Path | None = None) -> list[str]:
    """Add the ``constant_mix`` and ``equal_weight`` blocks if absent (D68).

    Both comparators project onto the four-asset universe through the same ``asset_mapping`` as the
    glide path, so the two rules differ only in their temporal shape. Written before ``baselines.py``
    exists, so no weight is an implicit choice made inside the code.

    Args:
        path: Destination; resolved from the configuration directory when omitted.

    Returns:
        The blocks actually appended.
    """
    path = path or baselines_path()
    text = path.read_text(encoding="utf-8")
    document = yaml.safe_load(text) or {}
    added: list[str] = []
    if not text.endswith("\n"):
        text += "\n"

    if "constant_mix" not in document:
        text += (
            "\n"
            "# -----------------------------------------------------------------------------\n"
            "# Constant mix 60/40 (D68)\n"
            "# -----------------------------------------------------------------------------\n"
            "# The most widely used allocation rule, rebalanced at the start of every month. It is\n"
            "# projected onto the four-asset universe through the SAME asset_mapping as the glide\n"
            "# path: no gold, all equity in us_equity, fixed income split phi / 1 - phi. With the\n"
            "# mapping shared, the two heuristics differ only in their temporal shape, which is what\n"
            "# the comparison is about. Nothing here is calibrated.\n"
            "constant_mix:\n"
            "  equity_weight: 0.60\n"
            "  fixed_income_weight: 0.40\n"
            "  calibrated: false\n"
            "  uses_asset_mapping: true\n"
        )
        added.append("constant_mix")

    if "equal_weight" not in document:
        text += (
            "\n"
            "# -----------------------------------------------------------------------------\n"
            "# Equal weight, 1/N (D68)\n"
            "# -----------------------------------------------------------------------------\n"
            "# 0.25 per column, the only comparator with a direct exposure to gold. It estimates no\n"
            "# parameter at all, which is precisely why it is hard to beat (DeMiguel et al., 2009).\n"
            "equal_weight:\n"
            "  weight_per_asset: 0.25\n"
            "  calibrated: false\n"
            "  uses_asset_mapping: false\n"
        )
        added.append("equal_weight")

    if added:
        path.write_text(text, encoding="utf-8")
    return added


# ======================================================================================
# Command line
# ======================================================================================
def _print_scale_check(config: DatasetConfig) -> None:
    print("Scale check - instantaneous parallel shock of 100 bp\n")
    header = f"{'bond':8} {'y0':>6} {'duration':>9} {'convexity':>10} {'approx':>10} {'exact':>10} {'error':>9}"
    print(header)
    print("-" * len(header))
    for spec in config.bonds:
        for base in (2.0, 4.0, 8.0):
            row = scale_check(spec, 100.0, base)
            print(
                f"{spec.name:8} {base:5.1f}% "
                f"{row['modified_duration']:9.4f} {row['convexity']:10.3f} "
                f"{row['price_effect_approximate_pct']:9.3f}% "
                f"{row['price_effect_exact_pct']:9.3f}% "
                f"{row['approximation_error_bp']:7.2f}bp"
            )


def _print_fixed_income_report(raw_dir: Path, config: DatasetConfig) -> None:
    for spec in config.bonds:
        returns = load_bond_returns(raw_dir / spec.filename, spec)
        window = returns.loc[config.window_start : config.window_end]
        stats = column_statistics(window)
        print(
            f"{spec.name}: {stats['n_months']} months {window.index[0]} to {window.index[-1]} | "
            f"geometric mean {stats['geometric_mean_annual']:7.4%} | "
            f"volatility {stats['std_annualised']:7.4%} | "
            f"worst month {stats['min_month']:7.4%} | "
            f"autocorrelation {stats['autocorrelation_lag1']:6.3f}"
        )
        calendar = (1.0 + window).groupby(window.index.year).prod() - 1.0
        print("\t worst calendar years: " + ", ".join(f"{y} {v:.2%}" for y, v in calendar.nsmallest(3).items()))

    # A28: the two sample-mean modified durations and the split they imply.
    durations = sample_mean_modified_durations(raw_dir, config)
    path = baselines_path()
    target = float(
        (yaml.safe_load(path.read_text(encoding="utf-8")) or {})["asset_mapping"]["fixed_income_split"][
            "target_duration_years"
        ]
    )
    print(f"\nsample-mean modified duration, {config.window_start} to {config.window_end}:")
    for name, value in durations.items():
        print(f"\t {name}: {value:6.4f} years")

    phi, rewritten = write_fixed_income_split(durations, config, path)
    ordered = sorted(config.bonds, key=lambda spec: spec.maturity_years)
    print(
        f"\nphi * D({ordered[-1].name}) + (1 - phi) * D({ordered[0].name}) = {target} "
        f"-> phi = {phi:.6f}, 1 - phi = {1.0 - phi:.6f}"
    )
    print(f"{'rewrote' if rewritten else 'already current in'} {path}")
    added = ensure_comparator_blocks(path)
    if added:
        print(f"added comparator blocks to {path}: {', '.join(added)}")


def _print_build(raw_dir: Path, config: DatasetConfig) -> None:
    matrix = build_matrix(raw_dir, config)
    metadata = write_matrix(matrix, config)
    statistics = metadata["statistics"]
    if not isinstance(statistics, dict):  # pragma: no cover - defensive
        raise TypeError("malformed metadata")
    print(f"matrix {MATRIX_PATH}  {metadata['n_rows']} rows x {len(matrix.columns)} columns")
    print(f"sha256 {metadata['sha256']}\n")
    print(f"{'column':12} {'geom/yr':>9} {'vol/yr':>9} {'worst':>9} {'ac(1)':>8}")
    for name, values in statistics.items():
        print(
            f"{name:12} {values['geometric_mean_annual']:8.2%} {values['std_annualised']:8.2%} "
            f"{values['min_month']:8.2%} {values['autocorrelation_lag1']:8.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidation stage: raw files to the D x 4 return matrix.")
    parser.add_argument("--scale-check", action="store_true", help="run the mandatory scale check and exit")
    parser.add_argument("--fixed-income-report", action="store_true", help="diagnostics on the reconstructed series")
    parser.add_argument("--build", action="store_true", help="build and write the consolidated matrix")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIRECTORY)
    args = parser.parse_args(argv)

    config = load_config()
    try:
        if args.scale_check:
            _print_scale_check(config)
        if args.fixed_income_report:
            _print_fixed_income_report(args.raw_dir, config)
        if args.build:
            _print_build(args.raw_dir, config)
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    if not (args.scale_check or args.fixed_income_report or args.build):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
