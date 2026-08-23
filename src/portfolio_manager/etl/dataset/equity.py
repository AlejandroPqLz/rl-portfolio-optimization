"""US equity: Kenneth French research factors to monthly total returns.

The provider publishes the market return with the risk-free rate already subtracted,
because its audience studies risk premia. An investor accumulating towards a goal earns the total return,
so the subtraction is undone here: ``Mkt-RF + RF``. It is an identity, not an estimate.
"""

# Imports
# =====================================================================
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Constants
# =====================================================================
EQUITY_MISSING_MARKER: float = -99.99
EQUITY_HEADER_PREFIX: str = ",Mkt-RF"
EQUITY_MONTHLY_TERMINATOR: str = "Annual Factors"
EQUITY_ENCODING: str = "latin-1"


# Functions
# =====================================================================
def _archived_csv(path: Path) -> str:
    """Return the text of the single CSV member of a zip, decoded, without extracting it."""
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path.name}: expected exactly one CSV member, found {members}")
        return archive.read(members[0]).decode(EQUITY_ENCODING)


def load_equity_returns(path: Path) -> pd.Series:
    """Monthly simple total return of the US market portfolio.

    The provider publishes the market return net of the risk-free rate because its audience studies risk premia.
    An investor accumulating towards a goal earns the total return, so the subtraction is undone here: ``Mkt-RF + RF``.

    Args:
        path: Path to the raw zip under ``data/raw/``.

    Returns:
        Simple monthly returns in decimal form, indexed by month.
    """
    # Read the CSV text from the zip and split it into lines ------
    lines = _archived_csv(path).splitlines()
    header_indices = [i for i, line in enumerate(lines) if line.startswith(EQUITY_HEADER_PREFIX)]
    if not header_indices:
        raise ValueError(f"{path.name}: header starting with {EQUITY_HEADER_PREFIX!r} not found")
    header_index = header_indices[0]

    terminators = [i for i, line in enumerate(lines) if EQUITY_MONTHLY_TERMINATOR in line]
    if not terminators:
        raise ValueError(f"{path.name}: {EQUITY_MONTHLY_TERMINATOR!r} not found; the monthly block has no end")
    end_index = terminators[0]  # reading past this line silently mixes annual factors in

    # Parse the CSV text into a DataFrame ------
    names = [name.strip() for name in lines[header_index].split(",")][1:]
    records: dict[str, list[float]] = {name: [] for name in names}
    labels: list[str] = []
    for line in lines[header_index + 1 : end_index]:
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        labels.append(fields[0])
        for name, value in zip(names, fields[1:], strict=True):
            records[name].append(float(value))
    frame = pd.DataFrame(records, index=pd.PeriodIndex(labels, freq="M"), dtype=np.float64)

    # Mask the missing marker and compute total return ------
    frame = frame.mask(np.isclose(frame, EQUITY_MISSING_MARKER))
    total = (frame["Mkt-RF"] + frame["RF"]) / 100.0  # convert Mkt-RF + RF from percent to decimal
    total.name = "us_equity"
    return total.astype(np.float64)
