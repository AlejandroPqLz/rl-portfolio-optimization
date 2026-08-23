"""Gold: World Bank monthly commodity prices to monthly simple returns.

The published figure is the average of the month rather than the close of its last trading day.
It is the only column of the universe in that convention: averaging acts as a moving-average filter, so the measured
volatility is understated, the contemporaneous correlation with equity is attenuated, and positive first-order
autocorrelation appears even though the underlying daily series has no memory.
"""

# Imports
# =====================================================================
from pathlib import Path

import numpy as np
import pandas as pd

# Constants
# =====================================================================
GOLD_SHEET: str = "Monthly Prices"
GOLD_HEADER_ROW: int = 4
GOLD_FIRST_DATA_ROW: int = 6
GOLD_COLUMN_HEADER: str = "Gold"
GOLD_MISSING_MARKER: str = "\u2026"


# Function
# =====================================================================
def load_gold_returns(path: Path) -> pd.Series:
    """Monthly simple return of the gold price.

    Simple, not logarithmic: the log return of a portfolio is not the weighted sum of the log returns of its components,
    and the state transition computes ``w'r``, where ``w`` is the weight vector of the portfolio and ``r`` is the vector
    of simple returns of its components. The published figure is the average of the month rather than the close of its
    last trading day.

    Args:
        path: Path to the raw workbook under ``data/raw/``.

    Returns:
        Simple monthly returns in decimal form, indexed by month.
    """
    # Get gold prices ------
    sheet = pd.read_excel(path, sheet_name=GOLD_SHEET, header=None)
    headers = sheet.iloc[GOLD_HEADER_ROW].astype(str).str.strip()
    matches = [position for position, name in enumerate(headers) if name == GOLD_COLUMN_HEADER]
    if len(matches) != 1:
        # Located by name and never by position: the column shifts when commodities are added.
        raise ValueError(f"{path.name}: expected one column named {GOLD_COLUMN_HEADER!r}, found {len(matches)}")
    column = matches[0]

    body = sheet.iloc[GOLD_FIRST_DATA_ROW:]
    labels = body.iloc[:, 0].astype(str).str.strip().str.replace("M", "-", regex=False)
    values = pd.to_numeric(
        body.iloc[:, column].astype(str).str.strip().replace(GOLD_MISSING_MARKER, np.nan),
        errors="coerce",
    )

    # Compute simple returns ------
    prices = pd.Series(values.to_numpy(dtype=np.float64), index=pd.PeriodIndex(labels, freq="M")).dropna()
    returns = prices / prices.shift(1) - 1.0
    returns.name = "gold"
    return returns.dropna().astype(np.float64)
