"""Action A26: the gold loader must reject a gap in the price series before differencing (D41).

A return is a ratio of two consecutive prices. If the price of the month before the window start is
missing, the ratio spans two months and is written as the return of a single one: the series comes
out with the right number of rows, the right index and a silently wrong first observation, with
exit code zero and no warning anywhere. The guard that rejects the gap is therefore the only thing
standing between a provider's missing cell and a corrupted column, and this test fixes it.
"""

# Imports
# =====================================================================
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from portfolio_manager.etl.dataset.gold import (
    GOLD_COLUMN_HEADER,
    GOLD_FIRST_DATA_ROW,
    GOLD_HEADER_ROW,
    GOLD_SHEET,
    load_gold_returns,
)

# Constants
# =====================================================================
MONTHS = ["1989M11", "1989M12", "1990M01", "1990M02", "1990M03"]
PRICES = [380.0, 400.0, 410.0, 405.0, 420.0]


# Helpers
# =====================================================================
def _write_workbook(path: Path, months: list[str], prices: list[float], *, extra_column: bool = False) -> Path:
    """Write a workbook with the layout the World Bank publishes.

    The header sits on its own row with the unit row below it, data starts two rows further down and
    the month labels use the ``1990M01`` form. A second commodity column is optional, because the
    loader locates ``Gold`` by name precisely so that an added commodity cannot shift it.

    Args:
        path: Destination file.
        months: Month labels, one per row.
        prices: Gold price per row.
        extra_column: Whether to place another commodity before the gold column.

    Returns:
        The path written.
    """
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = GOLD_SHEET

    headers: list[str] = [""]
    if extra_column:
        headers.append("Crude oil")
    headers.append(GOLD_COLUMN_HEADER)
    gold_column = len(headers)

    for position, name in enumerate(headers, start=1):
        sheet.cell(row=GOLD_HEADER_ROW + 1, column=position, value=name)
        sheet.cell(row=GOLD_HEADER_ROW + 2, column=position, value="($/troy oz)" if name else "")

    for offset, (label, price) in enumerate(zip(months, prices, strict=True)):
        row = GOLD_FIRST_DATA_ROW + 1 + offset
        sheet.cell(row=row, column=1, value=label)
        if extra_column:
            sheet.cell(row=row, column=2, value=70.0)
        sheet.cell(row=row, column=gold_column, value=price)

    workbook.save(path)
    return path


# Tests
# =====================================================================
def test_contiguous_series_produces_one_return_per_month_gap(tmp_path):
    """The control case: with no gap the loader returns one return per consecutive pair."""
    path = _write_workbook(tmp_path / "gold.xlsx", MONTHS, PRICES)
    returns = load_gold_returns(path)

    assert len(returns) == len(MONTHS) - 1
    assert str(returns.index[0]) == "1989-12"
    assert returns.dtype == np.float64
    np.testing.assert_allclose(returns.to_numpy(), np.diff(PRICES) / np.array(PRICES[:-1]))


def test_gap_in_the_price_series_is_rejected_before_differencing(tmp_path):
    """A missing month must raise, naming the month, instead of producing a two-month return."""
    months = [label for label in MONTHS if label != "1989M12"]
    prices = [price for label, price in zip(MONTHS, PRICES, strict=True) if label != "1989M12"]
    path = _write_workbook(tmp_path / "gold_gap.xlsx", months, prices)

    with pytest.raises(ValueError, match="months with no price"):
        load_gold_returns(path)


def test_the_rejected_gap_is_the_one_that_would_corrupt_january_1990(tmp_path):
    """The failure names the missing month, which is what makes the message actionable.

    Without the guard this exact gap yields a December-to-February ratio labelled as the return of
    January 1990, the first month of the window, and nothing downstream can detect it.
    """
    months = [label for label in MONTHS if label != "1990M01"]
    prices = [price for label, price in zip(MONTHS, PRICES, strict=True) if label != "1990M01"]
    path = _write_workbook(tmp_path / "gold_january_gap.xlsx", months, prices)

    with pytest.raises(ValueError, match=r"1990-01"):
        load_gold_returns(path)


def test_missing_marker_is_treated_as_an_absent_month(tmp_path):
    """The provider writes an ellipsis rather than leaving the cell empty; both are a gap."""
    path = _write_workbook(tmp_path / "gold_marker.xlsx", MONTHS, PRICES)
    frame = pd.read_excel(path, sheet_name=GOLD_SHEET, header=None)
    frame.iloc[GOLD_FIRST_DATA_ROW + 1, 1] = "…"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=GOLD_SHEET, header=False, index=False)

    with pytest.raises(ValueError, match="months with no price"):
        load_gold_returns(path)


def test_gold_column_is_located_by_name_and_not_by_position(tmp_path):
    """An added commodity shifts the column; locating it by name is what survives that."""
    plain = load_gold_returns(_write_workbook(tmp_path / "plain.xlsx", MONTHS, PRICES))
    shifted = load_gold_returns(
        _write_workbook(tmp_path / "shifted.xlsx", MONTHS, PRICES, extra_column=True)
    )
    np.testing.assert_allclose(plain.to_numpy(), shifted.to_numpy())


def test_duplicate_gold_columns_are_rejected(tmp_path):
    """Two columns of that name means the sheet changed shape; guessing one would be worse."""
    path = tmp_path / "duplicated.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = GOLD_SHEET
    for column in (2, 3):
        sheet.cell(row=GOLD_HEADER_ROW + 1, column=column, value=GOLD_COLUMN_HEADER)
    for offset, (label, price) in enumerate(zip(MONTHS, PRICES, strict=True)):
        sheet.cell(row=GOLD_FIRST_DATA_ROW + 1 + offset, column=1, value=label)
        sheet.cell(row=GOLD_FIRST_DATA_ROW + 1 + offset, column=2, value=price)
        sheet.cell(row=GOLD_FIRST_DATA_ROW + 1 + offset, column=3, value=price)
    workbook.save(path)

    with pytest.raises(ValueError, match="expected one column"):
        load_gold_returns(path)
