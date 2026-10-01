"""CSV loading and shared schema validation for both dataframe engines."""

import csv
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

from wine_quality.config import NUMERIC_COLUMNS, REQUIRED_COLUMNS

Engine = Literal["pandas", "polars"]
WineFrame = pd.DataFrame | pl.DataFrame


class WineDataError(ValueError):
    """Raised when a wine dataset cannot satisfy the required schema."""


def _validate_columns(columns: list[str], path: Path) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise WineDataError(f"Missing required columns in {path}: {', '.join(missing)}")


def _load_pandas(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _validate_columns(list(frame.columns), path)
    for column in NUMERIC_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise WineDataError(f"Invalid numeric value in column '{column}' in {path}: {error}") from error
        values = frame[column].dropna()
        if not values.map(lambda value: abs(value) != float("inf")).all():
            raise WineDataError(f"Infinite numeric value in column '{column}' in {path}")
    frame["type"] = frame["type"].astype(object)
    return frame


def _load_polars(path: Path) -> pl.DataFrame:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        _validate_columns(next(csv.reader(stream), []), path)
    frame = pl.read_csv(path, try_parse_dates=False)
    for column in NUMERIC_COLUMNS:
        try:
            frame = frame.with_columns(pl.col(column).cast(pl.Float64, strict=True))
        except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise WineDataError(f"Invalid numeric value in column '{column}' in {path}: {error}") from error
        if frame.select((pl.col(column).is_infinite() & pl.col(column).is_not_null()).any()).item():
            raise WineDataError(f"Infinite numeric value in column '{column}' in {path}")
    frame = frame.with_columns(pl.col("type").cast(pl.String, strict=True))
    return frame


def load_wine_data(path: str | Path, engine: Engine = "pandas") -> WineFrame:
    """Load and validate the wine CSV as an explicit Pandas or Polars frame.

    Missing measurements remain null for EDA and are imputed by the modeling
    pipeline. Invalid numeric values and infinite values are rejected.
    """
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist or is not a file: {input_path}")
    if engine not in ("pandas", "polars"):
        raise ValueError(f"Unsupported engine '{engine}'. Choose 'pandas' or 'polars'.")
    try:
        if engine == "pandas":
            return _load_pandas(input_path)
        return _load_polars(input_path)
    except WineDataError:
        raise
    except (OSError, UnicodeError, pd.errors.ParserError, pl.exceptions.PolarsError) as error:
        raise WineDataError(f"Could not read CSV {input_path}: {error}") from error