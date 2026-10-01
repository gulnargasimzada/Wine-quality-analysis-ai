"""Side-effect-free summaries for wine quality data."""

from collections import Counter
from typing import Any

import pandas as pd
import polars as pl

from wine_quality.data_loader import WineFrame


def _counts(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(_label(value) for value in values).items()))


def _label(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def summarize_data(frame: WineFrame) -> dict[str, Any]:
    """Return dimensions, quality checks, distributions, and numeric statistics."""
    if isinstance(frame, pd.DataFrame):
        row_count, column_count = frame.shape
        dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        missing = {column: int(count) for column, count in frame.isna().sum().items()}
        duplicates = int(frame.duplicated().sum())
        numeric = frame.select_dtypes(include="number")
        statistics = {
            column: {
                "count": int(values.count()),
                "mean": float(values.mean()) if values.count() else None,
                "std": float(values.std()) if values.count() > 1 else None,
                "min": float(values.min()) if values.count() else None,
                "q25": float(values.quantile(0.25)) if values.count() else None,
                "median": float(values.median()) if values.count() else None,
                "q75": float(values.quantile(0.75)) if values.count() else None,
                "max": float(values.max()) if values.count() else None,
            }
            for column, values in numeric.items()
        }
        types = frame["type"].tolist()
        qualities = frame["quality"].tolist()
        quality_by_type = frame.groupby("type", dropna=False)["quality"].agg(["count", "mean"])
        type_quality = frame.groupby(["type", "quality"], dropna=False).size().reset_index(name="count")
        quality_by_type_result = {
            _label(index): {"count": int(row["count"]), "mean": float(row["mean"])}
            for index, row in quality_by_type.iterrows()
        }
        type_by_quality = {
            str(quality): _counts(group["type"].tolist())
            for quality, group in frame.groupby("quality", dropna=False)
        }
    else:
        row_count, column_count = frame.shape
        dtypes = {column: str(dtype) for column, dtype in frame.schema.items()}
        missing = {column: int(count) for column, count in zip(frame.columns, frame.null_count().row(0))}
        duplicates = frame.height - frame.unique().height
        numeric_columns = [column for column, dtype in frame.schema.items() if dtype.is_numeric()]
        statistics = {}
        for column in numeric_columns:
            values = frame.get_column(column).drop_nulls()
            statistics[column] = {
                "count": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "q25": values.quantile(0.25),
                "median": values.median(),
                "q75": values.quantile(0.75),
                "max": values.max(),
            }
        types = frame.get_column("type").to_list()
        qualities = frame.get_column("quality").to_list()
        quality_by_type = frame.group_by("type").agg(
            pl.col("quality").count().alias("count"), pl.col("quality").mean().alias("mean")
        )
        quality_by_type_result = {
            "<missing>" if row["type"] is None else str(row["type"]): {
                "count": int(row["count"]), "mean": float(row["mean"])
            }
            for row in quality_by_type.to_dicts()
        }
        type_quality = frame.group_by(["type", "quality"]).len()
        type_by_quality: dict[str, dict[str, int]] = {}
        for row in type_quality.to_dicts():
            type_by_quality.setdefault(_label(row["quality"]), {})[
                _label(row["type"])
            ] = int(row["len"])

    return {
        "shape": {"rows": row_count, "columns": column_count},
        "dtypes": dtypes,
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "summary_statistics": statistics,
        "type_distribution": _counts(types),
        "quality_distribution": _counts(qualities),
        "quality_by_type": quality_by_type_result,
        "type_by_quality": type_by_quality,
    }