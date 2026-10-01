"""Comparable Pandas and Polars loading and transformation benchmark."""

import math
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from wine_quality.data_loader import load_wine_data


def _workload(path: Path, engine: str) -> dict[str, Any]:
    frame = load_wine_data(path, engine=engine)  # type: ignore[arg-type]
    if isinstance(frame, pd.DataFrame):
        means = frame.select_dtypes(include="number").mean().to_dict()
        result = {
            "rows": len(frame),
            "columns": sorted(frame.columns),
            "missing": {column: int(count) for column, count in frame.isna().sum().items()},
            "duplicates": len(frame) - len(frame.drop_duplicates()),
            "quality_counts": {
                str(int(key)) if float(key).is_integer() else str(key): int(value)
                for key, value in frame["quality"].value_counts(dropna=False).items()
            },
            "numeric_means": {column: float(value) if pd.notna(value) else None for column, value in means.items()},
        }
    else:
        numeric_columns = [column for column, dtype in frame.schema.items() if dtype.is_numeric()]
        means = frame.select(pl.col(numeric_columns).mean()).row(0, named=True)
        result = {
            "rows": frame.height,
            "columns": sorted(frame.columns),
            "missing": {column: int(count) for column, count in zip(frame.columns, frame.null_count().row(0))},
            "duplicates": frame.height - frame.unique().height,
            "quality_counts": {
                str(int(row["quality"])) if float(row["quality"]).is_integer() else str(row["quality"]): int(row["len"])
                for row in frame.group_by("quality").len().to_dicts()
            },
            "numeric_means": {column: value for column, value in means.items()},
        }
    return result


def _equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if any(left[key] != right[key] for key in ("rows", "columns", "missing", "duplicates", "quality_counts")):
        return False
    return all(
        (left_value is None and right["numeric_means"][column] is None)
        or (
            left_value is not None
            and right["numeric_means"][column] is not None
            and math.isclose(left_value, right["numeric_means"][column], rel_tol=1e-9, abs_tol=1e-9)
        )
        for column, left_value in left["numeric_means"].items()
    )


def run_benchmark(
    path: str | Path, repeats: int = 3, warmups: int = 1
) -> dict[str, Any]:
    """Measure equivalent loading, validation, and summary operations for both engines."""
    input_path = Path(path).expanduser()
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmups < 0:
        raise ValueError("warmups cannot be negative")

    results: dict[str, Any] = {}
    for engine in ("pandas", "polars"):
        for _ in range(warmups):
            _workload(input_path, engine)
        durations = []
        workload_result = None
        for _ in range(repeats):
            start = time.perf_counter()
            workload_result = _workload(input_path, engine)
            durations.append(time.perf_counter() - start)
        results[engine] = {
            "median_seconds": statistics.median(durations),
            "min_seconds": min(durations),
            "max_seconds": max(durations),
            "stdev_seconds": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "result": workload_result,
        }

    pandas_result = results["pandas"]["result"]
    polars_result = results["polars"]["result"]
    return {
        "input": str(input_path.resolve()),
        "input_bytes": input_path.stat().st_size,
        "operation": "CSV load, schema validation, null/duplicate counts, quality counts, numeric means",
        "repeats": repeats,
        "warmups": warmups,
        "versions": {"pandas": pd.__version__, "polars": pl.__version__},
        "equivalent_results": _equivalent(pandas_result, polars_result),
        "engines": results,
    }