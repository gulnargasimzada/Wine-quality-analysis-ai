"""Leakage-safe linear regression baseline for wine quality."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wine_quality.config import FEATURE_COLUMNS, NUMERIC_COLUMNS, RANDOM_SEED, TARGET_COLUMN
from wine_quality.data_loader import WineFrame

NUMERIC_FEATURES = list(NUMERIC_COLUMNS[:-1])
CATEGORICAL_FEATURES = ["type"]


@dataclass
class ModelingResult:
    """Fitted estimator and reproducible held-out evaluation details."""

    pipeline: Pipeline
    metrics: dict[str, float | int]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    feature_columns: tuple[str, ...]


def _to_model_frame(frame: WineFrame) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    return pd.DataFrame(frame.select(FEATURE_COLUMNS + (TARGET_COLUMN,)).to_dict(as_series=False))


def build_regression_pipeline() -> Pipeline:
    """Build numeric and categorical preprocessing with a linear estimator."""
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    return Pipeline(steps=[("preprocessing", preprocessing), ("regressor", LinearRegression())])


def train_and_evaluate(
    frame: WineFrame, seed: int = RANDOM_SEED, test_size: float = 0.2
) -> ModelingResult:
    """Train on a deterministic split and return held-out MAE, RMSE, and R2."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")
    data = _to_model_frame(frame)
    if data.empty or len(data) < 2:
        raise ValueError("At least two rows are required to train and evaluate the model")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("The quality target contains missing values; remove or resolve them before modeling")

    features = data.loc[:, FEATURE_COLUMNS]
    target = data[TARGET_COLUMN].astype(float)
    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=seed)
    pipeline = build_regression_pipeline()
    pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
    predictions = pipeline.predict(features.iloc[test_indices])
    actual = target.iloc[test_indices]
    metrics: dict[str, Any] = {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "seed": seed,
    }
    return ModelingResult(
        pipeline=pipeline,
        metrics=metrics,
        train_indices=tuple(int(index) for index in train_indices),
        test_indices=tuple(int(index) for index in test_indices),
        feature_columns=tuple(FEATURE_COLUMNS),
    )