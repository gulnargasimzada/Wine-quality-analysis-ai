"""Shared paths and dataset settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "wine_quality_merged.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUPPORTED_ENGINES = ("pandas", "polars")
RANDOM_SEED = 42

NUMERIC_COLUMNS = (
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
)
CATEGORICAL_COLUMNS = ("type",)
REQUIRED_COLUMNS = (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS)
FEATURE_COLUMNS = (*NUMERIC_COLUMNS[:-1], "type")
TARGET_COLUMN = "quality"