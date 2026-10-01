"""Shared synthetic wine fixture and headless plotting configuration."""

import csv

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture
def synthetic_wine_csv(tmp_path):
    columns = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates",
        "alcohol", "quality", "type",
    ]
    rows = []
    for index in range(12):
        rows.append(
            {
                "fixed acidity": 7.0 + index / 10,
                "volatile acidity": 0.3 + index / 100,
                "citric acid": 0.2 + index / 100,
                "residual sugar": 2.0 + index / 10,
                "chlorides": 0.05 + index / 1000,
                "free sulfur dioxide": 10 + index,
                "total sulfur dioxide": 30 + index,
                "density": 0.990 + index / 10000,
                "pH": 3.0 + index / 100,
                "sulphates": 0.5 + index / 100,
                "alcohol": 9.0 + index / 10,
                "quality": 5 + index % 3,
                "type": "red" if index % 2 else "white",
            }
        )
    rows[-1] = rows[0].copy()
    rows[1]["alcohol"] = ""
    rows[2]["type"] = ""
    path = tmp_path / "wines.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path