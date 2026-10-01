import pytest

from wine_quality.data_loader import load_wine_data
from wine_quality.modeling import train_and_evaluate


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_model_is_reproducible_and_returns_regression_metrics(synthetic_wine_csv, engine):
    frame = load_wine_data(synthetic_wine_csv, engine=engine)
    first = train_and_evaluate(frame, seed=17)
    second = train_and_evaluate(frame, seed=17)
    assert first.test_indices == second.test_indices
    assert first.train_indices == second.train_indices
    assert first.metrics == second.metrics
    assert set(first.metrics) == {"mae", "rmse", "r2", "train_rows", "test_rows", "seed"}
    assert first.metrics["test_rows"] == 3
    assert "quality" not in first.feature_columns
    assert "type" in first.feature_columns


def test_model_rejects_invalid_test_size(synthetic_wine_csv):
    frame = load_wine_data(synthetic_wine_csv)
    with pytest.raises(ValueError, match="test_size"):
        train_and_evaluate(frame, test_size=1)


def test_model_requires_nonmissing_target(synthetic_wine_csv):
    content = synthetic_wine_csv.read_text(encoding="utf-8").replace(",5,white", ",,white", 1)
    synthetic_wine_csv.write_text(content, encoding="utf-8")
    frame = load_wine_data(synthetic_wine_csv)
    with pytest.raises(ValueError, match="quality target contains missing"):
        train_and_evaluate(frame)