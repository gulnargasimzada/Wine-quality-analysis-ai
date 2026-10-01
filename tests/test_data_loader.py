import pandas as pd
import polars as pl
import pytest

from wine_quality.data_loader import WineDataError, load_wine_data


@pytest.mark.parametrize("engine,expected_type", [("pandas", pd.DataFrame), ("polars", pl.DataFrame)])
def test_loads_both_engines_and_preserves_missing_measurements(synthetic_wine_csv, engine, expected_type):
    frame = load_wine_data(synthetic_wine_csv, engine=engine)
    assert isinstance(frame, expected_type)
    assert frame.shape == (12, 13)
    if engine == "pandas":
        assert frame["alcohol"].isna().sum() == 1
    else:
        assert frame["alcohol"].null_count() == 1


def test_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_wine_data(tmp_path / "missing.csv")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_missing_required_column_is_rejected(synthetic_wine_csv, engine):
    content = synthetic_wine_csv.read_text(encoding="utf-8").replace(",quality,type", ",type")
    synthetic_wine_csv.write_text(content, encoding="utf-8")
    with pytest.raises(WineDataError, match="Missing required columns.*quality"):
        load_wine_data(synthetic_wine_csv, engine=engine)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_malformed_numeric_value_is_rejected(synthetic_wine_csv, engine):
    content = synthetic_wine_csv.read_text(encoding="utf-8").replace("7.0,0.3", "not-a-number,0.3", 1)
    synthetic_wine_csv.write_text(content, encoding="utf-8")
    with pytest.raises(WineDataError, match="Invalid numeric value.*fixed acidity"):
        load_wine_data(synthetic_wine_csv, engine=engine)


def test_unknown_engine_is_rejected(synthetic_wine_csv):
    with pytest.raises(ValueError, match="Unsupported engine"):
        load_wine_data(synthetic_wine_csv, engine="other")