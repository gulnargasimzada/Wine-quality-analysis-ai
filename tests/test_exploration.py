import pytest

from wine_quality.data_loader import load_wine_data
from wine_quality.exploration import summarize_data


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_summary_reports_dimensions_nulls_duplicates_and_distributions(synthetic_wine_csv, engine):
    summary = summarize_data(load_wine_data(synthetic_wine_csv, engine=engine))
    assert summary["shape"] == {"rows": 12, "columns": 13}
    assert summary["missing_values"]["alcohol"] == 1
    assert summary["missing_values"]["type"] == 1
    assert summary["duplicate_rows"] == 1
    assert summary["type_distribution"] == {"<missing>": 1, "red": 5, "white": 6}
    assert summary["quality_distribution"] == {"5": 5, "6": 4, "7": 3}
    assert summary["quality_by_type"]["red"]["count"] == 5
    assert summary["summary_statistics"]["quality"]["mean"] == pytest.approx(5.8333333333)