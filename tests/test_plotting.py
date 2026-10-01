import pytest

from wine_quality.data_loader import load_wine_data
from wine_quality.plotting import save_plots


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_plotting_saves_expected_nonempty_artifacts(synthetic_wine_csv, tmp_path, engine):
    paths = save_plots(load_wine_data(synthetic_wine_csv, engine=engine), tmp_path / "plots")
    assert {path.name for path in paths} == {
        "alcohol_by_type_boxplot.png",
        "quality_distribution.png",
        "alcohol_vs_quality.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)