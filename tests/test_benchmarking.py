import pytest

from wine_quality.benchmarking import run_benchmark


def test_benchmark_reports_equivalence_versions_and_timings(synthetic_wine_csv):
    result = run_benchmark(synthetic_wine_csv, repeats=2, warmups=0)
    assert result["equivalent_results"] is True
    assert result["versions"]["pandas"]
    assert result["versions"]["polars"]
    assert result["engines"]["pandas"]["result"]["rows"] == 12
    assert result["engines"]["polars"]["median_seconds"] >= 0
    assert result["engines"]["polars"]["stdev_seconds"] >= 0


@pytest.mark.parametrize("repeats,warmups", [(0, 0), (1, -1)])
def test_benchmark_rejects_invalid_run_counts(synthetic_wine_csv, repeats, warmups):
    with pytest.raises(ValueError):
        run_benchmark(synthetic_wine_csv, repeats=repeats, warmups=warmups)