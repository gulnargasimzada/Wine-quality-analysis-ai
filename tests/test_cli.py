import json

from wine_quality.cli import main


def test_cli_runs_analysis_and_writes_outputs(synthetic_wine_csv, tmp_path, capsys):
    output_dir = tmp_path / "outputs"
    status = main(
        [
            "--input", str(synthetic_wine_csv),
            "--output-dir", str(output_dir),
            "--benchmark-repeats", "1",
            "--benchmark-warmups", "0",
        ]
    )
    assert status == 0
    assert json.loads((output_dir / "eda_summary.json").read_text())["shape"]["rows"] == 12
    assert json.loads((output_dir / "model_metrics.json").read_text())["seed"] == 42
    assert json.loads((output_dir / "benchmark.json").read_text())["equivalent_results"] is True
    stdout = capsys.readouterr().out
    assert "pandas: median=" in stdout
    assert "polars: median=" in stdout


def test_cli_returns_nonzero_for_invalid_input(tmp_path, capsys):
    status = main(["--input", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path / "out")])
    assert status == 1
    assert "does not exist" in capsys.readouterr().err