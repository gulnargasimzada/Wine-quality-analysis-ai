# Wine Quality Analysis

A reproducible analysis of the merged wine-quality CSV. The package supports Pandas and Polars loading and exploration, compares equivalent CSV workloads, evaluates a leakage-safe linear regression baseline, and saves headless plots and JSON reports.

## Requirements

- Python 3.11, 3.12, 3.13, or 3.14
- The merged dataset at `wine_quality_merged.csv` in the repository root, or a path supplied with `--input`
- Docker Desktop with Compose for container runs

The CSV is not copied into the Docker image. The checked-in/root-level dataset remains in place; runtime paths can be overridden.

## Install and Run

From the repository root, create and activate a virtual environment, then install the project and its pinned runtime/test dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the full analysis with defaults:

```powershell
python -m wine_quality.cli
```

Or select input, engine, and output directory explicitly:

```powershell
python -m wine_quality.cli --input .\wine_quality_merged.csv --engine polars --output-dir .\outputs
```

The installed `wine-quality` command is also available. Use `python -m wine_quality.cli --help` to see all options. The default workflow runs EDA and modeling with the selected engine and benchmarks both engines.

## Outputs

The output directory is created automatically. A normal run writes:

- `eda_summary.json`: dimensions, types, missing values, duplicates, distributions, grouped quality summaries, and numeric statistics
- `model_metrics.json`: held-out MAE, RMSE, R², split sizes, and seed
- `benchmark.json`: engine versions, input size, equivalent workload results, and repeated timing summaries
- `alcohol_by_type_boxplot.png`, `quality_distribution.png`, and `alcohol_vs_quality.png`

Benchmark times are descriptive measurements, not performance gates. Hardware, cache state, and library versions affect them. Input feature nulls are reported in EDA and imputed using statistics fit on the training split; missing quality targets are rejected by the model.

## Test

Run unit and CLI tests against synthetic data; the tests do not depend on the real dataset:

```powershell
python -m pytest
```

## Docker

Build and run the default analysis, mounting the CSV read-only and outputs as a writable host directory:

```powershell
docker compose build
docker compose run --rm analysis
```

Run the suite in the container:

```powershell
docker compose run --rm test
```

The `analysis` service reads `wine_quality_merged.csv` from the host at `./wine_quality_merged.csv` and writes plots and JSON summaries to the host `./outputs` directory. The dataset is never copied into the image.

For a different dataset on Windows, replace the input bind mount in `docker-compose.yml` with an absolute host path such as `C:/datasets/wine_quality_merged.csv:/data/wine_quality_merged.csv:ro`. The output mount can similarly be changed. To run the image directly, pass container paths explicitly:

```powershell
docker run --rm -v "${PWD}/wine_quality_merged.csv:/data/wine_quality_merged.csv:ro" -v "${PWD}/outputs:/outputs" wine-quality-analysis:local --input /data/wine_quality_merged.csv --output-dir /outputs
```

## Modeling Assumptions

`quality` is treated as a numeric regression target. All other measured numeric columns and `type` are predictors; preprocessing is fit only on training rows. The fixed random seed and split make evaluations reproducible within the same software/runtime environment, but do not make metrics a business acceptance threshold.

## Manual Smoke Test Results
- **Setup & Environment:** Validated dependencies (`pip install -r requirements.txt`).
- **Unit Tests:** Executed `python -m pytest`; all 21 unit tests passed.
- **Pipeline Execution:** Ran `python -m wine_quality.cli` on 6,497 rows.
- **Model Verification:** Linear regression trained successfully (MAE: 0.564, RMSE: 0.736, R²: 0.267).
- **Benchmark Integrity:** Confirmed output equivalence between Pandas and Polars (Polars median: 0.039s vs. Pandas median: 0.061s).
- **Artifacts:** Verified 3 plots and benchmark summaries generated under `outputs/`.