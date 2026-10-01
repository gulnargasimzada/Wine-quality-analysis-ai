# Repository B: Wine Quality Analysis Implementation Plan

## 1. Requirements

### Functional requirements

- Load the merged wine quality CSV from a configurable path and report clear errors for missing files, unreadable input, missing required columns, and invalid values.
- Support both Pandas and Polars for equivalent data loading and transformation workflows.
- Explore the dataset: dimensions, column types, missing values, duplicates, summary statistics, wine-type distribution, and quality distribution; include comparisons by `type` and `quality` where useful.
- Train and evaluate a reproducible baseline model that predicts `quality` from the wine measurements and `type`. Treat `quality` as the initial numeric regression target; compare classification only if requirements later call for quality bands or exact-class prediction.
- Produce saved plots for key distributions and relationships, with output paths configurable and directories created or reported consistently.
- Benchmark representative Pandas and Polars loading/transformation work on the same input and equivalent operations, and report measurements in a machine-readable and human-readable form.
- Expose a documented command-line or module entry point for running the analysis end to end.
- Run automated tests with pytest and provide containerized execution for reproducibility.

### Non-functional requirements

- Keep loading, EDA, modeling, plotting, and orchestration independently testable; avoid importing heavy plotting or modeling dependencies merely to load data.
- Make runs reproducible through fixed random seeds, explicit dependency versions, documented input data, and deterministic train/test splitting.
- Keep benchmark results informational, not pass/fail performance gates: timings vary by hardware, runtime, and cache state.
- Validate inputs and make data type conversions explicit. Preserve `type` as a categorical feature; do not silently coerce malformed values or drop rows without reporting it.
- Support local development and Docker execution on a mounted dataset without baking private or bulky data into the image.
- Keep generated plots, model artifacts, and benchmark outputs out of source control by default, while documenting how to retain or reproduce them.

## 2. Proposed Architecture and Code Changes

Use a small Python package under `src/`, with tests mirroring its public boundaries. The existing `wine_quality_merged.csv` is at the repository root; establish `data/raw/wine_quality_merged.csv` as the documented runtime location. During implementation, move the file if repository/data policy permits, or preserve its location and make the path configurable rather than creating an unnecessary duplicate.

```text
.
|-- data/
|   |-- raw/                 # Input dataset; mounted or locally provisioned
|   `-- processed/           # Optional derived data, generated at runtime
|-- docs/
|   |-- plan.md
|   `-- transcripts/
|-- src/
|   `-- wine_quality/
|       |-- __init__.py
|       |-- config.py
|       |-- data_loader.py
|       |-- exploration.py
|       |-- modeling.py
|       |-- plotting.py
|       |-- benchmarking.py
|       `-- cli.py
|-- tests/
|   |-- conftest.py
|   |-- test_data_loader.py
|   |-- test_exploration.py
|   |-- test_modeling.py
|   |-- test_plotting.py
|   `-- test_benchmarking.py
|-- outputs/                 # Generated artifacts; ignored by Git
|-- Dockerfile
|-- .dockerignore
|-- .gitignore
|-- pyproject.toml           # Package metadata, dependencies, pytest/tool config
`-- README.md
```

Keep the core APIs explicit about the selected dataframe engine. Prefer adapters or narrow engine-specific loading/transformation functions that return documented types; avoid converting between Pandas and Polars invisibly. Share engine-neutral validation and analysis rules where practical. Use a common transformed result contract for the benchmark so differences reflect engine work rather than differing logic.

For the initial model, use a scikit-learn regression pipeline with preprocessing that is fit only on training data: numerical feature handling and one-hot encoding for `type`. Start with a simple, interpretable baseline (for example, a tree ensemble or linear model selected during implementation), fixed random seed, and held-out metrics such as MAE and RMSE. Keep model choice behind a small function so the baseline can be changed without affecting loading or EDA.

## 3. Important Files and Responsibilities

- `src/wine_quality/config.py`: Centralize default input/output paths, supported dataframe engine names, seed, and run settings; allow command-line overrides.
- `src/wine_quality/data_loader.py`: Load CSV data with Pandas or Polars, validate required columns, normalize/validate types deliberately, and expose consistent validation errors and dataset summaries.
- `src/wine_quality/exploration.py`: Compute reusable EDA summaries and data-quality findings without plotting side effects.
- `src/wine_quality/modeling.py`: Define feature/target selection, leakage-safe preprocessing, deterministic train/test split, model fitting, and evaluation result structure.
- `src/wine_quality/plotting.py`: Build and save plots from prepared data or summaries; avoid embedding analysis logic in chart functions.
- `src/wine_quality/benchmarking.py`: Run equivalent Pandas and Polars workloads with warm-up and repeated measurements; record software versions, dataset size, operation, and timing.
- `src/wine_quality/cli.py`: Parse input path, engine, output directory, seed, and workflow options; orchestrate modules and return a useful nonzero exit status for invalid inputs or failed runs.
- `tests/conftest.py`: Provide small synthetic fixtures with the same schema as the real dataset so routine tests do not depend on a large or externally mounted file.
- `tests/test_data_loader.py`: Cover both engines, required schema, invalid/missing input, types, and missing-value behavior.
- `tests/test_exploration.py`: Verify summary values against small known fixtures, including missing values and category counts.
- `tests/test_modeling.py`: Check split reproducibility, target/feature handling, no preprocessing leakage, and metric result shape on synthetic data.
- `tests/test_plotting.py`: Use a noninteractive plotting backend and temporary output directory; verify expected files are created and non-empty.
- `tests/test_benchmarking.py`: Verify workload equivalence and result structure, not absolute speed.
- `pyproject.toml`: Declare supported Python version and pinned or bounded dependencies, package configuration, pytest settings, and any lint/type-check configuration adopted by the team.
- `Dockerfile`: Build a reproducible runtime image, install declared dependencies, set a working directory, and invoke the documented entry point without copying the dataset into the image.
- `.dockerignore`: Exclude virtual environments, caches, Git metadata, test artifacts, and local outputs from the build context.
- `.gitignore`: Exclude generated outputs, caches, virtual environments, and optionally local data according to the repository's data policy.
- `README.md`: Document setup, dataset placement/mounting, commands, outputs, tests, Docker usage, benchmark interpretation, and known assumptions.
- `docs/plan.md`: This architecture and implementation plan; update it if decisions below change.

## 4. Risks and Design Concerns

- **Schema and data types:** The current sample has 12 numeric columns plus `type`, but malformed rows or nulls may occur elsewhere. Confirm the full file's row count, nulls, unique values, and data types before fixing validation rules. Do not infer that every numeric-looking column is a model feature: exclude `quality` from features.
- **Target interpretation:** `quality` is an ordered numeric score. Regression is a practical baseline, but the domain may prefer ordinal classification or exact quality classification. Confirm this before treating baseline metrics as business acceptance criteria.
- **Pandas/Polars compatibility:** Their dtype, null, categorical, CSV parsing, and expression semantics differ. Define equivalent operations and compare normalized outputs with tolerances where floating-point aggregation can vary. Record library versions with benchmark output.
- **Fair performance comparison:** Benchmark the same file, hardware, operations, and output semantics. Separate CSV parsing from transformations if both matter; include warm-up and multiple repetitions, and report median plus spread. Avoid asserting that either library must win.
- **Dependencies and Python versions:** Select compatible Python, Polars, Pandas, scikit-learn, plotting, and pytest versions together; lock resolved versions for container builds. Confirm wheels are available for the chosen deployment platform.
- **Docker volume bindings:** Use a documented bind mount for input data and a separate writable mount for outputs. Explain Windows path syntax and container paths in the README. Ensure container UID/permissions and output-directory creation do not prevent writing plots or metrics.
- **Data handling and image size:** Do not bake the CSV into the image unless the data is explicitly small, redistributable, and intended to be versioned there. Clarify whether `data/` is tracked, ignored, or provisioned externally before moving the existing root-level CSV.
- **Reproducibility limits:** A seed and pinned packages improve repeatability, but CPU/library differences can still affect floating-point results and model behavior. Document software and runtime metadata; compare stable metrics within tolerances rather than requiring byte-identical outputs.
- **Plot testing:** GUI backends can fail in CI or containers. Force a headless backend in tests and verify artifacts without opening windows.
- **Benchmark test reliability:** Timing assertions are noisy and hardware-specific. Unit tests should validate benchmark mechanics and equivalent results; performance evaluation should be a separate manual or scheduled experiment.

## 5. Testing and Verification Steps

1. **Static setup check:** Build/install the package from `pyproject.toml`; verify imports, CLI help, dependency resolution, and pytest discovery in the selected Python environment.
2. **Unit tests:** Run `pytest` on synthetic fixtures. Cover schema and error paths, both dataframe engines, EDA summaries, deterministic modeling, metric calculations, plot creation, and benchmark result metadata.
3. **Cross-engine equivalence:** Run Pandas and Polars over the same fixture and real input. Compare row counts, normalized schema, key summaries, and transformed outputs, allowing documented numeric tolerances.
4. **Smoke test:** Run the documented CLI end to end against the real CSV. Confirm it reports the input and row count, completes EDA and model evaluation, and writes expected plots, metrics, and benchmark output to the configured output directory.
5. **Reproducibility check:** Repeat a model run with the same seed and environment; verify identical split membership and stable metrics within the chosen tolerance. Record package versions and configuration with run outputs.
6. **Docker build and run:** Build the image from a clean context, mount the dataset read-only at the documented container path, mount an output directory read-write, and run the same smoke workflow. Confirm output artifacts exist on the host and that no dataset copy is required in the image.
7. **Container test run:** Execute pytest inside the image or a dedicated test stage to confirm test dependencies and headless plotting work in the container environment.
8. **Benchmark review:** Execute repeated Pandas/Polars runs under documented conditions; report median and variability alongside versions, input size, and operations. Do not use a fixed wall-clock threshold as a CI gate.

## Implementation Decisions

- The first model is linear regression, as requested for Repository B; `quality` remains a numeric target.
- The current CSV remains at the repository root. CLI input paths are configurable and Docker mounts the data at runtime without copying it into the image.
- Docker uses Python 3.12; package metadata supports Python 3.11 through 3.13.
- The first release saves an alcohol-by-type boxplot, quality distribution, and alcohol-versus-quality scatter plot.
- EDA, model metrics, and benchmark details are saved as JSON; benchmark timing is informational and has no fixed performance threshold.
- Missing predictor values are reported and imputed inside the training-only preprocessing pipeline. Missing target values are rejected for modeling.