"""Command-line entry point for the complete wine quality analysis."""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from wine_quality.benchmarking import run_benchmark
from wine_quality.config import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_DIR, RANDOM_SEED, SUPPORTED_ENGINES
from wine_quality.data_loader import load_wine_data
from wine_quality.exploration import summarize_data
from wine_quality.modeling import train_and_evaluate
from wine_quality.plotting import save_plots


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore and model the merged wine quality dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to the merged wine CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSON and plot outputs")
    parser.add_argument("--engine", choices=SUPPORTED_ENGINES, default="pandas", help="Dataframe engine for analysis")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for the model split")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for evaluation")
    parser.add_argument("--benchmark-repeats", type=int, default=3, help="Measured runs per dataframe engine")
    parser.add_argument("--benchmark-warmups", type=int, default=1, help="Unmeasured warm-up runs per engine")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame = load_wine_data(args.input, engine=args.engine)
        summary = summarize_data(frame)
        modeling = train_and_evaluate(frame, seed=args.seed, test_size=args.test_size)
        plots = save_plots(frame, args.output_dir)
        benchmark = run_benchmark(args.input, repeats=args.benchmark_repeats, warmups=args.benchmark_warmups)

        _write_json(args.output_dir / "eda_summary.json", summary)
        _write_json(args.output_dir / "model_metrics.json", modeling.metrics)
        _write_json(args.output_dir / "benchmark.json", benchmark)
        print(f"Loaded {summary['shape']['rows']} rows x {summary['shape']['columns']} columns using {args.engine}.")
        print(f"Model metrics: {json.dumps(modeling.metrics, sort_keys=True)}")
        print(f"Pandas/Polars benchmark results equivalent: {benchmark['equivalent_results']}")
        for engine, measurement in benchmark["engines"].items():
            print(
                f"{engine}: median={measurement['median_seconds']:.6f}s, "
                f"min={measurement['min_seconds']:.6f}s, max={measurement['max_seconds']:.6f}s "
                f"(n={benchmark['repeats']}, warmups={benchmark['warmups']}, "
                f"version={benchmark['versions'][engine]})"
            )
        print(f"Wrote outputs to {args.output_dir.resolve()} ({len(plots)} plots).")
        return 0
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"wine-quality: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())