"""Headless plotting helpers that save analysis artifacts."""

from pathlib import Path

import pandas as pd
import polars as pl

from wine_quality.data_loader import WineFrame


def _plot_frame(frame: WineFrame) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.loc[:, ["type", "alcohol", "quality"]].copy()
    return pd.DataFrame(frame.select("type", "alcohol", "quality").to_dict(as_series=False))


def save_plots(frame: WineFrame, output_dir: str | Path) -> list[Path]:
    """Save alcohol-by-type boxplot, quality distribution, and alcohol/quality plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = _plot_frame(frame)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = [
        output_path / "alcohol_by_type_boxplot.png",
        output_path / "quality_distribution.png",
        output_path / "alcohol_vs_quality.png",
    ]

    grouped = [
        group["alcohol"].dropna().to_numpy()
        for _, group in plot_data.groupby("type", dropna=True, sort=True)
    ]
    labels = [str(value) for value in sorted(plot_data["type"].dropna().unique())]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(grouped, tick_labels=labels)
    axis.set(title="Alcohol content by wine type", xlabel="Wine type", ylabel="Alcohol (%)")
    figure.tight_layout()
    figure.savefig(paths[0], dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    plot_data["quality"].dropna().value_counts().sort_index().plot.bar(ax=axis, color="#547c65")
    axis.set(title="Wine quality distribution", xlabel="Quality", ylabel="Number of wines")
    figure.tight_layout()
    figure.savefig(paths[1], dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for wine_type, group in plot_data.groupby("type", dropna=True, sort=True):
        axis.scatter(group["quality"], group["alcohol"], alpha=0.45, label=str(wine_type))
    axis.set(title="Alcohol content and quality", xlabel="Quality", ylabel="Alcohol (%)")
    axis.legend(title="Wine type")
    figure.tight_layout()
    figure.savefig(paths[2], dpi=150)
    plt.close(figure)
    return paths