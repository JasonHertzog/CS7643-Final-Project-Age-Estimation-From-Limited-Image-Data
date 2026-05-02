"""Generate side-by-side metric comparison plots for experiment results.

By default, running this script with no arguments writes the linear-regression
baseline comparison plot to `outputs/linear_regression_baseline_comparison.png`.
Use `--graphs-only` when the title and note should be supplied by LaTeX
instead of embedded in the image. I used this option for the paper.

To reuse the plotting code for another comparison you need to provide a
`MetricComparisonConfig` whose `rows` contain one `ComparisonRow` per
model setup. Each row supplies the x-axis label, the two primary bar values,
and one secondary metric value. `plot_metric_comparison` creates the output
directory if needed, saves a PNG at `config.output_path`, and returns that
path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

TECH_GOLD = "#B3A369"
NAVY_BLUE = "#003057"
GRAY_MATTER = "#54585A"


@dataclass(frozen=True)
class ComparisonRow:
    label: str
    left_metric: float
    right_metric: float
    secondary_metric: float


@dataclass(frozen=True)
class MetricComparisonConfig:
    rows: tuple[ComparisonRow, ...]
    output_path: Path
    title: str
    left_metric_label: str
    right_metric_label: str
    primary_axis_title: str
    primary_axis_label: str
    secondary_axis_title: str
    secondary_axis_label: str
    secondary_scale: float
    secondary_value_format: str
    note: str
    show_title: bool
    show_note: bool


@dataclass(frozen=True)
class CliArgs:
    output: Path | None
    hide_title: bool
    hide_note: bool
    graphs_only: bool


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config() -> MetricComparisonConfig:
    root = project_root()
    return MetricComparisonConfig(
        rows=(
            ComparisonRow(
                label="Random init\nend-to-end",
                left_metric=7.5404,
                right_metric=7.4680,
                secondary_metric=0.4961,
            ),
            ComparisonRow(
                label="Frozen ImageNet\nfeatures",
                left_metric=9.9527,
                right_metric=10.1885,
                secondary_metric=0.3355,
            ),
            ComparisonRow(
                label="ImageNet init\nfull fine-tune",
                left_metric=5.8378,
                right_metric=5.9290,
                secondary_metric=0.5823,
            ),
        ),
        output_path=root / "outputs" / "linear_regression_baseline_comparison.png",
        title="Simple linear regression head baselines",
        left_metric_label="Best validation MAE",
        right_metric_label="Test MAE",
        primary_axis_title="Age error by backbone setting",
        primary_axis_label="MAE in years lower is better",
        secondary_axis_title="Predictions within five years",
        secondary_axis_label="Acc@5 percent higher is better",
        secondary_scale=100.0,
        secondary_value_format="{value:.1f}%",
        note="All three models use the same Linear(512,1) age-regression head.",
        show_title=True,
        show_note=True,
    )


def hide_top_right_spines(axis: Axes) -> None:
    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)


def label_bars(axis: Axes, bars: BarContainer, value_format: str) -> None:
    labels = [value_format.format(value=bar.get_height()) for bar in bars]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8)


def plot_metric_comparison(config: MetricComparisonConfig) -> Path:
    if len(config.rows) == 0:
        raise ValueError("At least one comparison row is required.")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    figure_obj, axes_obj = plt.subplots(1, 2, figsize=(10.8, 5.4), dpi=180)
    figure = cast(Figure, figure_obj)
    axes = cast(Sequence[Axes], axes_obj)
    primary_axis = axes[0]
    secondary_axis = axes[1]

    labels = [row.label for row in config.rows]
    positions = [float(index) for index in range(len(config.rows))]
    left_values = [row.left_metric for row in config.rows]
    right_values = [row.right_metric for row in config.rows]
    secondary_values = [
        row.secondary_metric * config.secondary_scale for row in config.rows
    ]

    width = 0.35
    left_positions = [position - width / 2.0 for position in positions]
    right_positions = [position + width / 2.0 for position in positions]

    figure.patch.set_facecolor("white")
    val_bars = cast(
        BarContainer,
        primary_axis.bar(
            left_positions,
            left_values,
            width,
            label=config.left_metric_label,
            color=TECH_GOLD,
        ),
    )
    test_bars = cast(
        BarContainer,
        primary_axis.bar(
            right_positions,
            right_values,
            width,
            label=config.right_metric_label,
            color=NAVY_BLUE,
        ),
    )
    primary_axis.set_title(config.primary_axis_title)
    primary_axis.set_ylabel(config.primary_axis_label)
    primary_axis.set_xticks(positions)
    primary_axis.set_xticklabels(labels)
    primary_axis.set_ylim(0, max(left_values + right_values) * 1.18)
    primary_axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    hide_top_right_spines(primary_axis)
    label_bars(primary_axis, val_bars, "{value:.2f}")
    label_bars(primary_axis, test_bars, "{value:.2f}")

    secondary_bars = cast(
        BarContainer,
        secondary_axis.bar(
            positions,
            secondary_values,
            color=[TECH_GOLD, GRAY_MATTER, NAVY_BLUE],
        ),
    )
    secondary_axis.set_title(config.secondary_axis_title)
    secondary_axis.set_ylabel(config.secondary_axis_label)
    secondary_axis.set_xticks(positions)
    secondary_axis.set_xticklabels(labels)
    secondary_axis.set_ylim(0, max(secondary_values) * 1.18)
    secondary_axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    hide_top_right_spines(secondary_axis)
    label_bars(secondary_axis, secondary_bars, config.secondary_value_format)

    if config.show_title:
        figure.suptitle(config.title, fontsize=15, fontweight="bold", y=1.0)
    figure.legend(
        handles=[val_bars, test_bars],
        labels=[config.left_metric_label, config.right_metric_label],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.285, 0.07),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.2,
        columnspacing=1.4,
    )
    if config.show_note:
        figure.text(
            0.5,
            0.015,
            config.note,
            ha="center",
            fontsize=10,
            color="#444444",
        )
    top_margin = 0.95 if config.show_title else 1.0
    bottom_margin = 0.16 if config.show_note else 0.13
    figure.tight_layout(pad=2.0, rect=(0.0, bottom_margin, 1.0, top_margin))
    figure.savefig(config.output_path, bbox_inches="tight")
    plt.close(figure)
    return config.output_path


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hide-title", action="store_true")
    parser.add_argument("--hide-note", action="store_true")
    parser.add_argument("--graphs-only", action="store_true")
    namespace = parser.parse_args()
    return CliArgs(
        output=cast(Path | None, namespace.output),
        hide_title=cast(bool, namespace.hide_title),
        hide_note=cast(bool, namespace.hide_note),
        graphs_only=cast(bool, namespace.graphs_only),
    )


def main() -> None:
    args = parse_args()
    config = default_config()
    show_title = not (args.hide_title or args.graphs_only)
    show_note = not (args.hide_note or args.graphs_only)
    config = replace(config, show_title=show_title, show_note=show_note)
    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else project_root() / args.output
        config = replace(config, output_path=output_path)
    saved_path = plot_metric_comparison(config)
    print(saved_path)


if __name__ == "__main__":
    main()
