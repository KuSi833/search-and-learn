import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.color import Color
from rich.console import Console
from rich.table import Table

from .colors import BLUE, CYAN, GOLD, GREEN, ORANGE, PURPLE, RED
from .util import (
    format_mean_std,
    format_runtime,
    runtime_stats,
)

# Constants
NUM_PROBLEMS = 500  # Number of problems in the benchmark


@dataclass
class ExperimentResult:
    """Represents results from a single experiment run."""

    method: str
    n: str  # Can be "4", "8", "16", or "4→8" etc.
    accuracies: List[float]  # Range of accuracy values from multiple runs
    runtimes: Optional[List[timedelta]] = (
        None  # Range of runtime values from multiple runs
    )
    flops: Optional[List[float]] = None  # Range of FLOPS values (to be added later)
    notes: str = ""


console = Console()


def display_hyperparameter_scaling_table(results: List[ExperimentResult]):
    """Display hyperparameter scaling results in a nice Rich table."""

    # Create the table
    table = Table(
        title="Hyperparameter Scaling Results",
        box=box.SIMPLE_HEAVY,
        title_style="bold blue",
    )

    table.add_column("Method", style="bold")
    table.add_column("n", justify="center")
    table.add_column("Accuracy", justify="right")
    table.add_column("# Runs", justify="center")
    table.add_column("Runtime", justify="right")
    table.add_column("Avg Latency/Problem", justify="right")
    table.add_column("Notes", style="dim")

    for result in results:
        formatted_acc = format_accuracy_stats(result)
        num_runs = len(result.accuracies)
        acc_style = get_accuracy_style(result, formatted_acc)

        # Format runtime if available
        runtime_str = format_runtime_stats(result) or "—"

        # Format latency per problem if available
        latency_str = format_latency_per_problem_stats(result) or "—"

        table.add_row(
            result.method,
            result.n,
            acc_style,
            str(num_runs),
            runtime_str,
            latency_str,
            result.notes,
        )

    console.print()
    console.print(table)


def format_accuracy_stats(result: ExperimentResult) -> str:
    """Format accuracy statistics for display."""
    # Convert to percentage if not already (CGAI is already in %)
    if result.method == "CGAI":
        display_accs = result.accuracies
        unit = "%"
    else:
        display_accs = [acc * 100 for acc in result.accuracies]
        unit = "%"

    return format_mean_std(display_accs) + unit


def format_runtime_stats(result: ExperimentResult) -> Optional[str]:
    """Format runtime statistics for display."""
    if result.runtimes is None:
        return None

    mean, std = runtime_stats(result.runtimes)
    return format_runtime(mean, std)


def format_latency_per_problem_stats(result: ExperimentResult) -> Optional[str]:
    """Format average latency per problem statistics for display (runtime/NUM_PROBLEMS)."""
    if result.runtimes is None:
        return None

    # Convert runtimes to seconds and divide by number of problems
    latencies_per_problem = [
        rt.total_seconds() / NUM_PROBLEMS for rt in result.runtimes
    ]

    mean_latency = np.mean(latencies_per_problem)
    std_latency = (
        np.std(latencies_per_problem, ddof=1) if len(latencies_per_problem) > 1 else 0.0
    )

    # Format as seconds with appropriate precision
    if mean_latency >= 1:
        return f"{mean_latency:.2f}±{std_latency:.2f}s"
    else:
        return f"{mean_latency * 1000:.0f}±{std_latency * 1000:.0f}ms"


def get_accuracy_style(result: ExperimentResult, formatted_acc: str) -> str:
    """Apply color styling based on accuracy performance."""
    if len(result.accuracies) > 1:
        # Convert to percentage for comparison
        if result.method == "CGAI":
            mean_val = np.mean(result.accuracies)
        else:
            mean_val = np.mean(result.accuracies) * 100

        if mean_val > 85:
            return f"[green]{formatted_acc}[/green]"
        elif mean_val > 83:
            return f"[yellow]{formatted_acc}[/yellow]"
        else:
            return f"[red]{formatted_acc}[/red]"
    else:
        return formatted_acc


def _rich_color_to_hex(color: Color) -> str:
    """Convert a Rich Color into a hex string usable by matplotlib."""
    triplet = color.get_truecolor()
    return f"#{triplet.red:02X}{triplet.green:02X}{triplet.blue:02X}"


def _parse_n_to_numeric(n_value: str) -> int:
    """Parse the numeric position for an n label like "2", "3", "4", or "2→3".

    We use the last number present so that "2→3" maps to 3.
    """
    numbers = re.findall(r"\d+", n_value)
    return int(numbers[-1]) if numbers else 0


def _numeric_to_n(n_value: int) -> int:
    """Return n value directly since it's already stored as n, not generations."""
    return n_value


def plot_hyperparameter_scaling(
    results: List[ExperimentResult],
    save_path: Path = Path(
        "./figures/report/hyperparameter_scaling/hyperparameter_scaling.pdf"
    ),
) -> None:
    """Create a line plot with uncertainty (mean ± std) for each method.

    - Different colors per method using the palette from colors.py
    - Shaded region shows uncertainty (±1 std)
    - Handles methods with a single n by plotting an errorbar point
    """

    # Assign distinct colors to known methods; fall back to a cycle
    color_palette: Dict[str, str] = {
        "WBoN": _rich_color_to_hex(BLUE),
        "DVTS": _rich_color_to_hex(ORANGE),
        "Beam Search": _rich_color_to_hex(PURPLE),
        "CGAI": _rich_color_to_hex(GREEN),
    }
    fallback_cycle = [
        _rich_color_to_hex(GOLD),
        _rich_color_to_hex(CYAN),
        _rich_color_to_hex(RED),
    ]

    # Group results by method
    grouped: Dict[str, List[ExperimentResult]] = {}
    for res in results:
        grouped.setdefault(res.method, []).append(res)

    plt.figure(figsize=(9, 5.5), dpi=150)  # Increased from (7, 4.2) to (9, 5.5)

    # Collect all unique n positions (already stored as n values)
    x_ticks: List[int] = []
    for res in results:
        n_val = _parse_n_to_numeric(res.n)
        x_ticks.append(n_val)
    x_ticks = sorted(set(x_ticks))
    x_ticklabels: List[str] = [str(x) for x in x_ticks]

    # Plot each method
    for idx, (method, items) in enumerate(grouped.items()):
        color = color_palette.get(method, fallback_cycle[idx % len(fallback_cycle)])

        # Prepare data
        xs: List[int] = []
        means: List[float] = []
        stds: List[float] = []
        for item in items:
            xs.append(_parse_n_to_numeric(item.n))
            if item.method == "CGAI":
                accs = item.accuracies  # already in %
            else:
                accs = [a * 100 for a in item.accuracies]
            means.append(float(np.mean(accs)))
            stds.append(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0)

        # Sort by x
        order = np.argsort(xs)
        xs = list(np.array(xs)[order])
        means = list(np.array(means)[order])
        stds = list(np.array(stds)[order])

        if len(xs) >= 2:
            plt.plot(
                xs,
                means,
                label=method,
                color=color,
                marker="o",
                linewidth=2.6,
                markersize=8,
            )  # Increased linewidth and markersize
            lower = np.array(means) - np.array(stds)
            upper = np.array(means) + np.array(stds)
            plt.fill_between(xs, lower, upper, color=color, alpha=0.2)
        else:
            # Single point: draw as errorbar
            plt.errorbar(
                xs,
                means,
                yerr=stds,
                fmt="o",
                color=color,
                elinewidth=1.6,  # Increased from 1.2 to 1.6
                capsize=4,  # Increased from 3 to 4
                markersize=8,  # Added markersize
                label=method,
            )

    plt.xlabel("n", fontsize=16)  # Increased font size
    plt.ylabel("Accuracy (%)", fontsize=16)  # Increased font size
    plt.xticks(x_ticks, x_ticklabels, fontsize=13)  # Increased tick label size
    plt.yticks(fontsize=13)  # Increased tick label size
    plt.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    plt.legend(frameon=True, fontsize=12)  # Increased legend font size
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_vs_latency(
    results: List[ExperimentResult],
    save_path: Path = Path(
        "./figures/report/hyperparameter_scaling/accuracy_vs_latency.pdf"
    ),
) -> None:
    """Create a scatter plot showing accuracy vs average latency per problem.

    This plot demonstrates the trade-off between accuracy and computational efficiency.
    X-axis: Average latency per problem (seconds), Y-axis: Accuracy (%)
    """

    # Assign distinct colors to known methods
    color_palette: Dict[str, str] = {
        "WBoN": _rich_color_to_hex(BLUE),
        "DVTS": _rich_color_to_hex(ORANGE),
        "Beam Search": _rich_color_to_hex(PURPLE),
        "CGAI": _rich_color_to_hex(GREEN),
    }
    fallback_cycle = [
        _rich_color_to_hex(GOLD),
        _rich_color_to_hex(CYAN),
        _rich_color_to_hex(RED),
    ]

    # Assign distinct markers for different n values
    marker_map = {2: "o", 3: "s", 4: "^"}  # n=2, n=3, n=4

    # Group results by method
    grouped: Dict[str, List[ExperimentResult]] = {}
    for res in results:
        grouped.setdefault(res.method, []).append(res)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=150)

    # Collect all data points for analysis
    all_points = []
    # Collect points that need latency annotations (n=4 and CGAI)
    annotate_points = []

    # Plot each method
    for idx, (method, items) in enumerate(grouped.items()):
        color = color_palette.get(method, fallback_cycle[idx % len(fallback_cycle)])

        # Collect data for this method to draw connecting lines
        method_data = []

        for item in items:
            if item.runtimes is None:
                continue  # Skip items without runtime data

            # Process accuracy data
            if item.method == "CGAI":
                acc = np.mean(item.accuracies)  # already in %
            else:
                acc = np.mean(item.accuracies) * 100

            # Process latency data
            latencies_per_problem = [
                rt.total_seconds() / NUM_PROBLEMS for rt in item.runtimes
            ]
            latency = np.mean(latencies_per_problem)

            n_value = _parse_n_to_numeric(item.n)
            marker = marker_map.get(n_value, "D")

            # Create legend label - special case for CGAI to show arrow
            if method == "CGAI":
                legend_label = f"{method} (n=2→3)"
            else:
                legend_label = f"{method} (n={n_value})"

            # Plot scatter points
            ax.scatter(
                latency,
                acc,
                color=color,
                marker=marker,
                s=104,  # Increased from 80 to ~130% (104)
                alpha=0.8,
                label=legend_label,
                edgecolors="black",
                linewidth=1.3,  # Increased from 1 to 1.3
                zorder=2,  # Above the lines
            )

            # Store data for connecting lines
            method_data.append((n_value, latency, acc))
            all_points.append((latency, acc, method, n_value))

            # Check if this point should be annotated (n=4 or CGAI)
            if n_value == 4 or method == "CGAI":
                annotate_points.append((latency, acc, method, n_value))

        # Connect points of the same method with lines
        if len(method_data) > 1:
            # Sort by n value for proper line connection
            method_data.sort(key=lambda x: x[0])
            x_coords = [point[1] for point in method_data]  # latency
            y_coords = [point[2] for point in method_data]  # accuracy

            ax.plot(
                x_coords,
                y_coords,
                color=color,
                linestyle="-",
                alpha=0.6,
                linewidth=2.6,  # Increased from 2 to 2.6
                zorder=1,  # Behind the points
            )

    # Customize axes
    ax.set_xlabel(
        "Average Latency per Problem (s)", fontsize=16
    )  # Increased from 12 to 16
    ax.set_ylabel("Accuracy (%)", fontsize=16)  # Increased from 12 to 16

    # Increase tick label sizes
    ax.tick_params(
        axis="both", which="major", labelsize=13
    )  # Increased tick label size

    # Add grid
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    # Set x-axis to log scale if there's a wide range
    latency_values = [point[0] for point in all_points]
    if latency_values and max(latency_values) / min(latency_values) > 10:
        ax.set_xscale("log")
        ax.set_xlabel(
            "Average Latency per Problem (s) [log scale]", fontsize=16
        )  # Increased from 12 to 16

    # Add horizontal reference line for Qwen2.5-Math-72B CoT performance
    cot_accuracy = 85.9  # Qwen2.5-Math-72B-Instruct Chain-of-Thought performance
    ax.axhline(
        y=cot_accuracy,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        zorder=0.5,
        label="Qwen2.5-Math-72B CoT",
    )

    # Collect specific points for comparison arrows
    wbon_n3_point = None
    beam_n4_point = None
    cgai_point = None

    for latency, acc, method, n_value in all_points:
        if method == "WBoN" and n_value == 3:
            wbon_n3_point = (latency, acc)
        elif method == "Beam Search" and n_value == 4:
            beam_n4_point = (latency, acc)
        elif method == "CGAI":
            cgai_point = (latency, acc)

    # Add axis-aligned comparison arrows
    arrow_offset = 0.02  # Small offset so arrows don't touch CGAI point

    if wbon_n3_point and cgai_point:
        # Vertical arrow showing accuracy improvement (aligned with CGAI x-position)
        acc_improvement = cgai_point[1] - wbon_n3_point[1]
        arrow_start = (cgai_point[0], wbon_n3_point[1])
        arrow_end = (
            cgai_point[0],
            cgai_point[1] - arrow_offset * (cgai_point[1] - wbon_n3_point[1]),
        )

        ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            arrowprops=dict(
                arrowstyle="->",
                color=_rich_color_to_hex(GREEN),
                alpha=0.8,
                linewidth=2,
                linestyle="-",
            ),
            zorder=1.5,
        )
        # Add accuracy improvement label
        mid_y = (arrow_start[1] + arrow_end[1]) / 2
        ax.text(
            cgai_point[0] + 0.15,  # Slightly to the right of the arrow
            mid_y + 0.5,
            f"+{acc_improvement:.1f}%",
            ha="left",
            va="center",
            fontsize=12,
            color=_rich_color_to_hex(GREEN),
            alpha=1,
            weight="bold",
        )

    if beam_n4_point and cgai_point:
        # Horizontal arrow showing latency reduction (aligned with CGAI y-position)
        latency_reduction = beam_n4_point[0] - cgai_point[0]
        arrow_start = (beam_n4_point[0], cgai_point[1])
        arrow_end = (
            cgai_point[0] + arrow_offset * (beam_n4_point[0] - cgai_point[0]) - 1.83,
            cgai_point[1],
        )

        ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            arrowprops=dict(
                arrowstyle="->",
                color=_rich_color_to_hex(GREEN),
                alpha=1,
                linewidth=2,
                linestyle="-",
            ),
            zorder=1.5,
        )
        # Add latency reduction label
        mid_x = (arrow_start[0] + arrow_end[0]) / 2
        if latency_reduction >= 1:
            reduction_text = f"-{latency_reduction:.1f}s"
        else:
            reduction_text = f"-{latency_reduction * 1000:.0f}ms"
        ax.text(
            mid_x,
            cgai_point[1] - 0.2,  # Slightly above the arrow
            reduction_text,
            ha="center",
            va="bottom",
            fontsize=12,
            color=_rich_color_to_hex(GREEN),
            alpha=0.9,
            weight="bold",
        )

    # Add latency annotations next to specific points
    for latency, acc, method, n_value in annotate_points:
        # Format latency appropriately
        if latency >= 1:
            label_text = f"{latency:.2f}s"
        else:
            label_text = f"{latency * 1000:.0f}ms"

        # Position the text next to the point (slightly offset)
        ax.annotate(
            label_text,
            xy=(latency, acc),
            xytext=(-45, 5),  # Slightly increased offset
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=13,  # Increased from 10 to 13
            color="black",
            zorder=3,
        )

    # Create legend
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, fontsize=12
    )  # Increased from 9 to 12
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def hyperparameter_scaling_report():
    """Generate the main hyperparameter scaling report."""
    RESULTS = [
        ExperimentResult(
            method="WBoN",
            n="2",
            accuracies=[0.854, 0.856, 0.84],
            runtimes=[
                timedelta(hours=1, minutes=25, seconds=10),
                timedelta(hours=1, minutes=20, seconds=59),
                timedelta(hours=1, minutes=24, seconds=36),
            ],
        ),
        ExperimentResult(
            method="WBoN",
            n="3",
            accuracies=[0.874, 0.828, 0.866],
            runtimes=[
                timedelta(hours=1, minutes=40, seconds=10),
                timedelta(hours=1, minutes=39, seconds=19),
                timedelta(hours=1, minutes=38, seconds=34),
            ],
        ),
        ExperimentResult(
            method="WBoN",
            n="4",
            accuracies=[0.858, 0.860, 0.862],
            runtimes=[
                timedelta(hours=1, minutes=54, seconds=58),
                timedelta(hours=1, minutes=56, seconds=15),
                timedelta(hours=1, minutes=53, seconds=42),
            ],
        ),
        ExperimentResult(
            method="DVTS",
            n="2",
            accuracies=[0.832, 0.834, 0.828],
            runtimes=[
                timedelta(hours=1, minutes=27, seconds=58),
                timedelta(hours=1, minutes=28, seconds=15),
                timedelta(hours=1, minutes=28, seconds=17),
            ],
        ),
        ExperimentResult(
            method="DVTS",
            n="3",
            accuracies=[0.835, 0.840, 0.838],
            runtimes=[
                timedelta(hours=1, minutes=32, seconds=50),
                timedelta(hours=1, minutes=35, seconds=57),
                timedelta(hours=1, minutes=31, seconds=56),
            ],
        ),
        ExperimentResult(
            method="DVTS",
            n="4",
            accuracies=[0.840, 0.844, 0.842],
            runtimes=[
                timedelta(hours=1, minutes=58, seconds=55),
                timedelta(hours=2, minutes=1, seconds=15),
                timedelta(hours=1, minutes=55, seconds=10),
            ],
        ),
        ExperimentResult(
            method="Beam Search",
            n="2",
            accuracies=[0.826, 0.831, 0.833],
            runtimes=[
                timedelta(hours=3, minutes=54),
                timedelta(hours=4, minutes=2),
                timedelta(hours=3, minutes=58),
            ],
        ),
        ExperimentResult(
            method="Beam Search",
            n="3",
            accuracies=[0.852, 0.856, 0.849],
            runtimes=[
                timedelta(hours=10, minutes=47),
                timedelta(hours=11, minutes=18),
                timedelta(hours=11, minutes=3),
            ],
        ),
        ExperimentResult(
            method="Beam Search",
            n="4",
            accuracies=[0.872, 0.878, 0.864],
            runtimes=[
                timedelta(hours=15, minutes=41),
                timedelta(hours=16, minutes=7),
                timedelta(hours=15, minutes=41),
            ],
        ),
        ExperimentResult(
            method="CGAI",
            n="2→3",
            accuracies=[87.40, 87.20, 87.60],
            runtimes=[
                timedelta(hours=1, minutes=25, seconds=10)
                + timedelta(minutes=13, seconds=45),
                timedelta(hours=1, minutes=24, seconds=36)
                + timedelta(minutes=13, seconds=45),
                timedelta(hours=1, minutes=24, seconds=24)
                + timedelta(minutes=13, seconds=45),
            ],
            notes="Different scale",
        ),
    ]
    display_hyperparameter_scaling_table(RESULTS)
    plot_accuracy_vs_latency(RESULTS)
