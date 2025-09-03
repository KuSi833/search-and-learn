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
    """Parse the numeric position for an n label like "4", "8", or "4→8".

    We use the last number present so that "4→8" maps to 8.
    """
    numbers = re.findall(r"\d+", n_value)
    return int(numbers[-1]) if numbers else 0


def _numeric_to_n(numeric_generations: int) -> int:
    """Convert total generations (e.g., 4, 8, 16) to n where generations=2^n."""
    if numeric_generations <= 0:
        return 0
    return int(np.log2(numeric_generations))


def plot_hyperparameter_scaling(
    results: List[ExperimentResult],
    save_path: Path = Path(
        "./figures/report/hyperparameter_scaling/hyperparameter_scaling.png"
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

    plt.figure(figsize=(7, 4.2), dpi=150)

    # Collect all unique n positions (n = log2(generations))
    x_ticks: List[int] = []
    for res in results:
        numeric = _parse_n_to_numeric(res.n)
        n_val = _numeric_to_n(numeric)
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
            xs.append(_numeric_to_n(_parse_n_to_numeric(item.n)))
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
            plt.plot(xs, means, label=method, color=color, marker="o", linewidth=2)
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
                elinewidth=1.2,
                capsize=3,
                label=method,
            )

    plt.xlabel("n")
    plt.ylabel("Accuracy (%)")
    plt.xticks(x_ticks, x_ticklabels)
    plt.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_vs_latency(
    results: List[ExperimentResult],
    save_path: Path = Path(
        "./figures/report/hyperparameter_scaling/accuracy_vs_latency.png"
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
    marker_map = {2: "o", 3: "s", 4: "^"}  # n=2 (4 gens), n=3 (8 gens), n=4 (16 gens)

    # Group results by method
    grouped: Dict[str, List[ExperimentResult]] = {}
    for res in results:
        grouped.setdefault(res.method, []).append(res)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    # Collect all data points for analysis
    all_points = []

    # Plot each method
    for idx, (method, items) in enumerate(grouped.items()):
        color = color_palette.get(method, fallback_cycle[idx % len(fallback_cycle)])

        accuracies = []
        latencies = []
        n_values = []

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

            accuracies.append(acc)
            latencies.append(latency)
            n_values.append(_numeric_to_n(_parse_n_to_numeric(item.n)))
            all_points.append(
                (latency, acc, method, _numeric_to_n(_parse_n_to_numeric(item.n)))
            )

        if not accuracies:  # Skip if no valid data
            continue

        # Plot points for each n value
        for n in set(n_values):
            n_mask = [nv == n for nv in n_values]
            n_accs = [acc for acc, mask in zip(accuracies, n_mask) if mask]
            n_lats = [lat for lat, mask in zip(latencies, n_mask) if mask]

            if n_accs:
                marker = marker_map.get(n, "D")

                ax.scatter(
                    n_lats,
                    n_accs,
                    color=color,
                    marker=marker,
                    s=80,
                    alpha=0.8,
                    label=f"{method} (n={n})",
                    edgecolors="white",
                    linewidth=1,
                )

    # Customize axes
    ax.set_xlabel("Average Latency per Problem (s)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)

    # Add grid
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    # Set x-axis to log scale if there's a wide range
    latency_values = [point[0] for point in all_points]
    if latency_values and max(latency_values) / min(latency_values) > 10:
        ax.set_xscale("log")
        ax.set_xlabel("Average Latency per Problem (s) [log scale]", fontsize=12)

    # Create legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True, fontsize=9)

    plt.title(
        "Accuracy vs Computational Cost Trade-off",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
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
            n="4",
            accuracies=[0.854, 0.856, 0.84],
            runtimes=[
                timedelta(hours=1, minutes=25, seconds=10),
                timedelta(hours=1, minutes=20, seconds=59),
                timedelta(hours=1, minutes=24, seconds=36),
            ],
        ),
        ExperimentResult(
            method="WBoN",
            n="8",
            accuracies=[0.874, 0.828, 0.866],
            runtimes=[
                timedelta(hours=1, minutes=40, seconds=10),
                timedelta(hours=1, minutes=39, seconds=19),
                timedelta(hours=1, minutes=38, seconds=34),
            ],
        ),
        ExperimentResult(
            method="WBoN",
            n="16",
            accuracies=[0.846, 0.85, 0.868, 0.862],
            runtimes=[
                timedelta(minutes=41, seconds=58),
                timedelta(minutes=42, seconds=40),
                timedelta(minutes=41, seconds=37),
            ],
        ),
        ExperimentResult(
            method="DVTS",
            n="4",
            accuracies=[0.832, 0.834, 0.828],
            runtimes=[
                timedelta(hours=1, minutes=27, seconds=58),
                timedelta(hours=1, minutes=28, seconds=15),
                timedelta(hours=1, minutes=28, seconds=17),
            ],
        ),
        ExperimentResult(
            method="DVTS",
            n="8",
            accuracies=[0.82, 0.84, 0.84],
            runtimes=[
                timedelta(minutes=34, seconds=50),
                timedelta(minutes=37, seconds=57),
                timedelta(minutes=34, seconds=56),
            ],
        ),
        ExperimentResult(
            method="DVTS",
            n="16",
            accuracies=[0.834, 0.834, 0.836],
            runtimes=[
                timedelta(minutes=38, seconds=55),
                timedelta(minutes=39, seconds=15),
                timedelta(minutes=43, seconds=10),
            ],
        ),
        ExperimentResult(
            method="Beam Search",
            n="4",
            accuracies=[0.826, 0.831, 0.833],
            runtimes=[
                timedelta(hours=3, minutes=54),
                timedelta(hours=4, minutes=2),
                timedelta(hours=3, minutes=58),
            ],
        ),
        ExperimentResult(
            method="Beam Search",
            n="8",
            accuracies=[0.801, 0.81, 0.826],
            runtimes=[
                timedelta(hours=10, minutes=47),
                timedelta(hours=11, minutes=18),
                timedelta(hours=11, minutes=3),
            ],
        ),
        ExperimentResult(
            method="Beam Search",
            n="16",
            accuracies=[0.872, 0.878, 0.864],
            runtimes=[
                timedelta(hours=15, minutes=41),
                timedelta(hours=16, minutes=7),
                timedelta(hours=15, minutes=41),
            ],
        ),
        ExperimentResult(
            method="CGAI",
            n="4→8",
            accuracies=[87.40, 87.20, 87.60],
            runtimes=[
                timedelta(hours=1, minutes=25, seconds=10)
                + timedelta(minutes=26, seconds=8),
                timedelta(hours=1, minutes=24, seconds=36)
                + timedelta(minutes=25, seconds=54),
                timedelta(hours=1, minutes=24, seconds=24)
                + timedelta(minutes=26, seconds=15),
            ],
            notes="Different scale",
        ),
    ]
    display_hyperparameter_scaling_table(RESULTS)
    plot_accuracy_vs_latency(RESULTS)
    # plot_hyperparameter_scaling(RESULTS)

    # WITH BATCHING (sort of mixed and all over the place)
    #     RESULTS = [
    #     ExperimentResult(
    #         method="WBoN",
    #         n="4",
    #         accuracies=[0.854, 0.856, 0.84],
    #         runtimes=[
    #             timedelta(minutes=9, seconds=47),
    #             timedelta(minutes=11, seconds=5),
    #             timedelta(minutes=9, seconds=41),
    #         ],
    #     ),
    #     ExperimentResult(
    #         method="WBoN",
    #         n="8",
    #         accuracies=[0.874, 0.828, 0.866],
    #         runtimes=[
    #             timedelta(minutes=16, seconds=15),
    #             timedelta(minutes=18, seconds=33),
    #             timedelta(minutes=20, seconds=13),
    #         ],
    #     ),
    #     ExperimentResult(
    #         method="WBoN",
    #         n="16",
    #         accuracies=[0.846, 0.85, 0.868, 0.862],
    #         runtimes=[
    #             timedelta(minutes=41, seconds=58),
    #             timedelta(minutes=42, seconds=40),
    #             timedelta(minutes=41, seconds=37),
    #             timedelta(minutes=41, seconds=58),
    #         ],
    #     ),
    #     ExperimentResult(
    #         method="DVTS",
    #         n="4",
    #         accuracies=[0.832, 0.834, 0.828],
    #         runtimes=[
    #             timedelta(minutes=34, seconds=51),
    #             timedelta(minutes=35, seconds=4),
    #             timedelta(minutes=34, seconds=54),
    #         ],
    #     ),
    #     ExperimentResult(
    #         method="DVTS",
    #         n="8",
    #         accuracies=[0.82, 0.84, 0.84],
    #         runtimes=[
    #             timedelta(minutes=34, seconds=50),
    #             timedelta(minutes=37, seconds=57),
    #             timedelta(minutes=34, seconds=56),
    #         ],
    #     ),
    #     ExperimentResult(
    #         method="DVTS",
    #         n="16",
    #         accuracies=[0.834, 0.834, 0.836],
    #         runtimes=[
    #             timedelta(minutes=38, seconds=55),
    #             timedelta(minutes=39, seconds=15),
    #             timedelta(minutes=43, seconds=10),
    #         ],
    #     ),
    #     ExperimentResult(
    #         method="Beam Search",
    #         n="4",
    #         accuracies=[0.826, 0.831, 0.833],
    #         runtimes=[
    #             timedelta(hours=1, minutes=45),
    #             timedelta(hours=1, minutes=43),
    #             timedelta(hours=1, minutes=46),
    #         ],
    #     ),
    #     ExperimentResult(
    #         method="Beam Search",
    #         n="8",
    #         accuracies=[0.801, 0.81, 0.826],
    #         runtimes=[
    #             timedelta(hours=10, minutes=47),
    #             timedelta(hours=11, minutes=18),
    #             timedelta(hours=11, minutes=3),
    #         ],
    #     ),
    #     ExperimentResult(
    #         method="Beam Search",
    #         n="16",
    #         accuracies=[0.872, 0.878, 0.864],
    #         runtimes=[
    #             timedelta(hours=15, minutes=41),
    #             timedelta(hours=16, minutes=7),
    #             timedelta(hours=15, minutes=41),
    #         ],
    #     ),
    #     ExperimentResult(
    #         method="CGAI",
    #         n="4→8",
    #         accuracies=[87.40, 87.20, 87.60],
    #         runtimes=[
    #             timedelta(minutes=9, seconds=47) + timedelta(minutes=3, seconds=4),
    #             timedelta(minutes=9, seconds=41) + timedelta(minutes=3, seconds=3),
    #             timedelta(minutes=9, seconds=32) + timedelta(minutes=3, seconds=5),
    #         ],
    #         notes="Different scale",
    #     ),
    # ]
