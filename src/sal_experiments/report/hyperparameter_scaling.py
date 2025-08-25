from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .util import (
    format_mean_std,
    format_runtime,
    runtime_stats,
)


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
    table.add_column("Notes", style="dim")

    for result in results:
        formatted_acc = format_accuracy_stats(result)
        num_runs = len(result.accuracies)
        acc_style = get_accuracy_style(result, formatted_acc)

        # Format runtime if available
        runtime_str = format_runtime_stats(result) or "—"

        table.add_row(
            result.method,
            result.n,
            acc_style,
            str(num_runs),
            runtime_str,
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


def hyperparameter_scaling_report():
    """Generate the main hyperparameter scaling report."""
    RESULTS = [
        ExperimentResult(
            method="WBoN",
            n="4",
            accuracies=[0.854, 0.856, 0.84],
            runtimes=[
                timedelta(minutes=9, seconds=47),
                timedelta(minutes=11, seconds=5),
                timedelta(minutes=9, seconds=41),
            ],
        ),
        ExperimentResult(
            method="WBoN",
            n="8",
            accuracies=[0.87, 0.828],
            runtimes=[
                timedelta(minutes=16, seconds=15),
                timedelta(minutes=14, seconds=33),
            ],
        ),
        ExperimentResult(
            method="WBoN",
            n="16",
            accuracies=[0.846],
            runtimes=[
                timedelta(minutes=27, seconds=49),
            ],
        ),
        ExperimentResult(
            method="DVTS",
            n="4",
            accuracies=[0.832, 0.834, 0.828],
            runtimes=[
                timedelta(minutes=34, seconds=51),
                timedelta(minutes=35, seconds=4),
                timedelta(minutes=34, seconds=54),
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
        ),
        ExperimentResult(
            method="Beam Search",
            n="16",
            accuracies=[0.826, 0.831, 0.833],
            runtimes=[
                timedelta(hours=2, minutes=21),
            ],
        ),
        ExperimentResult(
            method="CGAI",
            n="4→8",
            accuracies=[87.40, 87.20, 87.60],
            runtimes=[
                timedelta(minutes=9, seconds=47) + timedelta(minutes=3, seconds=4),
                timedelta(minutes=9, seconds=41) + timedelta(minutes=3, seconds=3),
                timedelta(minutes=9, seconds=32) + timedelta(minutes=3, seconds=5),
            ],
            notes="Different scale",
        ),
    ]
    display_hyperparameter_scaling_table(RESULTS)
