#!/usr/bin/env python3
"""
Selection experiments: uncertainty analysis across multiple runs to assess robustness.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Import functions from the inference visualizer
sys.path.append(str(Path(__file__).parent.parent))
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    _select_uncertain_indices,
    _summarise_uncertainty,
    load_jsonl,
)
from sal.utils.constants import BENCHMARK_SUBSETS_ROOT, Benchmarks

console = Console()


def check_runs_availability(run_ids: List[str]) -> None:
    """Check if all run paths exist, crash if any are missing."""
    not_available = set()
    for run_id in run_ids:
        out_file = Path("./output") / run_id / "inference_output.jsonl"
        if not out_file.exists():
            console.print(Text(f"Missing: {out_file}", style="red"))
            not_available.add(run_id)
    if len(not_available) > 0:
        exit()


def analyze_uncertainty_multi_runs(run_ids: List[str]) -> None:
    """
    Run uncertainty analysis for multiple run IDs, showing results for each run separately.

    Args:
        run_ids: List of W&B run ids (directories under ./output)
    """
    if not run_ids:
        console.print(Text("No run IDs provided", style="red"))
        return

    console.print(
        Text(
            f"Running uncertainty analysis for {len(run_ids)} runs...",
            style="bold blue",
        )
    )
    console.print()

    for i, run_id in enumerate(run_ids):
        if i > 0:
            console.print("\n" + "=" * 80 + "\n")

        # Run the same analysis as cmd_uncertainty for each run
        out_file = Path("./output") / run_id / "inference_output.jsonl"

        try:
            records = load_jsonl(out_file)
            if not records:
                console.print(Text(f"No records found in {out_file}", style="red"))
                continue
        except FileNotFoundError:
            console.print(Text(f"File not found: {out_file}", style="red"))
            continue

        metrics_list: List[Dict[str, Any]] = []
        labels: List[bool] = []

        for rec in records:
            qa = _get_question_answer_from_record(rec)
            labels.append(bool(qa.is_correct))
            metrics_list.append(_compute_uncertainty_metrics(rec))

        total = len(labels)
        acc = (100.0 * sum(1 for y in labels if y) / total) if total > 0 else 0.0
        header = Table.grid(padding=(0, 1))
        header.add_column(style="bold cyan")
        header.add_column()
        header.add_row("Run", run_id)
        header.add_row("File", f"output/{run_id}/inference_output.jsonl")
        header.add_row("Samples", str(total))
        header.add_row("Accuracy (assumed pred)", f"{acc:.1f}%")
        console.print(
            Panel(
                header,
                title=f"Uncertainty analysis ({i + 1}/{len(run_ids)})",
                box=box.ROUNDED,
            )
        )

        _summarise_uncertainty(metrics_list, labels)


def analyze_uncertainty_with_check(run_ids: List[str]) -> None:
    """Check all runs exist, then run uncertainty analysis."""
    check_runs_availability(run_ids)
    analyze_uncertainty_multi_runs(run_ids)


def fusion_base_runs():
    # BoN n=4

    return [
        "gfw8x07r",  # 78
        "77pyab58",  # 80
        "tqfyvf5w",  # 82
    ]


def fusion_base_runs_best():
    return [
        "gfw8x07r",  # 78
        "77pyab58",  # 80
        "tqfyvf5w",  # 82
    ]


if __name__ == "__main__":
    # Example usage - specify your run IDs here
    run_ids = fusion_base_runs_best()
    # Uncomment and modify the run_ids list above, then run:
    # analyze_uncertainty_with_check(run_ids)  # Recommended: checks availability first
    #
    # Or use the direct version if you're sure all runs are available:
    analyze_uncertainty_with_check(run_ids)
