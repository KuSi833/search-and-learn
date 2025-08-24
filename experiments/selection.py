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
from sal.utils.constants import BENCHMARK_SUBSETS_ROOT, Benchmark, Benchmarks
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    _select_uncertain_indices,
    _summarise_uncertainty,
    load_jsonl,
)

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


def export_uncertain_subset(
    run_id: str,
    coverage: float,
    metric: str,
    benchmark: Benchmark,
) -> None:
    """
    Export uncertain subset for a single run.

    Args:
        run_id: W&B run id
        coverage: Percentage coverage to export (e.g. 20 for top 20%)
        metric: Uncertainty metric to rank by
        benchmark_key: Benchmark key (default: "math500")
        name: Optional custom name for output file
    """
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        return

    selected = _select_uncertain_indices(records, coverage, metric)
    unique_ids = [records[i]["unique_id"] for i in selected]

    output_root = BENCHMARK_SUBSETS_ROOT / benchmark.hf_name / run_id / "coverage"

    coverage_str = (
        str(int(coverage)) if abs(coverage - round(coverage)) < 1e-9 else str(coverage)
    )
    output_file = output_root / f"{coverage_str}.json"

    output_root.mkdir(parents=True, exist_ok=True)

    # Structured payload with metadata for later mapping and reproducibility
    subset_payload = {
        "version": 1,
        "type": "uncertain_subset",
        "benchmark_key": benchmark.key,
        "hf_name": benchmark.hf_name,
        "run_id": run_id,
        "metric": metric,
        "coverage_pct": coverage,
        "unique_ids": sorted(unique_ids),
    }

    with open(output_file, "w") as f:
        json.dump(subset_payload, f, indent=2)

    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("Metric", metric)
    header.add_row("Coverage", f"{coverage:.1f}%")
    header.add_row("Exported", str(len(unique_ids)))
    header.add_row("Saved to", str(output_file))
    console.print(Panel(header, title="Export uncertain subset", box=box.ROUNDED))


def export_uncertain_multi_runs(
    run_ids: List[str],
    coverage: float,
    metric: str = "agreement_ratio",
    benchmark: Benchmark = Benchmarks.MATH500.value,
) -> None:
    """
    Export uncertain subsets for multiple runs.

    Args:
        run_ids: List of W&B run ids
        coverage: Percentage coverage to export for each run
        metric: Uncertainty metric to rank by
        benchmark_key: Benchmark key (default: "math500")
        name_prefix: Optional prefix for output file names
    """
    check_runs_availability(run_ids)

    console.print(
        Text(
            f"Exporting uncertain subsets for {len(run_ids)} runs...", style="bold blue"
        )
    )
    console.print()

    for i, run_id in enumerate(run_ids):
        if i > 0:
            console.print()

        export_uncertain_subset(run_id, coverage, metric, benchmark)


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

    # Uncertainty analysis
    analyze_uncertainty_with_check(run_ids)

    # Export uncertain subsets (uncomment to use)
    # for coverage in [10, 20]:
    #     export_uncertain_multi_runs(
    #         run_ids=run_ids,
    #         coverage=coverage,  # Export top 20% most uncertain
    #         metric="agreement_ratio",  # Uncertainty metric
    #     )
