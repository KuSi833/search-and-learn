#!/usr/bin/env python3
"""
Minimal threshold evaluator.

You specify thresholds and directions for the three metrics, and this script
reports, for each run_id:
- TP (incorrect selected), FP (correct selected), FN, TN
- Precision, Recall, F1
- Total selected and percentage

Selection rule defaults to 'union' (selected if any metric passes its threshold),
but can be changed via --mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table

from sal.utils.constants import BENCHMARK_SUBSETS_ROOT, Benchmarks
from sal.utils.runs import fusion_base_runs_best
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    load_jsonl,
)

METRICS = ["consensus_support", "agreement_ratio", "entropy_freq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed thresholds per run")
    # Thresholds and directions
    parser.add_argument(
        "--cs-dir",
        choices=["<=", ">="],
        default="<=",
        help="Direction for consensus_support",
    )
    parser.add_argument(
        "--cs-thresh", type=float, required=True, help="Threshold for consensus_support"
    )

    parser.add_argument(
        "--ar-dir",
        choices=["<=", ">="],
        default="<=",
        help="Direction for agreement_ratio",
    )
    parser.add_argument(
        "--ar-thresh", type=float, required=True, help="Threshold for agreement_ratio"
    )

    parser.add_argument(
        "--ef-dir",
        choices=["<=", ">="],
        default=">=",
        help="Direction for entropy_freq",
    )
    parser.add_argument(
        "--ef-thresh", type=float, required=True, help="Threshold for entropy_freq"
    )

    parser.add_argument(
        "--mode",
        choices=["union", "intersection", "cs", "ar", "ef"],
        default="union",
        help="Selection mode: union (any), intersection (all), or single metric",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export selected questions as dataset subsets (creates JSON files)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["math500", "aime24"],
        default="math500",
        help="Benchmark to use for dataset export",
    )

    return parser.parse_args()


def is_selected_by_threshold(value: float, direction: str, threshold: float) -> bool:
    if direction == "<=":
        return value <= threshold
    return value >= threshold


def evaluate_run(
    run_id: str,
    thresholds: Dict[str, Tuple[str, float]],
    mode: str,
    export_selected: bool = False,
) -> Dict[str, float]:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)

    total = len(records)
    total_correct = 0
    total_incorrect = 0

    tp = fp = fn = tn = 0
    selected = 0
    selected_unique_ids = []

    for idx, rec in enumerate(records):
        qa = _get_question_answer_from_record(rec)
        metrics = _compute_uncertainty_metrics(rec)

        if qa.is_correct:
            total_correct += 1
        else:
            total_incorrect += 1

        # per-metric boolean
        cs_selected = is_selected_by_threshold(
            float(metrics.get("consensus_support", 0.0)),
            *thresholds["consensus_support"],
        )
        ar_selected = is_selected_by_threshold(
            float(metrics.get("agreement_ratio", 0.0)), *thresholds["agreement_ratio"]
        )
        ef_selected = is_selected_by_threshold(
            float(metrics.get("entropy_freq", 0.0)), *thresholds["entropy_freq"]
        )

        if mode == "cs":
            is_sel = cs_selected
        elif mode == "ar":
            is_sel = ar_selected
        elif mode == "ef":
            is_sel = ef_selected
        elif mode == "intersection":
            is_sel = cs_selected and ar_selected and ef_selected
        else:  # union
            is_sel = cs_selected or ar_selected or ef_selected

        if is_sel:
            selected += 1
            if export_selected:
                selected_unique_ids.append(rec.get("unique_id", f"unknown_{idx}"))
            if qa.is_correct:
                fp += 1
            else:
                tp += 1
        else:
            if qa.is_correct:
                tn += 1
            else:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "run_id": run_id,
        "total": total,
        "selected": selected,
        "total_correct": total_correct,
        "total_incorrect": total_incorrect,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "selected_unique_ids": selected_unique_ids if export_selected else [],
    }


def export_threshold_dataset(
    run_ids: List[str],
    thresholds: Dict[str, Tuple[str, float]],
    mode: str,
    benchmark_key: str,
) -> None:
    """Export selected questions as dataset subsets following repo conventions."""
    console = Console()
    benchmark = Benchmarks.from_key(benchmark_key)

    # Create a descriptive name for the threshold combination
    thresh_desc = "_".join(
        [
            f"{metric}_{direction.replace('<=', 'le').replace('>=', 'ge')}_{threshold}"
            for metric, (direction, threshold) in thresholds.items()
        ]
    )
    dataset_name = f"threshold_{mode}_{thresh_desc}"

    all_selected_ids = set()

    for run_id in run_ids:
        result = evaluate_run(run_id, thresholds, mode, export_selected=True)
        run_selected_ids = result["selected_unique_ids"]
        all_selected_ids.update(run_selected_ids)

        # Export per-run subset (following existing pattern)
        output_dir = (
            BENCHMARK_SUBSETS_ROOT / benchmark.hf_name / run_id / "threshold_subsets"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset_name}.json"

        payload = {
            "version": 1,
            "type": "threshold_subset",
            "benchmark_key": benchmark.key,
            "hf_name": benchmark.hf_name,
            "run_id": run_id,
            "selection_mode": mode,
            "thresholds": {
                metric: {"direction": direction, "threshold": threshold}
                for metric, (direction, threshold) in thresholds.items()
            },
            "metrics": {
                "selected": result["selected"],
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
                "tn": result["tn"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            },
            "unique_ids": sorted(run_selected_ids),
        }

        with open(output_file, "w") as f:
            json.dump(payload, f, indent=2)

        console.print(
            f"Exported {len(run_selected_ids)} questions for run {run_id} to {output_file}"
        )

    # Export aggregated dataset across all runs
    agg_output_dir = (
        BENCHMARK_SUBSETS_ROOT / benchmark.hf_name / "aggregated" / "threshold_subsets"
    )
    agg_output_dir.mkdir(parents=True, exist_ok=True)
    agg_output_file = agg_output_dir / f"{dataset_name}_all_runs.json"

    agg_payload = {
        "version": 1,
        "type": "threshold_subset_aggregated",
        "benchmark_key": benchmark.key,
        "hf_name": benchmark.hf_name,
        "run_ids": run_ids,
        "selection_mode": mode,
        "thresholds": {
            metric: {"direction": direction, "threshold": threshold}
            for metric, (direction, threshold) in thresholds.items()
        },
        "unique_ids": sorted(list(all_selected_ids)),
    }

    with open(agg_output_file, "w") as f:
        json.dump(agg_payload, f, indent=2)

    console.print(f"\n[bold green]Aggregated dataset exported:[/bold green]")
    console.print(f"File: {agg_output_file}")
    console.print(f"Total unique questions: {len(all_selected_ids)}")
    console.print(f"Selection mode: {mode}")
    console.print(f"Thresholds: {thresholds}")


def main():
    args = parse_args()
    console = Console()

    thresholds = {
        "consensus_support": (args.cs_dir, args.cs_thresh),
        "agreement_ratio": (args.ar_dir, args.ar_thresh),
        "entropy_freq": (args.ef_dir, args.ef_thresh),
    }

    run_ids = fusion_base_runs_best()

    table = Table(
        title="Threshold Evaluation per Run",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Run ID", style="cyan")
    table.add_column("Selected", justify="right", style="blue")
    table.add_column("% Sel", justify="right", style="dim blue")
    table.add_column("T", justify="right", style="green")
    table.add_column("F", justify="right", style="red")
    table.add_column("TP", justify="right", style="green")
    table.add_column("FP", justify="right", style="red")
    table.add_column("FN", justify="right", style="green")
    table.add_column("TN", justify="right", style="red")
    table.add_column("Precision", justify="right", style="bright_blue")
    table.add_column("Recall", justify="right", style="bright_green")
    table.add_column("F1", justify="right", style="bright_magenta")

    agg_tp = agg_fp = agg_fn = agg_tn = agg_sel = agg_total = 0
    agg_correct = agg_incorrect = 0

    for run_id in run_ids:
        res = evaluate_run(run_id, thresholds, args.mode, export_selected=args.export)
        sel_pct = (res["selected"] / res["total"]) * 100 if res["total"] > 0 else 0.0
        table.add_row(
            run_id,
            str(res["selected"]),
            f"{sel_pct:.1f}%",
            str(res["total_correct"]),
            str(res["total_incorrect"]),
            str(res["tp"]),
            str(res["fp"]),
            str(res["fn"]),
            str(res["tn"]),
            f"{res['precision']:.3f}",
            f"{res['recall']:.3f}",
            f"{res['f1']:.3f}",
        )

        agg_tp += res["tp"]
        agg_fp += res["fp"]
        agg_fn += res["fn"]
        agg_tn += res["tn"]
        agg_sel += res["selected"]
        agg_total += res["total"]
        agg_correct += res["total_correct"]
        agg_incorrect += res["total_incorrect"]

    console.print(table)

    # Aggregate summary
    precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
    recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    sel_pct = (agg_sel / agg_total) * 100 if agg_total > 0 else 0.0

    console.print()
    console.print("[bold]Aggregate across runs:[/bold]")
    console.print(
        f"Selected: {agg_sel}/{agg_total} ({sel_pct:.1f}%), T: {agg_correct}, F: {agg_incorrect}, TP: {agg_tp}, FP: {agg_fp}, FN: {agg_fn}, TN: {agg_tn}"
    )
    console.print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # Export datasets if requested
    if args.export:
        export_threshold_dataset(run_ids, thresholds, args.mode, args.benchmark)


if __name__ == "__main__":
    main()
