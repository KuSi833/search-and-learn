#!/usr/bin/env python3
"""
Selection experiments: uncertainty analysis across multiple runs to assess robustness.
Updated to include comprehensive evaluation of all uncertainty signals and ensemble approaches.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sal.utils.runs import fusion_base_runs_best

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

# Core uncertainty signals to evaluate
CORE_UNCERTAINTY_SIGNALS = [
    "agreement_ratio",
    "entropy_freq",
    "consensus_support",
]

# Extended signals for comprehensive analysis
EXTENDED_UNCERTAINTY_SIGNALS = CORE_UNCERTAINTY_SIGNALS + [
    "prm_margin",
    "prm_std",
    "entropy_weighted",
]


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


def _compute_ensemble_uncertainty_disjunctive(
    metrics_list: List[Dict[str, Any]],
    signals: List[str],
    coverage_pct: float,
    labels: List[bool],
) -> Set[int]:
    """
    Compute ensemble uncertainty selection using disjunctive strategy.
    An item is selected if it's uncertain according to ANY of the signals.

    Args:
        metrics_list: List of uncertainty metrics per sample
        signals: List of uncertainty signal names to combine
        coverage_pct: Target coverage percentage
        labels: Ground truth labels for direction inference

    Returns:
        Set of indices selected by ensemble
    """
    n_total = len(metrics_list)
    target_k = max(1, int(round(n_total * (coverage_pct / 100.0))))

    # Get uncertain indices for each signal
    signal_selections: List[Set[int]] = []
    for signal in signals:
        vals = [m.get(signal, 0.0) for m in metrics_list]
        # Determine uncertainty direction
        corr_vals = [v for v, y in zip(vals, labels) if y]
        inc_vals = [v for v, y in zip(vals, labels) if not y]
        mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
        mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
        low_is_uncertain = mean_corr > mean_inc

        # Sort by uncertainty (most uncertain first)
        idxs = list(range(n_total))
        idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)

        # Take top uncertain for this signal
        signal_k = max(1, int(round(n_total * (coverage_pct / 100.0))))
        signal_selections.append(set(idxs[:signal_k]))

    # Disjunctive combination (union)
    ensemble_selection = set()
    for selection in signal_selections:
        ensemble_selection.update(selection)

    # If ensemble is too large, rank by average uncertainty across signals
    if len(ensemble_selection) > target_k:
        # Compute average rank across signals for items in ensemble
        avg_ranks = {}
        for idx in ensemble_selection:
            ranks = []
            for signal in signals:
                vals = [m.get(signal, 0.0) for m in metrics_list]
                corr_vals = [v for v, y in zip(vals, labels) if y]
                inc_vals = [v for v, y in zip(vals, labels) if not y]
                mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
                mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
                low_is_uncertain = mean_corr > mean_inc

                idxs = list(range(n_total))
                idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)
                rank = idxs.index(idx)
                ranks.append(rank)
            avg_ranks[idx] = sum(ranks) / len(ranks)

        # Take top target_k by average rank
        sorted_ensemble = sorted(ensemble_selection, key=lambda i: avg_ranks[i])
        ensemble_selection = set(sorted_ensemble[:target_k])

    return ensemble_selection


def analyze_uncertainty_multi_runs(run_ids: List[str]) -> None:
    """
    Run comprehensive uncertainty analysis aggregated across multiple runs.
    Shows statistical robustness with mean ± std across runs.

    Args:
        run_ids: List of W&B run ids (directories under ./output)
    """
    if not run_ids:
        console.print(Text("No run IDs provided", style="red"))
        return

    console.print(
        Text(
            f"Running aggregated uncertainty analysis across {len(run_ids)} runs...",
            style="bold blue",
        )
    )
    console.print()

    # Collect data from all runs for aggregated analysis
    all_run_data: List[Tuple[str, List[Dict[str, Any]], List[bool]]] = []
    total_samples = 0
    total_accuracy_sum = 0.0

    for run_id in run_ids:
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

        all_run_data.append((run_id, metrics_list, labels))

        # Accumulate stats
        total_samples += len(labels)
        if len(labels) > 0:
            acc = 100.0 * sum(1 for y in labels if y) / len(labels)
            total_accuracy_sum += acc

    # Summary header
    avg_accuracy = total_accuracy_sum / len(all_run_data) if all_run_data else 0.0
    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Runs analyzed", str(len(all_run_data)))
    header.add_row("Run IDs", ", ".join(run_ids))
    header.add_row("Total samples", str(total_samples))
    header.add_row("Average accuracy", f"{avg_accuracy:.1f}%")
    console.print(
        Panel(
            header,
            title="Multi-run aggregated analysis",
            box=box.ROUNDED,
        )
    )

    # Multi-run statistical analysis
    analyze_multi_run_statistics(all_run_data)

    # Ensemble selection analysis
    console.print("\n" + "=" * 40 + "\n")
    analyze_ensemble_selection(all_run_data)


def analyze_multi_run_statistics(
    all_run_data: List[Tuple[str, List[Dict[str, Any]], List[bool]]],
) -> None:
    """
    Analyze statistical robustness across multiple runs.

    Args:
        all_run_data: List of (run_id, metrics_list, labels) tuples
    """
    console.print(
        Panel(
            Text("Multi-run statistical robustness analysis", style="bold"),
            box=box.ROUNDED,
        )
    )

    # Coverage levels to analyze
    coverage_levels = [10, 20, 30, 40, 50]

    # Collect precision/recall stats across runs for each signal
    signal_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for run_id, metrics_list, labels in all_run_data:
        total_incorrect = sum(1 for y in labels if not y)
        total = len(labels)

        if total == 0 or total_incorrect == 0:
            continue

        for signal in CORE_UNCERTAINTY_SIGNALS:
            vals = [m.get(signal, 0.0) for m in metrics_list]

            # Determine uncertainty direction for this run
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]
            mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
            mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
            low_is_uncertain = mean_corr > mean_inc

            # Sort by uncertainty
            idxs = list(range(total))
            idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)

            for coverage in coverage_levels:
                k = max(1, int(round(total * (coverage / 100.0))))
                flagged = set(idxs[:k])

                # Precision: fraction of flagged that are incorrect
                flagged_incorrect = sum(1 for i in flagged if not labels[i])
                precision = flagged_incorrect / len(flagged) if flagged else 0.0

                # Recall: fraction of incorrect that are flagged
                recall = (
                    flagged_incorrect / total_incorrect if total_incorrect > 0 else 0.0
                )

                signal_stats[signal][f"precision_{coverage}"].append(precision * 100)
                signal_stats[signal][f"recall_{coverage}"].append(recall * 100)

    # Display statistics table
    stats_table = Table(title="Multi-run statistics (mean ± std)", box=box.SIMPLE_HEAVY)
    stats_table.add_column("Signal", style="bold")
    stats_table.add_column("Metric", style="bold")
    for coverage in coverage_levels:
        stats_table.add_column(f"{coverage}% cov", justify="right")

    for signal in CORE_UNCERTAINTY_SIGNALS:
        # Precision row
        precision_row = [signal, "Precision"]
        for coverage in coverage_levels:
            values = signal_stats[signal][f"precision_{coverage}"]
            if values:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                precision_row.append(f"{mean_val:.1f}±{std_val:.1f}")
            else:
                precision_row.append("-")
        stats_table.add_row(*precision_row)

        # Recall row
        recall_row = ["", "Recall"]
        for coverage in coverage_levels:
            values = signal_stats[signal][f"recall_{coverage}"]
            if values:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                recall_row.append(f"{mean_val:.1f}±{std_val:.1f}")
            else:
                recall_row.append("-")
        stats_table.add_row(*recall_row)

        # Empty row for spacing
        if signal != CORE_UNCERTAINTY_SIGNALS[-1]:
            stats_table.add_row("", "", *[""] * len(coverage_levels))

    console.print(stats_table)


def analyze_ensemble_selection(
    all_run_data: List[Tuple[str, List[Dict[str, Any]], List[bool]]],
) -> None:
    """
    Analyze ensemble selection strategies across runs.

    Args:
        all_run_data: List of (run_id, metrics_list, labels) tuples
    """
    console.print(
        Panel(Text("Ensemble selection analysis", style="bold"), box=box.ROUNDED)
    )

    coverage_levels = [20, 30, 40]  # Focus on key coverage levels

    # Define ensemble combinations to test
    ensemble_combinations = [
        (["agreement_ratio"], "Agreement only"),
        (["entropy_freq"], "Entropy only"),
        (["consensus_support"], "Group score only"),
        (["agreement_ratio", "entropy_freq"], "Agreement + Entropy"),
        (["agreement_ratio", "consensus_support"], "Agreement + Group score"),
        (["entropy_freq", "consensus_support"], "Entropy + Group score"),
        (["agreement_ratio", "entropy_freq", "consensus_support"], "All three signals"),
    ]

    # Results table
    ensemble_table = Table(
        title="Ensemble selection performance (mean recall %)", box=box.SIMPLE_HEAVY
    )
    ensemble_table.add_column("Strategy", style="bold")
    for coverage in coverage_levels:
        ensemble_table.add_column(f"{coverage}% cov", justify="right")
    ensemble_table.add_column("Avg", justify="right", style="bold")

    for signals, name in ensemble_combinations:
        row_data = [name]
        coverage_recalls = []

        for coverage in coverage_levels:
            recalls = []

            for run_id, metrics_list, labels in all_run_data:
                total_incorrect = sum(1 for y in labels if not y)
                if total_incorrect == 0:
                    continue

                if len(signals) == 1:
                    # Single signal - use existing logic
                    # Convert to our format for single signal
                    vals = [m.get(signals[0], 0.0) for m in metrics_list]
                    corr_vals = [v for v, y in zip(vals, labels) if y]
                    inc_vals = [v for v, y in zip(vals, labels) if not y]
                    mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
                    mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
                    low_is_uncertain = mean_corr > mean_inc

                    idxs = list(range(len(labels)))
                    idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)
                    k = max(1, int(round(len(labels) * (coverage / 100.0))))
                    selected_set = set(idxs[:k])
                else:
                    # Multi-signal ensemble
                    selected_set = _compute_ensemble_uncertainty_disjunctive(
                        metrics_list, signals, coverage, labels
                    )

                # Compute recall
                flagged_incorrect = sum(1 for i in selected_set if not labels[i])
                recall = (
                    flagged_incorrect / total_incorrect if total_incorrect > 0 else 0.0
                )
                recalls.append(recall * 100)

            if recalls:
                mean_recall = statistics.mean(recalls)
                coverage_recalls.append(mean_recall)
                row_data.append(f"{mean_recall:.1f}")
            else:
                row_data.append("-")

        # Average across coverage levels
        if coverage_recalls:
            avg_recall = statistics.mean(coverage_recalls)
            row_data.append(f"{avg_recall:.1f}")
        else:
            row_data.append("-")

        ensemble_table.add_row(*row_data)

    console.print(ensemble_table)


def analyze_uncertainty_with_check(run_ids: List[str]) -> None:
    """Check all runs exist, then run comprehensive uncertainty analysis."""
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
        benchmark: Benchmark instance
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


def export_ensemble_uncertain_subset(
    run_id: str,
    coverage: float,
    signals: List[str],
    benchmark: Benchmark,
    strategy_name: str = "ensemble",
) -> None:
    """
    Export uncertain subset using ensemble selection strategy.

    Args:
        run_id: W&B run id
        coverage: Percentage coverage to export
        signals: List of uncertainty signals to combine
        benchmark: Benchmark instance
        strategy_name: Name for the ensemble strategy
    """
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        return

    # Compute metrics and labels
    metrics_list: List[Dict[str, Any]] = []
    labels: List[bool] = []
    for rec in records:
        qa = _get_question_answer_from_record(rec)
        labels.append(bool(qa.is_correct))
        metrics_list.append(_compute_uncertainty_metrics(rec))

    # Get ensemble selection
    selected_indices = _compute_ensemble_uncertainty_disjunctive(
        metrics_list, signals, coverage, labels
    )
    unique_ids = [records[i]["unique_id"] for i in selected_indices]

    output_root = BENCHMARK_SUBSETS_ROOT / benchmark.hf_name / run_id / "ensemble"
    output_root.mkdir(parents=True, exist_ok=True)

    coverage_str = (
        str(int(coverage)) if abs(coverage - round(coverage)) < 1e-9 else str(coverage)
    )
    signals_str = "+".join(signals)
    output_file = output_root / f"{strategy_name}_{signals_str}_{coverage_str}.json"

    # Structured payload with ensemble metadata
    subset_payload = {
        "version": 1,
        "type": "ensemble_uncertain_subset",
        "benchmark_key": benchmark.key,
        "hf_name": benchmark.hf_name,
        "run_id": run_id,
        "ensemble_signals": signals,
        "strategy": "disjunctive",
        "strategy_name": strategy_name,
        "coverage_pct": coverage,
        "unique_ids": sorted(unique_ids),
    }

    with open(output_file, "w") as f:
        json.dump(subset_payload, f, indent=2)

    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("Ensemble", " + ".join(signals))
    header.add_row("Strategy", "Disjunctive")
    header.add_row("Coverage", f"{coverage:.1f}%")
    header.add_row("Exported", str(len(unique_ids)))
    header.add_row("Saved to", str(output_file))
    console.print(
        Panel(header, title="Export ensemble uncertain subset", box=box.ROUNDED)
    )


def export_uncertain_multi_runs(
    run_ids: List[str],
    coverage: float,
    metric: str = "agreement_ratio",
    benchmark: Benchmark = Benchmarks.MATH500.value,
) -> None:
    """
    Export uncertain subsets for multiple runs using single metrics.

    Args:
        run_ids: List of W&B run ids
        coverage: Percentage coverage to export for each run
        metric: Uncertainty metric to rank by
        benchmark: Benchmark instance
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


def export_ensemble_multi_runs(
    run_ids: List[str],
    coverage: float,
    signals: List[str],
    benchmark: Benchmark = Benchmarks.MATH500.value,
    strategy_name: str = "ensemble",
) -> None:
    """
    Export ensemble uncertain subsets for multiple runs.

    Args:
        run_ids: List of W&B run ids
        coverage: Percentage coverage to export for each run
        signals: List of uncertainty signals to combine
        benchmark: Benchmark instance
        strategy_name: Name for the ensemble strategy
    """
    check_runs_availability(run_ids)

    console.print(
        Text(
            f"Exporting ensemble uncertain subsets for {len(run_ids)} runs...",
            style="bold blue",
        )
    )
    console.print()

    for i, run_id in enumerate(run_ids):
        if i > 0:
            console.print()

        export_ensemble_uncertain_subset(
            run_id, coverage, signals, benchmark, strategy_name
        )


def generate_aggregated_figures(run_ids: List[str]) -> None:
    """
    Generate aggregated uncertainty analysis figures across all runs.
    Creates cumulative stacked plots averaged across all runs.

    Args:
        run_ids: List of W&B run ids
    """
    console.print(
        Text(
            f"Generating aggregated figures across {len(run_ids)} runs...",
            style="bold blue",
        )
    )

    try:
        # Collect data per run (keep runs separate)
        per_run_data = {}

        for run_id in run_ids:
            out_file = Path("./output") / run_id / "inference_output.jsonl"
            try:
                records = load_jsonl(out_file)
                if not records:
                    console.print(Text(f"No records found in {out_file}", style="red"))
                    continue
            except FileNotFoundError:
                console.print(Text(f"File not found: {out_file}", style="red"))
                continue

            metrics_list = []
            labels = []
            for rec in records:
                qa = _get_question_answer_from_record(rec)
                labels.append(bool(qa.is_correct))
                metrics_list.append(_compute_uncertainty_metrics(rec))

            per_run_data[run_id] = {"metrics": metrics_list, "labels": labels}

        if not per_run_data:
            console.print(Text("No data found across all runs", style="red"))
            return

        # Create aggregated output directory
        output_dir = Path("./figures/selection/aggregated")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate multi-run violin plots showing individual runs
        _save_multi_run_violin_plots(
            per_run_data,
            CORE_UNCERTAINTY_SIGNALS,
            output_dir,
            run_ids,
        )

        console.print(
            Text(
                f"✓ Aggregated figures generated across {len(run_ids)} runs",
                style="bold green",
            )
        )
        console.print(
            Text(
                f"Saved to: {output_dir}/",
                style="dim",
            )
        )

    except Exception as e:
        console.print(
            Text(f"Error generating aggregated figures: {str(e)}", style="red")
        )


def _save_aggregated_cumulative_stacked(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: List[str],
    outdir: Path,
    run_name: str,
    bins: int,
) -> None:
    """
    Save cumulative stacked plots aggregated across multiple runs.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        if not chosen_metrics:
            return

        plt.style.use("seaborn-v0_8")
        num_metrics = len(chosen_metrics)
        cols = 2 if num_metrics <= 6 else 3
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(6.5 * cols, 4.0 * rows), squeeze=False
        )

        n_total = len(labels)
        for idx, m in enumerate(chosen_metrics):
            r = idx // cols
            c = idx % cols
            ax = axes[r][c]
            vals = [float(mm.get(m, 0.0)) for mm in metrics_list]
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]

            if not corr_vals and not inc_vals:
                ax.axis("off")
                continue

            # Determine uncertainty direction
            mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
            mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
            low_is_uncertain = mean_corr > mean_inc

            all_vals = corr_vals + inc_vals
            vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0.0, 1.0)
            if vmin == vmax:
                vmin -= 0.5
                vmax += 0.5

            counts_inc, edges = np.histogram(inc_vals, bins=bins, range=(vmin, vmax))
            counts_cor, _ = np.histogram(corr_vals, bins=bins, range=(vmin, vmax))

            # Handle different uncertainty directions
            if low_is_uncertain:
                # Low values = uncertain, so cumulate from left (uncertain) to right (certain)
                cum_inc = np.cumsum(counts_inc) / max(1, n_total)
                cum_cor = np.cumsum(counts_cor) / max(1, n_total)
                x = edges[1:]  # right edges as x positions
                direction_note = "← uncertain | certain →"
            else:
                # High values = uncertain, so cumulate from right (uncertain) to left (certain)
                cum_inc = np.cumsum(counts_inc[::-1])[::-1] / max(1, n_total)
                cum_cor = np.cumsum(counts_cor[::-1])[::-1] / max(1, n_total)
                x = edges[1:]  # right edges as x positions
                direction_note = "← certain | uncertain →"

            ax.fill_between(
                x, 0, cum_inc, step="pre", color="#d62728", alpha=0.5, label="incorrect"
            )
            ax.fill_between(
                x,
                cum_inc,
                cum_inc + cum_cor,
                step="pre",
                color="#2ca02c",
                alpha=0.5,
                label="correct",
            )
            ax.set_ylim(0, 1)
            ax.set_xlim(vmin, vmax)
            ax.set_title(f"{m} (aggregated)\\n{direction_note}")
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.legend(frameon=False, fontsize=9)

        for j in range(num_metrics, rows * cols):
            r = j // cols
            c = j % cols
            axes[r][c].axis("off")

        fig.suptitle(f"Cumulative stacked (≤ x) by correctness - {run_name}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "aggregated_cumulative_stacked.png", dpi=200)
        plt.close(fig)

    except ImportError:
        console.print(Text("matplotlib not available for plotting", style="yellow"))
    except Exception as e:
        console.print(
            Text(f"Error creating cumulative stacked plot: {str(e)}", style="red")
        )


def _save_aggregated_summary(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: List[str],
    outdir: Path,
    run_ids: List[str],
) -> None:
    """
    Save aggregated summary statistics to JSON.
    """
    try:
        import json

        # Compute means by correctness
        means_summary = {}
        for metric in chosen_metrics:
            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]
            mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
            mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
            direction = "↑" if mean_corr > mean_inc else "↓"

            means_summary[metric] = {
                "mean_correct": float(mean_corr),
                "mean_incorrect": float(mean_inc),
                "direction": direction,
                "low_is_uncertain": mean_corr > mean_inc,
            }

        # Coverage analysis
        coverage_analysis = {}
        coverage_pcts = [10, 20, 30, 40, 50]
        total_incorrect = sum(1 for y in labels if not y)
        total = len(labels)

        for metric in chosen_metrics:
            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
            low_is_uncertain = means_summary[metric]["low_is_uncertain"]

            idxs = list(range(total))
            idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)

            coverage_results = {}
            for p in coverage_pcts:
                k = max(1, int(round(total * (p / 100.0))))
                flagged = set(idxs[:k])
                incorrect_flagged = sum(1 for i in flagged if not labels[i])
                recall_incorrect = (
                    100.0 * incorrect_flagged / total_incorrect
                    if total_incorrect > 0
                    else 0.0
                )
                precision = 100.0 * incorrect_flagged / len(flagged) if flagged else 0.0
                coverage_results[f"{p}%"] = {
                    "recall": float(recall_incorrect),
                    "precision": float(precision),
                    "flagged_count": len(flagged),
                    "incorrect_flagged": incorrect_flagged,
                }

            coverage_analysis[metric] = coverage_results

        summary = {
            "aggregated_analysis": True,
            "run_ids": run_ids,
            "total_samples": total,
            "total_correct": int(sum(1 for y in labels if y)),
            "total_incorrect": int(sum(1 for y in labels if not y)),
            "accuracy": float(100.0 * sum(1 for y in labels if y) / total)
            if total > 0
            else 0.0,
            "means_by_correctness": means_summary,
            "coverage_analysis": coverage_analysis,
        }

        with open(outdir / "aggregated_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    except Exception as e:
        console.print(Text(f"Error creating summary: {str(e)}", style="red"))


def _save_selection_precision_plots(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: List[str],
    outdir: Path,
    run_name: str,
) -> None:
    """
    Save selection-based precision plots that show the actual composition
    of selected uncertain samples (what the precision numbers represent).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        if not chosen_metrics:
            return

        plt.style.use("seaborn-v0_8")
        num_metrics = len(chosen_metrics)
        cols = 2 if num_metrics <= 4 else 3
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False
        )

        coverage_levels = [10, 20, 30, 40, 50]
        total = len(labels)
        total_incorrect = sum(1 for y in labels if not y)

        for idx, metric in enumerate(chosen_metrics):
            r = idx // cols
            c = idx % cols
            ax = axes[r][c]

            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]

            # Determine uncertainty direction
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]
            mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
            mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
            low_is_uncertain = mean_corr > mean_inc

            # Sort by uncertainty (most uncertain first)
            idxs = list(range(total))
            idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)

            # Calculate precision for each coverage level
            precisions = []
            coverage_pcts = []

            for coverage in coverage_levels:
                k = max(1, int(round(total * (coverage / 100.0))))
                flagged = set(idxs[:k])
                incorrect_flagged = sum(1 for i in flagged if not labels[i])
                precision = 100.0 * incorrect_flagged / len(flagged) if flagged else 0.0

                precisions.append(precision)
                coverage_pcts.append(coverage)

            # Plot precision vs coverage
            ax.plot(
                coverage_pcts,
                precisions,
                "o-",
                linewidth=2,
                markersize=6,
                color="#d62728",
            )
            ax.set_xlabel("Coverage (%)")
            ax.set_ylabel("Precision (% incorrect)")
            ax.set_title(f"{metric}\\nPrecision of Uncertain Selection")
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.set_ylim(0, 100)

            # Add precision values as text annotations
            for i, (cov, prec) in enumerate(zip(coverage_pcts, precisions)):
                ax.annotate(
                    f"{prec:.1f}%",
                    (cov, prec),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                )

        # Turn off unused subplots
        for j in range(num_metrics, rows * cols):
            r = j // cols
            c = j % cols
            axes[r][c].axis("off")

        fig.suptitle(f"Selection Precision Analysis - {run_name}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "selection_precision_analysis.png", dpi=200)
        plt.close(fig)

        # Also create a bar chart showing the composition of top 20% uncertain
        fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4))
        if num_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(chosen_metrics):
            ax = axes[idx]

            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]

            # Determine uncertainty direction
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]
            mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
            mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
            low_is_uncertain = mean_corr > mean_inc

            # Sort by uncertainty and take top 20%
            idxs = list(range(total))
            idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)
            k = max(1, int(round(total * 0.2)))  # Top 20%
            flagged = set(idxs[:k])

            incorrect_flagged = sum(1 for i in flagged if not labels[i])
            correct_flagged = len(flagged) - incorrect_flagged

            # Create stacked bar
            ax.bar(
                ["Top 20% Most\\nUncertain"],
                [incorrect_flagged],
                color="#d62728",
                alpha=0.7,
                label="Incorrect",
            )
            ax.bar(
                ["Top 20% Most\\nUncertain"],
                [correct_flagged],
                bottom=[incorrect_flagged],
                color="#2ca02c",
                alpha=0.7,
                label="Correct",
            )

            precision = 100.0 * incorrect_flagged / len(flagged)
            ax.set_title(f"{metric}\\nPrecision: {precision:.1f}%")
            ax.set_ylabel("Number of samples")
            ax.legend()

            # Add count annotations
            if incorrect_flagged > 0:
                ax.text(
                    0,
                    incorrect_flagged / 2,
                    str(incorrect_flagged),
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                )
            if correct_flagged > 0:
                ax.text(
                    0,
                    incorrect_flagged + correct_flagged / 2,
                    str(correct_flagged),
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                )

        fig.suptitle(f"Composition of Top 20% Most Uncertain - {run_name}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "top20_composition.png", dpi=200)
        plt.close(fig)

    except ImportError:
        console.print(
            Text("matplotlib not available for selection plots", style="yellow")
        )
    except Exception as e:
        console.print(Text(f"Error creating selection plots: {str(e)}", style="red"))


def _save_separability_analysis(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: List[str],
    outdir: Path,
    run_name: str,
) -> None:
    """
    Save violin plots with statistical tests to show separability between correct/incorrect samples.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats

        if not chosen_metrics:
            return

        # Only keep violin plots with statistical tests
        _save_violin_plots_with_stats(
            metrics_list, labels, chosen_metrics, outdir, run_name
        )

    except ImportError:
        console.print(Text("scipy not available for violin plots", style="yellow"))
    except Exception as e:
        console.print(Text(f"Error creating violin plots: {str(e)}", style="red"))


def _save_multi_run_violin_plots(
    per_run_data: Dict[str, Dict[str, List]],
    chosen_metrics: List[str],
    outdir: Path,
    run_ids: List[str],
) -> None:
    """
    Create violin plots showing individual runs with different shades.

    Args:
        per_run_data: Dict[run_id -> {"metrics": List[Dict], "labels": List[bool]}]
        chosen_metrics: List of metric names to plot
        outdir: Output directory
        run_ids: List of run IDs for consistent ordering
    """
    try:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats

        if not chosen_metrics or not per_run_data:
            return

        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(
            1, len(chosen_metrics), figsize=(4 * len(chosen_metrics), 6)
        )
        if len(chosen_metrics) == 1:
            axes = [axes]

        # Generate colors for each run
        n_runs = len(run_ids)
        green_colors = [
            mcolors.to_rgba("#2ca02c", alpha=0.4 + 0.4 * i / max(1, n_runs - 1))
            for i in range(n_runs)
        ]
        red_colors = [
            mcolors.to_rgba("#d62728", alpha=0.4 + 0.4 * i / max(1, n_runs - 1))
            for i in range(n_runs)
        ]

        for idx, metric in enumerate(chosen_metrics):
            ax = axes[idx]

            # Collect data for overall statistics
            all_corr_vals = []
            all_inc_vals = []

            # Plot individual runs
            for run_idx, run_id in enumerate(run_ids):
                if run_id not in per_run_data:
                    continue

                run_data = per_run_data[run_id]
                vals = [float(mm.get(metric, 0.0)) for mm in run_data["metrics"]]
                corr_vals = [v for v, y in zip(vals, run_data["labels"]) if y]
                inc_vals = [v for v, y in zip(vals, run_data["labels"]) if not y]

                if not corr_vals or not inc_vals:
                    continue

                # Add to overall statistics
                all_corr_vals.extend(corr_vals)
                all_inc_vals.extend(inc_vals)

                # Create violin plots for this run
                if corr_vals:
                    parts_corr = ax.violinplot(
                        [corr_vals], positions=[0.9 + run_idx * 0.02], widths=0.5
                    )
                    parts_corr["bodies"][0].set_facecolor(green_colors[run_idx])
                    parts_corr["bodies"][0].set_alpha(0.6)

                if inc_vals:
                    parts_inc = ax.violinplot(
                        [inc_vals], positions=[1.9 + run_idx * 0.02], widths=0.5
                    )
                    parts_inc["bodies"][0].set_facecolor(red_colors[run_idx])
                    parts_inc["bodies"][0].set_alpha(0.6)

            # Overall statistical test
            if len(all_corr_vals) > 1 and len(all_inc_vals) > 1:
                result = stats.ttest_ind(all_corr_vals, all_inc_vals)
                t_stat, p_value = result.statistic, result.pvalue
                p_val = float(p_value)  # Ensure p_value is a float
                significance = (
                    "***"
                    if p_val < 0.001
                    else "**"
                    if p_val < 0.01
                    else "*"
                    if p_val < 0.05
                    else "ns"
                )

                # Add significance annotation
                if all_corr_vals and all_inc_vals:
                    y_max = max(max(all_corr_vals), max(all_inc_vals))
                    y_min = min(min(all_corr_vals), min(all_inc_vals))
                    y_range = y_max - y_min
                    y_sig = y_max + 0.1 * y_range

                    ax.plot([1, 2], [y_sig, y_sig], "k-", linewidth=1)
                    ax.text(
                        1.5,
                        y_sig + 0.02 * y_range,
                        significance,
                        ha="center",
                        fontsize=12,
                        fontweight="bold",
                    )

                ax.set_title(f"{metric}\\np-value: {p_val:.2e} (n={n_runs} runs)")
            else:
                ax.set_title(f"{metric} (n={n_runs} runs)")

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Correct", "Incorrect"])
            ax.set_ylabel("Metric Value")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 2.5)

        # Add legend for runs
        from matplotlib.patches import Rectangle

        legend_elements = [
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=green_colors[-1],
                alpha=0.6,
                label="Correct (individual runs)",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=red_colors[-1],
                alpha=0.6,
                label="Incorrect (individual runs)",
            ),
        ]
        axes[0].legend(handles=legend_elements, loc="upper left")

        fig.suptitle(f"Multi-Run Distribution Comparison - {len(run_ids)} runs")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "separability_violins.png", dpi=200)
        plt.close(fig)

        console.print(
            Text(f"✓ Multi-run violin plots saved with {n_runs} runs", style="green")
        )

    except Exception as e:
        console.print(
            Text(f"Error creating multi-run violin plots: {str(e)}", style="red")
        )


def _save_normalized_histograms_with_overlap(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: List[str],
    outdir: Path,
    run_name: str,
) -> None:
    """Strategy 1: Normalized histograms showing overlap area between correct/incorrect."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.style.use("seaborn-v0_8")
        num_metrics = len(chosen_metrics)
        cols = 2 if num_metrics <= 4 else 3
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
        )

        for idx, metric in enumerate(chosen_metrics):
            r = idx // cols
            c = idx % cols
            ax = axes[r][c]

            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]

            if not corr_vals or not inc_vals:
                ax.axis("off")
                continue

            # Create normalized histograms
            bins = 30
            all_vals = corr_vals + inc_vals
            vmin, vmax = min(all_vals), max(all_vals)

            # Plot normalized histograms
            ax.hist(
                corr_vals,
                bins=bins,
                range=(vmin, vmax),
                alpha=0.6,
                density=True,
                color="#2ca02c",
                label=f"Correct (n={len(corr_vals)})",
            )
            ax.hist(
                inc_vals,
                bins=bins,
                range=(vmin, vmax),
                alpha=0.6,
                density=True,
                color="#d62728",
                label=f"Incorrect (n={len(inc_vals)})",
            )

            # Calculate and show means
            mean_corr = np.mean(corr_vals)
            mean_inc = np.mean(inc_vals)
            ax.axvline(
                mean_corr, color="#2ca02c", linestyle="--", alpha=0.8, linewidth=2
            )
            ax.axvline(
                mean_inc, color="#d62728", linestyle="--", alpha=0.8, linewidth=2
            )

            # Calculate separation metric (Cohen's d)
            pooled_std = np.sqrt(
                (
                    (len(corr_vals) - 1) * np.var(corr_vals)
                    + (len(inc_vals) - 1) * np.var(inc_vals)
                )
                / (len(corr_vals) + len(inc_vals) - 2)
            )
            cohens_d = abs(mean_corr - mean_inc) / pooled_std if pooled_std > 0 else 0

            ax.set_title(f"{metric}\\nSeparation (Cohen's d): {cohens_d:.2f}")
            ax.set_xlabel("Metric Value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Turn off unused subplots
        for j in range(num_metrics, rows * cols):
            r = j // cols
            c = j % cols
            axes[r][c].axis("off")

        fig.suptitle(f"Distribution Separability Analysis - {run_name}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "separability_histograms.png", dpi=200)
        plt.close(fig)

    except Exception as e:
        console.print(
            Text(f"Error creating histogram separability plots: {str(e)}", style="red")
        )


def _save_roc_style_curves(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: List[str],
    outdir: Path,
    run_name: str,
) -> None:
    """Strategy 2: ROC-style curves showing discrimination ability."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.style.use("seaborn-v0_8")
        num_metrics = len(chosen_metrics)
        cols = 2 if num_metrics <= 4 else 3
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
        )

        for idx, metric in enumerate(chosen_metrics):
            r = idx // cols
            c = idx % cols
            ax = axes[r][c]

            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]

            # Determine uncertainty direction
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]
            mean_corr = np.mean(corr_vals) if corr_vals else 0.0
            mean_inc = np.mean(inc_vals) if inc_vals else 0.0
            low_is_uncertain = mean_corr > mean_inc

            # Create thresholds across the range
            all_vals = sorted(set(vals))
            if len(all_vals) < 2:
                ax.axis("off")
                continue

            thresholds = np.linspace(min(all_vals), max(all_vals), 100)
            tpr_list, fpr_list = [], []

            total_incorrect = sum(1 for y in labels if not y)
            total_correct = sum(1 for y in labels if y)

            for threshold in thresholds:
                if low_is_uncertain:
                    # Low values are uncertain
                    flagged_as_uncertain = [
                        i for i, v in enumerate(vals) if v <= threshold
                    ]
                else:
                    # High values are uncertain
                    flagged_as_uncertain = [
                        i for i, v in enumerate(vals) if v >= threshold
                    ]

                tp = sum(
                    1 for i in flagged_as_uncertain if not labels[i]
                )  # True positives (correctly identified incorrect)
                fp = sum(
                    1 for i in flagged_as_uncertain if labels[i]
                )  # False positives (incorrectly flagged correct)

                tpr = (
                    tp / total_incorrect if total_incorrect > 0 else 0
                )  # Recall of incorrect
                fpr = fp / total_correct if total_correct > 0 else 0  # False alarm rate

                tpr_list.append(tpr)
                fpr_list.append(fpr)

            # Plot ROC-style curve
            ax.plot(fpr_list, tpr_list, linewidth=2, color="#1f77b4")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

            # Calculate AUC
            auc = np.trapz(tpr_list, fpr_list) if len(tpr_list) > 1 else 0

            ax.set_xlabel("False Positive Rate\\n(Fraction of Correct Flagged)")
            ax.set_ylabel("True Positive Rate\\n(Recall of Incorrect)")
            ax.set_title(f"{metric}\\nAUC: {auc:.3f}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        # Turn off unused subplots
        for j in range(num_metrics, rows * cols):
            r = j // cols
            c = j % cols
            axes[r][c].axis("off")

        fig.suptitle(f"ROC-style Discrimination Analysis - {run_name}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "separability_roc.png", dpi=200)
        plt.close(fig)

    except Exception as e:
        console.print(
            Text(f"Error creating ROC separability plots: {str(e)}", style="red")
        )


def _save_violin_plots_with_stats(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: List[str],
    outdir: Path,
    run_name: str,
) -> None:
    """Strategy 3: Violin plots with statistical significance tests."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats

        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(
            1, len(chosen_metrics), figsize=(4 * len(chosen_metrics), 6)
        )
        if len(chosen_metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(chosen_metrics):
            ax = axes[idx]

            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]

            if not corr_vals or not inc_vals:
                ax.axis("off")
                continue

            # Create violin plots
            parts = ax.violinplot([corr_vals, inc_vals], positions=[1, 2], widths=0.6)

            # Color the violins
            parts["bodies"][0].set_facecolor("#2ca02c")
            parts["bodies"][0].set_alpha(0.6)
            parts["bodies"][1].set_facecolor("#d62728")
            parts["bodies"][1].set_alpha(0.6)

            # Add box plots on top
            bp = ax.boxplot(
                [corr_vals, inc_vals],
                positions=[1, 2],
                widths=0.2,
                patch_artist=True,
                showfliers=False,
            )
            bp["boxes"][0].set_facecolor("#2ca02c")
            bp["boxes"][1].set_facecolor("#d62728")

            # Statistical test
            if len(corr_vals) > 1 and len(inc_vals) > 1:
                result = stats.ttest_ind(corr_vals, inc_vals)
                t_stat, p_value = result.statistic, result.pvalue
                p_val = float(p_value)  # Ensure p_value is a float
                significance = (
                    "***"
                    if p_val < 0.001
                    else "**"
                    if p_val < 0.01
                    else "*"
                    if p_val < 0.05
                    else "ns"
                )

                # Add significance annotation
                y_max = max(max(corr_vals), max(inc_vals))
                y_min = min(min(corr_vals), min(inc_vals))
                y_range = y_max - y_min
                y_sig = y_max + 0.1 * y_range

                ax.plot([1, 2], [y_sig, y_sig], "k-", linewidth=1)
                ax.text(
                    1.5,
                    y_sig + 0.02 * y_range,
                    significance,
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Correct", "Incorrect"])
            ax.set_ylabel("Metric Value")
            ax.set_title(
                f"{metric}\\np-value: {p_val:.2e}" if "p_val" in locals() else metric
            )
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Distribution Comparison with Statistical Tests - {run_name}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "separability_violins.png", dpi=200)
        plt.close(fig)

    except Exception as e:
        console.print(
            Text(f"Error creating violin separability plots: {str(e)}", style="red")
        )


def _save_threshold_distance_plots(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: List[str],
    outdir: Path,
    run_name: str,
) -> None:
    """Strategy 4: Show distance from optimal threshold (normalized for comparison)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(
            1, len(chosen_metrics), figsize=(5 * len(chosen_metrics), 5)
        )
        if len(chosen_metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(chosen_metrics):
            ax = axes[idx]

            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]

            if not corr_vals or not inc_vals:
                ax.axis("off")
                continue

            # Find optimal threshold (maximize F1 score)
            mean_corr = np.mean(corr_vals)
            mean_inc = np.mean(inc_vals)
            low_is_uncertain = mean_corr > mean_inc

            # Use midpoint between means as threshold
            optimal_threshold = (mean_corr + mean_inc) / 2

            # Normalize distances from threshold
            all_vals = vals
            val_range = max(all_vals) - min(all_vals)

            if val_range == 0:
                ax.axis("off")
                continue

            # Calculate normalized distances
            corr_distances = [
                (v - optimal_threshold) / val_range for v, y in zip(vals, labels) if y
            ]
            inc_distances = [
                (v - optimal_threshold) / val_range
                for v, y in zip(vals, labels)
                if not y
            ]

            # If low_is_uncertain, flip the sign so negative = uncertain
            if low_is_uncertain:
                corr_distances = [-d for d in corr_distances]
                inc_distances = [-d for d in inc_distances]

            # Create scatter plot
            y_corr = np.random.normal(1, 0.05, len(corr_distances))  # Add jitter
            y_inc = np.random.normal(0, 0.05, len(inc_distances))

            ax.scatter(
                corr_distances,
                y_corr,
                alpha=0.6,
                color="#2ca02c",
                s=20,
                label="Correct",
            )
            ax.scatter(
                inc_distances,
                y_inc,
                alpha=0.6,
                color="#d62728",
                s=20,
                label="Incorrect",
            )

            # Add threshold line
            ax.axvline(
                0,
                color="black",
                linestyle="--",
                alpha=0.8,
                linewidth=2,
                label="Optimal Threshold",
            )

            # Add mean lines
            ax.axvline(
                np.mean(corr_distances),
                color="#2ca02c",
                linestyle="-",
                alpha=0.8,
                linewidth=2,
            )
            ax.axvline(
                np.mean(inc_distances),
                color="#d62728",
                linestyle="-",
                alpha=0.8,
                linewidth=2,
            )

            ax.set_xlabel(
                "Normalized Distance from Threshold\\n← Uncertain | Certain →"
            )
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Incorrect", "Correct"])
            ax.set_title(f"{metric}\\nThreshold-based Separation")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.3, 1.3)

        fig.suptitle(f"Threshold Distance Analysis - {run_name}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "separability_threshold.png", dpi=200)
        plt.close(fig)

    except Exception as e:
        console.print(
            Text(f"Error creating threshold separability plots: {str(e)}", style="red")
        )


if __name__ == "__main__":
    # Get run IDs from the fusion experiments
    run_ids = fusion_base_runs_best()

    console.print(
        Panel(
            Text(
                f"Comprehensive Selection Experiments\nRun IDs: {', '.join(run_ids)}",
                style="bold",
            ),
            box=box.DOUBLE,
        )
    )

    # 1. Comprehensive uncertainty analysis with statistical robustness
    console.print(
        Text("\n🔍 PHASE 1: Comprehensive Uncertainty Analysis", style="bold magenta")
    )
    analyze_uncertainty_with_check(run_ids)

    # 2. Generate updated figures with all signals
    console.print(
        Text("\n📊 PHASE 2: Generate Comprehensive Figures", style="bold magenta")
    )
    generate_aggregated_figures(run_ids)

    console.print(Text("\n✅ Selection experiments completed!", style="bold green"))
    console.print(
        Text(
            "Check ./figures/selection/<run_id>/ for updated visualizations",
            style="dim",
        )
    )
