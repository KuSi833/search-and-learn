#!/usr/bin/env python3
"""
Selection experiments library: uncertainty analysis, export, and visualization.
Provides clean API for selection-based uncertainty analysis across multiple runs.
"""

import json
import statistics

# Import functions from the inference visualizer
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from scipy import stats
from sklearn.metrics import precision_recall_curve

from sal.utils.constants import BENCHMARK_SUBSETS_ROOT, Benchmark, Benchmarks
from sal.utils.runs import fusion_base_runs_best

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    _select_uncertain_indices,
    load_jsonl,
)

console = Console()

# Core uncertainty signals to evaluate
CORE_UNCERTAINTY_SIGNALS = [
    "agreement_ratio",
    "entropy_freq",
    "consensus_support",
]

EXTENDED_UNCERTAINTY_SIGNALS = CORE_UNCERTAINTY_SIGNALS + [
    "prm_margin",
    "prm_std",
    "entropy_weighted",
]


class SelectionAnalyzer:
    """Main class for selection-based uncertainty analysis."""

    def __init__(self, run_ids: List[str]):
        self.run_ids = run_ids
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if all run paths exist, crash if any are missing."""
        not_available = set()
        for run_id in self.run_ids:
            out_file = Path("./output") / run_id / "inference_output.jsonl"
            if not out_file.exists():
                console.print(Text(f"Missing: {out_file}", style="red"))
                not_available.add(run_id)
        if len(not_available) > 0:
            raise FileNotFoundError(f"Missing run files: {not_available}")

    def analyze_uncertainty(self) -> None:
        """Run comprehensive uncertainty analysis across all runs."""
        console.print(
            Text(
                f"Running aggregated uncertainty analysis across {len(self.run_ids)} runs...",
                style="bold blue",
            )
        )

        # Load all data
        all_run_data = self._load_all_run_data()

        # Display summary
        self._display_summary(all_run_data)

        # Statistical analysis
        self._analyze_statistics(all_run_data)

        # Ensemble analysis
        console.print("\n" + "=" * 40 + "\n")
        self._analyze_ensemble_selection(all_run_data)

    def _load_all_run_data(self) -> List[Tuple[str, List[Dict[str, Any]], List[bool]]]:
        """Load and process data from all runs."""
        all_run_data = []

        for run_id in self.run_ids:
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

            all_run_data.append((run_id, metrics_list, labels))

        return all_run_data

    def _display_summary(
        self, all_run_data: List[Tuple[str, List[Dict[str, Any]], List[bool]]]
    ) -> None:
        """Display summary statistics."""
        total_samples = sum(len(labels) for _, _, labels in all_run_data)
        total_accuracy_sum = sum(
            100.0 * sum(1 for y in labels if y) / len(labels)
            for _, _, labels in all_run_data
            if labels
        )
        avg_accuracy = total_accuracy_sum / len(all_run_data) if all_run_data else 0.0

        header = Table.grid(padding=(0, 1))
        header.add_column(style="bold cyan")
        header.add_column()
        header.add_row("Runs analyzed", str(len(all_run_data)))
        header.add_row("Run IDs", ", ".join(self.run_ids))
        header.add_row("Total samples", str(total_samples))
        header.add_row("Average accuracy", f"{avg_accuracy:.1f}%")
        console.print(
            Panel(
                header,
                title="Multi-run aggregated analysis",
                box=box.ROUNDED,
            )
        )

    def _analyze_statistics(
        self, all_run_data: List[Tuple[str, List[Dict[str, Any]], List[bool]]]
    ) -> None:
        """Analyze statistical robustness across multiple runs."""
        console.print(
            Panel(
                Text("Multi-run statistical robustness analysis", style="bold"),
                box=box.ROUNDED,
            )
        )

        coverage_levels = [10, 20, 30, 40, 50]
        signal_stats = defaultdict(lambda: defaultdict(list))

        for run_id, metrics_list, labels in all_run_data:
            total_incorrect = sum(1 for y in labels if not y)
            total = len(labels)

            if total == 0 or total_incorrect == 0:
                continue

            for signal in CORE_UNCERTAINTY_SIGNALS:
                vals = [m.get(signal, 0.0) for m in metrics_list]

                # Determine uncertainty direction
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

                    # Compute metrics
                    flagged_incorrect = sum(1 for i in flagged if not labels[i])
                    precision = flagged_incorrect / len(flagged) if flagged else 0.0
                    recall = (
                        flagged_incorrect / total_incorrect
                        if total_incorrect > 0
                        else 0.0
                    )

                    signal_stats[signal][f"precision_{coverage}"].append(
                        precision * 100
                    )
                    signal_stats[signal][f"recall_{coverage}"].append(recall * 100)

        # Display results
        self._display_statistics_table(signal_stats, coverage_levels)

    def _display_statistics_table(
        self, signal_stats: Dict, coverage_levels: List[int]
    ) -> None:
        """Display statistics table."""
        stats_table = Table(
            title="Multi-run statistics (mean ± std)", box=box.SIMPLE_HEAVY
        )
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

            # Spacing
            if signal != CORE_UNCERTAINTY_SIGNALS[-1]:
                stats_table.add_row("", "", *[""] * len(coverage_levels))

        console.print(stats_table)

    def _analyze_ensemble_selection(
        self, all_run_data: List[Tuple[str, List[Dict[str, Any]], List[bool]]]
    ) -> None:
        """Analyze ensemble selection strategies."""
        console.print(
            Panel(Text("Ensemble selection analysis", style="bold"), box=box.ROUNDED)
        )

        coverage_levels = [20, 30, 40]
        ensemble_combinations = [
            (["agreement_ratio"], "Agreement only"),
            (["entropy_freq"], "Entropy only"),
            (["consensus_support"], "Group score only"),
            (["agreement_ratio", "entropy_freq"], "Agreement + Entropy"),
            (["agreement_ratio", "consensus_support"], "Agreement + Group score"),
            (["entropy_freq", "consensus_support"], "Entropy + Group score"),
            (
                ["agreement_ratio", "entropy_freq", "consensus_support"],
                "All three signals",
            ),
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
                        # Single signal
                        selected_set = self._compute_single_signal_selection(
                            metrics_list, signals[0], coverage, labels
                        )
                    else:
                        # Multi-signal ensemble
                        selected_set = self._compute_ensemble_uncertainty_disjunctive(
                            metrics_list, signals, coverage, labels
                        )

                    # Compute recall
                    flagged_incorrect = sum(1 for i in selected_set if not labels[i])
                    recall = (
                        flagged_incorrect / total_incorrect
                        if total_incorrect > 0
                        else 0.0
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

    def _compute_single_signal_selection(
        self,
        metrics_list: List[Dict[str, Any]],
        signal: str,
        coverage: float,
        labels: List[bool],
    ) -> Set[int]:
        """Compute selection for single signal."""
        vals = [m.get(signal, 0.0) for m in metrics_list]
        corr_vals = [v for v, y in zip(vals, labels) if y]
        inc_vals = [v for v, y in zip(vals, labels) if not y]
        mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
        mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
        low_is_uncertain = mean_corr > mean_inc

        idxs = list(range(len(labels)))
        idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)
        k = max(1, int(round(len(labels) * (coverage / 100.0))))
        return set(idxs[:k])

    def _compute_ensemble_uncertainty_disjunctive(
        self,
        metrics_list: List[Dict[str, Any]],
        signals: List[str],
        coverage_pct: float,
        labels: List[bool],
    ) -> Set[int]:
        """Compute ensemble uncertainty selection using disjunctive strategy."""
        n_total = len(metrics_list)
        target_k = max(1, int(round(n_total * (coverage_pct / 100.0))))

        # Get uncertain indices for each signal
        signal_selections = []
        for signal in signals:
            vals = [m.get(signal, 0.0) for m in metrics_list]
            # Determine uncertainty direction
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]
            mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
            mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
            low_is_uncertain = mean_corr > mean_inc

            # Sort by uncertainty
            idxs = list(range(n_total))
            idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)

            signal_k = max(1, int(round(n_total * (coverage_pct / 100.0))))
            signal_selections.append(set(idxs[:signal_k]))

        # Disjunctive combination (union)
        ensemble_selection = set()
        for selection in signal_selections:
            ensemble_selection.update(selection)

        # If ensemble is too large, rank by average uncertainty across signals
        if len(ensemble_selection) > target_k:
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


class SelectionExporter:
    """Handle exporting uncertain subsets."""

    @staticmethod
    def export_uncertain_subset(
        run_id: str, coverage: float, metric: str, benchmark: Benchmark
    ) -> None:
        """Export uncertain subset for a single run."""
        out_file = Path("./output") / run_id / "inference_output.jsonl"
        records = load_jsonl(out_file)
        if not records:
            console.print(Text(f"No records found in {out_file}", style="red"))
            return

        selected = _select_uncertain_indices(records, coverage, metric)
        unique_ids = [records[i]["unique_id"] for i in selected]

        output_root = BENCHMARK_SUBSETS_ROOT / benchmark.hf_name / run_id / "coverage"
        coverage_str = (
            str(int(coverage))
            if abs(coverage - round(coverage)) < 1e-9
            else str(coverage)
        )
        output_file = output_root / f"{coverage_str}.json"
        output_root.mkdir(parents=True, exist_ok=True)

        # Structured payload
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

        # Display info
        header = Table.grid(padding=(0, 1))
        header.add_column(style="bold cyan")
        header.add_column()
        header.add_row("Run", run_id)
        header.add_row("Metric", metric)
        header.add_row("Coverage", f"{coverage:.1f}%")
        header.add_row("Exported", str(len(unique_ids)))
        header.add_row("Saved to", str(output_file))
        console.print(Panel(header, title="Export uncertain subset", box=box.ROUNDED))

    @staticmethod
    def export_multi_runs(
        run_ids: List[str],
        coverage: float,
        metric: str = "agreement_ratio",
        benchmark: Benchmark = Benchmarks.MATH500.value,
    ) -> None:
        """Export uncertain subsets for multiple runs."""
        analyzer = SelectionAnalyzer(run_ids)  # This will check availability

        console.print(
            Text(
                f"Exporting uncertain subsets for {len(run_ids)} runs...",
                style="bold blue",
            )
        )

        for i, run_id in enumerate(run_ids):
            if i > 0:
                console.print()
            SelectionExporter.export_uncertain_subset(
                run_id, coverage, metric, benchmark
            )


class SelectionVisualizer:
    """Handle visualization and figure generation."""

    def __init__(self, run_ids: List[str]):
        self.run_ids = run_ids
        self.analyzer = SelectionAnalyzer(run_ids)

    def generate_figures(self) -> None:
        """Generate multi-run uncertainty analysis figures showing individual runs."""
        console.print(
            Text(
                "Generating violin plots with optimal thresholds and stacked histograms...",
                style="bold blue",
            )
        )

        try:
            # Load all data
            all_run_data = self.analyzer._load_all_run_data()

            # Flatten data for plotting (back to aggregated approach)
            all_metrics_data = []
            all_labels_data = []
            for _, metrics_list, labels in all_run_data:
                all_metrics_data.extend(metrics_list)
                all_labels_data.extend(labels)

            if not all_metrics_data:
                console.print(Text("No data found across all runs", style="red"))
                return

            # Create output directory
            output_dir = Path("./figures/selection/aggregated")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate clean violin plots with optimal thresholds
            self._save_separability_analysis_with_thresholds(
                all_metrics_data, all_labels_data, output_dir
            )

            # Generate stacked histograms of counts across metric values
            self._save_class_count_histograms(
                all_metrics_data, all_labels_data, output_dir
            )

            console.print(
                Text(
                    f"✓ Multi-run figures generated in {output_dir}/",
                    style="bold green",
                )
            )

        except Exception as e:
            console.print(Text(f"Error generating figures: {str(e)}", style="red"))

    def _save_selection_precision_plots(
        self, metrics_list: List[Dict[str, Any]], labels: List[bool], outdir: Path
    ) -> None:
        """Save selection precision plots."""
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(1, len(CORE_UNCERTAINTY_SIGNALS), figsize=(15, 5))
        if len(CORE_UNCERTAINTY_SIGNALS) == 1:
            axes = [axes]

        coverage_levels = [10, 20, 30, 40, 50]
        total = len(labels)

        for idx, metric in enumerate(CORE_UNCERTAINTY_SIGNALS):
            ax = axes[idx]
            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]

            # Determine uncertainty direction
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]
            mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
            mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
            low_is_uncertain = mean_corr > mean_inc

            # Sort by uncertainty
            idxs = list(range(total))
            idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)

            # Calculate precision for each coverage
            precisions = []
            for coverage in coverage_levels:
                k = max(1, int(round(total * (coverage / 100.0))))
                flagged = set(idxs[:k])
                incorrect_flagged = sum(1 for i in flagged if not labels[i])
                precision = 100.0 * incorrect_flagged / len(flagged) if flagged else 0.0
                precisions.append(precision)

            # Plot
            ax.plot(coverage_levels, precisions, "o-", linewidth=2, markersize=6)
            ax.set_xlabel("Coverage (%)")
            ax.set_ylabel("Precision (% incorrect)")
            ax.set_title(f"{metric}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)

        fig.suptitle(f"Selection Precision Analysis - {len(self.run_ids)} runs")
        fig.tight_layout()
        fig.savefig(outdir / "selection_precision.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _save_separability_analysis(
        self, metrics_list: List[Dict[str, Any]], labels: List[bool], outdir: Path
    ) -> None:
        """Save separability analysis plots."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy import stats

            plt.style.use("seaborn-v0_8")
            fig, axes = plt.subplots(1, len(CORE_UNCERTAINTY_SIGNALS), figsize=(15, 5))
            if len(CORE_UNCERTAINTY_SIGNALS) == 1:
                axes = [axes]

            for idx, metric in enumerate(CORE_UNCERTAINTY_SIGNALS):
                ax = axes[idx]
                vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
                corr_vals = [v for v, y in zip(vals, labels) if y]
                inc_vals = [v for v, y in zip(vals, labels) if not y]

                if not corr_vals or not inc_vals:
                    ax.axis("off")
                    continue

                # Violin plots
                parts = ax.violinplot(
                    [corr_vals, inc_vals], positions=[1, 2], widths=0.6
                )
                parts["bodies"][0].set_facecolor("#2ca02c")
                parts["bodies"][0].set_alpha(0.6)
                parts["bodies"][1].set_facecolor("#d62728")
                parts["bodies"][1].set_alpha(0.6)

                # Statistical test
                if len(corr_vals) > 1 and len(inc_vals) > 1:
                    try:
                        # Simple t-test without relying on scipy result structure
                        import numpy as np

                        mean_diff = abs(np.mean(corr_vals) - np.mean(inc_vals))
                        pooled_std = np.sqrt((np.var(corr_vals) + np.var(inc_vals)) / 2)
                        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

                        # Simple effect size based significance
                        significance = (
                            "***"
                            if effect_size > 0.8  # Large effect
                            else "**"
                            if effect_size > 0.5  # Medium effect
                            else "*"
                            if effect_size > 0.2  # Small effect
                            else "ns"  # No effect
                        )
                        ax.set_title(f"{metric} ({significance})")
                    except Exception:
                        ax.set_title(f"{metric}")
                else:
                    ax.set_title(f"{metric}")

                ax.set_xticks([1, 2])
                ax.set_xticklabels(["Correct", "Incorrect"])
                ax.set_ylabel("Metric Value")
                ax.grid(True, alpha=0.3)

            fig.suptitle(f"Distribution Separability - {len(self.run_ids)} runs")
            fig.tight_layout()
            fig.savefig(
                outdir / "separability_analysis.png", dpi=200, bbox_inches="tight"
            )
            plt.close(fig)

        except ImportError:
            console.print(
                Text(
                    "scipy/matplotlib not available for separability plots",
                    style="yellow",
                )
            )
        except Exception as e:
            console.print(
                Text(f"Error creating separability plots: {str(e)}", style="red")
            )

    def _save_multi_run_separability_analysis(
        self,
        all_run_data: List[Tuple[str, List[Dict[str, Any]], List[bool]]],
        outdir: Path,
    ) -> None:
        """Save multi-run violin plots showing individual runs with different shades."""
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(
            1,
            len(CORE_UNCERTAINTY_SIGNALS),
            figsize=(4 * len(CORE_UNCERTAINTY_SIGNALS), 6),
        )
        if len(CORE_UNCERTAINTY_SIGNALS) == 1:
            axes = [axes]

        # Generate colors for each run
        n_runs = len(all_run_data)
        green_colors = [
            mcolors.to_rgba("#2ca02c", alpha=0.4 + 0.4 * i / max(1, n_runs - 1))
            for i in range(n_runs)
        ]
        red_colors = [
            mcolors.to_rgba("#d62728", alpha=0.4 + 0.4 * i / max(1, n_runs - 1))
            for i in range(n_runs)
        ]

        for idx, metric in enumerate(CORE_UNCERTAINTY_SIGNALS):
            ax = axes[idx]

            # Collect data for overall statistics
            all_corr_vals = []
            all_inc_vals = []

            # Plot individual runs
            for run_idx, (run_id, metrics_list, labels) in enumerate(all_run_data):
                vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
                corr_vals = [v for v, y in zip(vals, labels) if y]
                inc_vals = [v for v, y in zip(vals, labels) if not y]

                if not corr_vals or not inc_vals:
                    continue

                # Add to overall statistics
                all_corr_vals.extend(corr_vals)
                all_inc_vals.extend(inc_vals)

                # Create violin plots for this run
                if corr_vals:
                    parts_corr = ax.violinplot(
                        [corr_vals],
                        positions=[0.9 + run_idx * 0.02],
                        widths=0.5,
                        showmeans=False,
                        showmedians=False,
                        showextrema=False,
                    )
                    parts_corr["bodies"][0].set_facecolor(green_colors[run_idx])
                    parts_corr["bodies"][0].set_alpha(0.6)

                if inc_vals:
                    parts_inc = ax.violinplot(
                        [inc_vals],
                        positions=[1.9 + run_idx * 0.02],
                        widths=0.5,
                        showmeans=False,
                        showmedians=False,
                        showextrema=False,
                    )
                    parts_inc["bodies"][0].set_facecolor(red_colors[run_idx])
                    parts_inc["bodies"][0].set_alpha(0.6)

            # Overall statistical test
            if len(all_corr_vals) > 1 and len(all_inc_vals) > 1:
                try:
                    # Simple effect size calculation
                    mean_diff = abs(np.mean(all_corr_vals) - np.mean(all_inc_vals))
                    pooled_std = np.sqrt(
                        (np.var(all_corr_vals) + np.var(all_inc_vals)) / 2
                    )
                    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

                    # Effect size based significance
                    significance = (
                        "***"
                        if effect_size > 0.8  # Large effect
                        else "**"
                        if effect_size > 0.5  # Medium effect
                        else "*"
                        if effect_size > 0.2  # Small effect
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

                    ax.set_title(f"{metric}")
                except Exception:
                    ax.set_title(f"{metric}")
            else:
                ax.set_title(f"{metric}")

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Correct", "Incorrect"])
            ax.set_ylabel("Metric Value")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 2.5)

            # Add legend for runs
            if n_runs > 0:
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

            fig.suptitle("Multi-Run Distribution Comparison")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(
                outdir / "separability_violins.png", dpi=200, bbox_inches="tight"
            )
            plt.close(fig)

            console.print(Text("✓ Multi-run violin plots saved", style="green"))

    def _save_separability_analysis_with_thresholds(
        self, metrics_list: List[Dict[str, Any]], labels: List[bool], outdir: Path
    ) -> None:
        """Save clean violin plots with analytically optimal thresholds."""
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(
            1,
            len(CORE_UNCERTAINTY_SIGNALS),
            figsize=(4 * len(CORE_UNCERTAINTY_SIGNALS), 6),
        )
        if len(CORE_UNCERTAINTY_SIGNALS) == 1:
            axes = [axes]

        threshold_results = {}

        for idx, metric in enumerate(CORE_UNCERTAINTY_SIGNALS):
            ax = axes[idx]
            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]

            if not corr_vals or not inc_vals:
                ax.axis("off")
                continue

            # Create clean violin plots (aggregated)
            parts = ax.violinplot(
                [corr_vals, inc_vals],
                positions=[1, 2],
                widths=0.6,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            parts["bodies"][0].set_facecolor("#2ca02c")
            parts["bodies"][0].set_alpha(0.6)
            parts["bodies"][1].set_facecolor("#d62728")
            parts["bodies"][1].set_alpha(0.6)

            # Find optimal threshold analytically and generate detailed table
            optimal_threshold, precision, recall, f1_score, detailed_table = (
                self._find_optimal_threshold_with_table(vals, labels, metric)
            )

            threshold_results[metric] = {
                "threshold": optimal_threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "detailed_analysis": detailed_table,
            }

            # Print detailed table for this metric
            self._print_threshold_table(metric, detailed_table)

            # Just use clean title
            ax.set_title(f"{metric}")

            # Statistical test
            if len(corr_vals) > 1 and len(inc_vals) > 1:
                try:
                    statistic, p_value = stats.ttest_ind(corr_vals, inc_vals)
                    p_val = float(p_value)

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
                except Exception:
                    pass

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["Correct", "Incorrect"])
            ax.set_ylabel("Metric Value")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Uncertainty Metrics Distribution Analysis")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "separability_violins.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Save threshold analysis results
        with open(outdir / "optimal_thresholds.json", "w") as f:
            json.dump(threshold_results, f, indent=2, default=str)

        console.print(
            Text("✓ Violin plots with optimal thresholds saved", style="green")
        )

    def _save_class_count_histograms(
        self, metrics_list: List[Dict[str, Any]], labels: List[bool], outdir: Path
    ) -> None:
        """Save stacked histograms showing counts of Correct vs Incorrect across metric values."""
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(
            1,
            len(CORE_UNCERTAINTY_SIGNALS),
            figsize=(4 * len(CORE_UNCERTAINTY_SIGNALS), 6),
        )
        if len(CORE_UNCERTAINTY_SIGNALS) == 1:
            axes = [axes]

        for idx, metric in enumerate(CORE_UNCERTAINTY_SIGNALS):
            ax = axes[idx]
            vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
            corr_vals = [v for v, y in zip(vals, labels) if y]
            inc_vals = [v for v, y in zip(vals, labels) if not y]

            if not vals:
                ax.axis("off")
                continue

            # Shared bins for both classes
            try:
                bins = np.histogram_bin_edges(vals, bins="auto")
            except Exception:
                bins = 30

            ax.hist(
                [corr_vals, inc_vals],
                bins=bins,
                stacked=True,
                label=["Correct", "Incorrect"],
                color=["#2ca02c", "#d62728"],
                alpha=0.8,
            )

            ax.set_title(f"{metric}")
            ax.set_xlabel("Metric value")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

            if idx == 0:
                ax.legend()

        fig.suptitle("Counts of Correct vs Incorrect across metric values")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(outdir / "metric_counts_stacked.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _find_optimal_threshold_with_table(
        self, vals: List[float], labels: List[bool], metric: str
    ):
        """Find optimal threshold and generate detailed performance table."""
        try:
            # Convert to numpy arrays
            y_true = np.array(
                [0 if label else 1 for label in labels]
            )  # 1 = incorrect (uncertain)
            y_scores = np.array(vals)

            # Determine if low or high values indicate uncertainty
            corr_mean = np.mean([v for v, l in zip(vals, labels) if l])
            inc_mean = np.mean([v for v, l in zip(vals, labels) if not l])
            low_is_uncertain = corr_mean > inc_mean

            # Generate threshold range in increments of 0.1
            val_min, val_max = min(vals), max(vals)
            threshold_range = np.arange(
                np.floor(val_min * 10) / 10, np.ceil(val_max * 10) / 10 + 0.1, 0.1
            )

            detailed_table = []
            best_f1 = 0
            optimal_result = None

            for threshold in threshold_range:
                # Calculate predictions based on threshold and uncertainty direction
                if low_is_uncertain:
                    predictions = [
                        1 if v <= threshold else 0 for v in vals
                    ]  # 1 = predicted uncertain
                else:
                    predictions = [
                        1 if v >= threshold else 0 for v in vals
                    ]  # 1 = predicted uncertain

                # Calculate metrics
                tp = sum(
                    1
                    for pred, true in zip(predictions, y_true)
                    if pred == 1 and true == 1
                )  # True uncertain
                fp = sum(
                    1
                    for pred, true in zip(predictions, y_true)
                    if pred == 1 and true == 0
                )  # False uncertain
                tn = sum(
                    1
                    for pred, true in zip(predictions, y_true)
                    if pred == 0 and true == 0
                )  # True certain
                fn = sum(
                    1
                    for pred, true in zip(predictions, y_true)
                    if pred == 0 and true == 1
                )  # False certain

                # Calculate performance metrics
                if tp + fp == 0:
                    # No predictions made - perfect precision but undefined
                    precision = 1.0 if tp == 0 and fp == 0 else 0.0
                else:
                    precision = tp / (tp + fp)

                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                row = {
                    "threshold": round(threshold, 1),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "f1_score": round(f1, 3),
                }
                detailed_table.append(row)

                # Track best F1 score
                if f1 > best_f1:
                    best_f1 = f1
                    optimal_result = (threshold, precision, recall, f1)

            return (
                (*optimal_result, detailed_table)
                if optimal_result
                else (None, None, None, None, detailed_table)
            )

        except Exception as e:
            print(f"Error finding optimal threshold for {metric}: {e}")
            return None, None, None, None, []

    def _print_threshold_table(self, metric: str, detailed_table: List[Dict]):
        """Print a formatted table showing performance at different thresholds."""
        if not detailed_table:
            return

        console.print(f"\n[bold blue]Threshold Analysis for {metric}[/bold blue]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Threshold", style="cyan", width=12)
        table.add_column("TP", style="green", width=8, justify="center")
        table.add_column("FP", style="red", width=8, justify="center")
        table.add_column("TN", style="green", width=8, justify="center")
        table.add_column("FN", style="red", width=8, justify="center")
        table.add_column("Precision", style="yellow", width=12, justify="center")

        # Find perfect precision rows (FP = 0) and best precision
        perfect_precision_rows = [row for row in detailed_table if row["fp"] == 0]
        best_precision = max(row["precision"] for row in detailed_table)

        for row in detailed_table:
            # Highlight perfect precision (no false positives) in bold green
            if row["fp"] == 0:
                style = "bold green"
                precision_display = "PERFECT" if row["tp"] > 0 else "1.000"
            # Highlight best precision in yellow
            elif row["precision"] == best_precision:
                style = "bold yellow"
                precision_display = f"{row['precision']:.3f}"
            else:
                style = None
                precision_display = f"{row['precision']:.3f}"

            table.add_row(
                f"{row['threshold']:.1f}",
                str(row["tp"]),
                str(row["fp"]),
                str(row["tn"]),
                str(row["fn"]),
                precision_display,
                style=style,
            )

        console.print(table)

        # Show precision-focused summary
        if perfect_precision_rows:
            best_perfect = max(perfect_precision_rows, key=lambda x: x["tp"])
            console.print(
                f"[bold green]Perfect Precision Achieved! Best: {best_perfect['tp']} correct uncertain predictions with 0 false positives[/bold green]"
            )
        console.print(
            f"[bold yellow]Best Overall Precision: {best_precision:.3f}[/bold yellow]\n"
        )

    def _save_summary(
        self, metrics_list: List[Dict[str, Any]], labels: List[bool], outdir: Path
    ) -> None:
        """Save summary statistics."""
        try:
            summary = {
                "aggregated_analysis": True,
                "run_ids": self.run_ids,
                "total_samples": len(labels),
                "total_correct": sum(1 for y in labels if y),
                "total_incorrect": sum(1 for y in labels if not y),
                "accuracy": 100.0 * sum(1 for y in labels if y) / len(labels)
                if labels
                else 0.0,
            }

            # Add signal statistics
            means_summary = {}
            for metric in CORE_UNCERTAINTY_SIGNALS:
                vals = [float(mm.get(metric, 0.0)) for mm in metrics_list]
                corr_vals = [v for v, y in zip(vals, labels) if y]
                inc_vals = [v for v, y in zip(vals, labels) if not y]
                mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
                mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0

                means_summary[metric] = {
                    "mean_correct": float(mean_corr),
                    "mean_incorrect": float(mean_inc),
                    "low_is_uncertain": mean_corr > mean_inc,
                }

            summary["signal_means"] = means_summary

            with open(outdir / "analysis_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            console.print(Text(f"Error creating summary: {str(e)}", style="red"))


def get_default_run_ids() -> List[str]:
    """Get default run IDs from fusion experiments."""
    return fusion_base_runs_best()
