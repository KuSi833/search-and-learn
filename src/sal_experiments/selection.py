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

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
    "group_top_frac",
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
            (["group_top_frac"], "Group score only"),
            (["agreement_ratio", "entropy_freq"], "Agreement + Entropy"),
            (["agreement_ratio", "group_top_frac"], "Agreement + Group score"),
            (["entropy_freq", "group_top_frac"], "Entropy + Group score"),
            (
                ["agreement_ratio", "entropy_freq", "group_top_frac"],
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
        """Generate aggregated uncertainty analysis figures across all runs."""
        console.print(
            Text(
                f"Generating aggregated figures across {len(self.run_ids)} runs...",
                style="bold blue",
            )
        )

        try:
            # Load all data
            all_run_data = self.analyzer._load_all_run_data()

            # Flatten data for plotting
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

            # Generate plots
            self._save_selection_precision_plots(
                all_metrics_data, all_labels_data, output_dir
            )
            self._save_separability_analysis(
                all_metrics_data, all_labels_data, output_dir
            )
            self._save_summary(all_metrics_data, all_labels_data, output_dir)

            console.print(
                Text(f"✓ Figures generated in {output_dir}/", style="bold green")
            )

        except Exception as e:
            console.print(Text(f"Error generating figures: {str(e)}", style="red"))

    def _save_selection_precision_plots(
        self, metrics_list: List[Dict[str, Any]], labels: List[bool], outdir: Path
    ) -> None:
        """Save selection precision plots."""
        try:
            import matplotlib.pyplot as plt

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
                    precision = (
                        100.0 * incorrect_flagged / len(flagged) if flagged else 0.0
                    )
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
            fig.savefig(
                outdir / "selection_precision.png", dpi=200, bbox_inches="tight"
            )
            plt.close(fig)

        except ImportError:
            console.print(Text("matplotlib not available for plotting", style="yellow"))
        except Exception as e:
            console.print(
                Text(f"Error creating precision plots: {str(e)}", style="red")
            )

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
