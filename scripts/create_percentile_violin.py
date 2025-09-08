#!/usr/bin/env python3
"""
Create violin plots using percentile-based thresholds instead of fixed count selection.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from rich.console import Console

# from sal.utils.runs import fusion_base_runs_best
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    load_jsonl,
)

# Configuration
PERCENTILE = 10  # Change this value to adjust the selection percentile


def get_metric_direction(
    metric: str,
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
) -> str:
    """Determine if a metric uses <= or >= threshold based on mean comparison.

    Returns "<=" if low values indicate uncertainty, ">=" if high values indicate uncertainty.
    """
    correct_mean = np.mean(correct_data[metric]) if correct_data[metric] else 0.0
    incorrect_mean = np.mean(incorrect_data[metric]) if incorrect_data[metric] else 0.0

    # If incorrect answers have higher mean values, high values indicate uncertainty
    if incorrect_mean > correct_mean:
        return ">="
    else:
        # If incorrect answers have lower mean values, low values indicate uncertainty
        return "<="


def is_low_uncertainty_metric(
    metric: str,
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
) -> bool:
    """Return True if low values indicate high uncertainty for this metric."""
    return get_metric_direction(metric, correct_data, incorrect_data) == "<="


def calculate_percentile_thresholds(
    all_data: Dict[str, List[float]],
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    percentile: float = 15.0,
) -> Dict[str, float]:
    """Calculate thresholds based on percentiles of the combined data.

    Args:
        percentile: Percentage of data to select as uncertain (e.g., 15.0 for 15%)
    """
    thresholds = {}

    for metric in all_data.keys():
        if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
            # For low uncertainty metrics, use the percentile directly
            thresholds[metric] = np.percentile(all_data[metric], percentile)
        else:
            # For high uncertainty metrics, use 100 - percentile
            thresholds[metric] = np.percentile(all_data[metric], 100 - percentile)

    return thresholds


def analyze_percentile_correctness(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, int]]:
    """Analyze how many correct vs incorrect answers fall within each percentile threshold."""
    analysis = {}

    for metric in thresholds.keys():
        if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
            # Count values <= threshold
            correct_selected = sum(
                1 for x in correct_data[metric] if x <= thresholds[metric]
            )
            incorrect_selected = sum(
                1 for x in incorrect_data[metric] if x <= thresholds[metric]
            )
        else:
            # Count values >= threshold
            correct_selected = sum(
                1 for x in correct_data[metric] if x >= thresholds[metric]
            )
            incorrect_selected = sum(
                1 for x in incorrect_data[metric] if x >= thresholds[metric]
            )

        analysis[metric] = {
            "correct_selected": correct_selected,
            "incorrect_selected": incorrect_selected,
            "total_selected": correct_selected + incorrect_selected,
        }

    return analysis


def calculate_f1_scores_percentile(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    percentile: float,
) -> Dict[str, float]:
    """Calculate F1 scores for each metric at the given percentile."""
    f1_scores = {}

    for metric in all_data.keys():
        # Calculate threshold for this percentile
        if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
            threshold = np.percentile(all_data[metric], percentile)
            # Count values <= threshold (uncertain)
            correct_selected = sum(1 for x in correct_data[metric] if x <= threshold)
            incorrect_selected = sum(
                1 for x in incorrect_data[metric] if x <= threshold
            )
        else:
            threshold = np.percentile(all_data[metric], 100 - percentile)
            # Count values >= threshold (uncertain)
            correct_selected = sum(1 for x in correct_data[metric] if x >= threshold)
            incorrect_selected = sum(
                1 for x in incorrect_data[metric] if x >= threshold
            )

        # Calculate confusion matrix values
        # TP = incorrect answers correctly identified as uncertain
        # FP = correct answers incorrectly identified as uncertain
        # FN = incorrect answers incorrectly identified as certain
        # TN = correct answers correctly identified as certain
        tp = (
            incorrect_selected  # True positives: incorrect answers flagged as uncertain
        )
        fp = correct_selected  # False positives: correct answers flagged as uncertain
        fn = (
            len(incorrect_data[metric]) - incorrect_selected
        )  # False negatives: incorrect answers not flagged
        tn = (
            len(correct_data[metric]) - correct_selected
        )  # True negatives: correct answers not flagged

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        f1_scores[metric] = f1

    return f1_scores


def create_percentile_violin_plots(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    run_ids: List[str],
    thresholds: Dict[str, float],
    correctness_analysis: Dict[str, Dict[str, int]],
    percentile: float,
    f1_scores: Dict[str, float],
):
    """Create split violin plots with percentile-based thresholds."""
    # Sort metrics by F1 score in descending order
    metrics = sorted(
        thresholds.keys(), key=lambda x: f1_scores.get(x, 0.0), reverse=True
    )
    n_metrics = len(metrics)

    # Create a grid layout - 3 columns, as many rows as needed
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))

    # Ensure axes is always 2D for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, metric in enumerate(metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]

        # Calculate proportional widths based on counts
        correct_count = len(correct_data[metric])
        incorrect_count = len(incorrect_data[metric])
        total_count = correct_count + incorrect_count

        # Scale widths proportionally (max width = 0.8)
        max_width = 1.0
        correct_width = max_width * (correct_count / total_count)
        incorrect_width = max_width * (incorrect_count / total_count)

        # Create violin plot for correct answers (left side)
        parts_correct = ax.violinplot(
            [correct_data[metric]],
            positions=[0.5],
            widths=[correct_width],
            showmeans=False,
            showmedians=False,
        )

        # Get threshold for this metric
        threshold_value = thresholds[metric]

        # Style correct violin (left side) with threshold-based transparency
        for pc in parts_correct["bodies"]:
            pc.set_facecolor("#2ca02c")

            # Keep only left half of the violin
            vertices = pc.get_paths()[0].vertices
            center_x = 0.5
            vertices[:, 0] = np.where(
                vertices[:, 0] > center_x, center_x, vertices[:, 0]
            )

            # Split vertices based on threshold
            if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
                # For these metrics, inside threshold means y <= threshold_value
                inside_vertices = vertices[vertices[:, 1] <= threshold_value]
                outside_vertices = vertices[vertices[:, 1] > threshold_value]
            else:
                # For high uncertainty metrics, inside threshold means y >= threshold_value
                inside_vertices = vertices[vertices[:, 1] >= threshold_value]
                outside_vertices = vertices[vertices[:, 1] < threshold_value]

            # Remove the original patch
            pc.set_alpha(0)

            # Add inside threshold patch (darker)
            if len(inside_vertices) > 2:
                inside_patch = Polygon(inside_vertices, facecolor="#2ca02c", alpha=0.8)
                ax.add_patch(inside_patch)

            # Add outside threshold patch (lighter)
            if len(outside_vertices) > 2:
                outside_patch = Polygon(
                    outside_vertices, facecolor="#2ca02c", alpha=0.3
                )
                ax.add_patch(outside_patch)

        # Remove any mean/median lines for correct
        for key in ["cmeans", "cmedians", "cmins", "cmaxes", "cbars"]:
            if key in parts_correct:
                parts_correct[key].set_visible(False)

        # Create violin plot for incorrect answers (right side)
        parts_incorrect = ax.violinplot(
            [incorrect_data[metric]],
            positions=[0.5],
            widths=[incorrect_width],
            showmeans=False,
            showmedians=False,
        )

        # Style incorrect violin (right side) with threshold-based transparency
        for pc in parts_incorrect["bodies"]:
            pc.set_facecolor("#d62728")

            # Keep only right half of the violin
            vertices = pc.get_paths()[0].vertices
            center_x = 0.5
            vertices[:, 0] = np.where(
                vertices[:, 0] < center_x, center_x, vertices[:, 0]
            )

            # Split vertices based on threshold
            if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
                # For these metrics, inside threshold means y <= threshold_value
                inside_vertices = vertices[vertices[:, 1] <= threshold_value]
                outside_vertices = vertices[vertices[:, 1] > threshold_value]
            else:
                # For high uncertainty metrics, inside threshold means y >= threshold_value
                inside_vertices = vertices[vertices[:, 1] >= threshold_value]
                outside_vertices = vertices[vertices[:, 1] < threshold_value]

            # Remove the original patch
            pc.set_alpha(0)

            # Add inside threshold patch (darker)
            if len(inside_vertices) > 2:
                inside_patch = Polygon(inside_vertices, facecolor="#d62728", alpha=0.8)
                ax.add_patch(inside_patch)

            # Add outside threshold patch (lighter)
            if len(outside_vertices) > 2:
                outside_patch = Polygon(
                    outside_vertices, facecolor="#d62728", alpha=0.3
                )
                ax.add_patch(outside_patch)

        # Remove any mean/median lines for incorrect
        for key in ["cmeans", "cmedians", "cmins", "cmaxes", "cbars"]:
            if key in parts_incorrect:
                parts_incorrect[key].set_visible(False)

        # Set proper y-axis limits to ensure violin plots stretch from top to bottom
        all_values = correct_data[metric] + incorrect_data[metric]
        y_min = min(all_values)
        y_max = max(all_values)

        # Add small padding to ensure full visibility
        y_range = y_max - y_min
        padding = y_range * 0.05 if y_range > 0 else 0.05
        ax.set_ylim(y_min - padding, y_max + padding)

        # Add threshold line and selection visualization
        threshold_value = thresholds[metric]
        analysis = correctness_analysis[metric]

        # Determine selection region based on metric type
        if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
            # For these metrics, we select values <= threshold (bottom region)
            y_fill_min = y_min - padding
            y_fill_max = threshold_value
        else:
            # For high uncertainty metrics, we select values >= threshold (top region)
            y_fill_min = threshold_value
            y_fill_max = y_max + padding

        # Add shaded region to show what's being selected
        ax.axhspan(y_fill_min, y_fill_max, alpha=0.15, color="orange", zorder=0)

        # Add threshold line
        ax.axhline(
            y=threshold_value,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Threshold: {threshold_value:.3f}",
        )

        # Customize plot
        ax.set_xlim(0, 1)
        ax.set_xticks([0.25, 0.75])
        ax.set_xticklabels(
            [
                f"Correct\n(n={len(correct_data[metric])})",
                f"Incorrect\n(n={len(incorrect_data[metric])})",
            ]
        )
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}")

        # Include F1 score in the title
        f1_score = f1_scores.get(metric, 0.0)
        title = f"{metric.replace('_', ' ').title()} (F1: {f1_score:.3f})"
        ax.set_title(title, fontsize=10)

        # Create enhanced legend with selection counts
        direction = get_metric_direction(metric, correct_data, incorrect_data)
        analysis = correctness_analysis[metric]
        selected_correct = analysis["correct_selected"]
        selected_incorrect = analysis["incorrect_selected"]

        # Create custom legend entries with separate correct/incorrect counts
        legend_elements = [
            Line2D(
                [0],
                [0],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold {direction} {threshold_value:.3f}",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="orange",
                alpha=0.15,
                label="Selected region:",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#2ca02c",
                alpha=0.6,
                label=f"  Correct: {selected_correct}",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#d62728",
                alpha=0.6,
                label=f"  Incorrect: {selected_incorrect}",
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0, 0.75),
            fontsize=10,
            framealpha=0.9,
        )
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row][col].set_visible(False)
        else:
            axes[col].set_visible(False)

    # Save the plot
    output_dir = Path("figures/selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir / f"split_violins_percentile_{percentile}pct_{'_'.join(run_ids)}.png"
    )
    plt.savefig(output_file, dpi=200, bbox_inches="tight")

    print(f"Percentile-based violin plots saved to: {output_file}")

    plt.close()
    return output_file


def export_percentile_data(
    output_file: Path,
    run_ids: List[str],
    thresholds: Dict[str, float],
    correctness_analysis: Dict[str, Dict[str, int]],
    all_correct_data: Dict[str, List[float]],
    all_incorrect_data: Dict[str, List[float]],
    metrics_list: List[str],
    percentile: float,
) -> None:
    """Export percentile-based analysis in LLM-readable JSON format."""

    # Calculate summary statistics for each metric
    export_data = {
        "metadata": {
            "description": "Uncertainty metrics analysis with percentile-based selection",
            "selection_method": "percentile",
            "percentile": percentile,
            "run_ids": run_ids,
            "total_metrics": len(metrics_list),
            "export_timestamp": str(output_file.parent),
        },
        "metrics": {},
    }

    for metric in metrics_list:
        if metric not in all_correct_data or metric not in all_incorrect_data:
            continue

        correct_values = all_correct_data[metric]
        incorrect_values = all_incorrect_data[metric]
        all_values = correct_values + incorrect_values

        if not all_values:
            continue

        # Calculate statistics
        correct_mean = float(np.mean(correct_values)) if correct_values else 0.0
        correct_std = float(np.std(correct_values)) if correct_values else 0.0
        incorrect_mean = float(np.mean(incorrect_values)) if incorrect_values else 0.0
        incorrect_std = float(np.std(incorrect_values)) if incorrect_values else 0.0

        # Determine direction and reasoning
        direction = (
            "≤"
            if is_low_uncertainty_metric(metric, all_correct_data, all_incorrect_data)
            else "≥"
        )
        interpretation = (
            "low_values_uncertain" if direction == "≤" else "high_values_uncertain"
        )
        reasoning = f"Incorrect mean ({incorrect_mean:.3f}) {'<' if direction == '≤' else '>'} Correct mean ({correct_mean:.3f})"

        analysis = correctness_analysis[metric]

        export_data["metrics"][metric] = {
            "threshold": {
                "value": float(thresholds[metric]),
                "direction": direction,
                "interpretation": interpretation,
                "reasoning": reasoning,
            },
            "statistics": {
                "correct": {
                    "count": len(correct_values),
                    "mean": correct_mean,
                    "std": correct_std,
                    "min": float(np.min(correct_values)) if correct_values else 0.0,
                    "max": float(np.max(correct_values)) if correct_values else 0.0,
                },
                "incorrect": {
                    "count": len(incorrect_values),
                    "mean": incorrect_mean,
                    "std": incorrect_std,
                    "min": float(np.min(incorrect_values)) if incorrect_values else 0.0,
                    "max": float(np.max(incorrect_values)) if incorrect_values else 0.0,
                },
                "combined": {
                    "count": len(all_values),
                    "mean": float(np.mean(all_values)),
                    "std": float(np.std(all_values)),
                    "min": float(np.min(all_values)),
                    "max": float(np.max(all_values)),
                },
            },
            "selection_analysis": {
                "correct_selected": analysis["correct_selected"],
                "incorrect_selected": analysis["incorrect_selected"],
                "total_selected": analysis["total_selected"],
                "selection_accuracy": float(
                    100 * analysis["incorrect_selected"] / analysis["total_selected"]
                )
                if analysis["total_selected"] > 0
                else 0.0,
                "selection_percentage": float(
                    100 * analysis["total_selected"] / len(all_values)
                )
                if all_values
                else 0.0,
            },
            "uncertainty_effectiveness": {
                "mean_difference": float(abs(incorrect_mean - correct_mean)),
                "effect_size": float(
                    abs(incorrect_mean - correct_mean)
                    / np.sqrt((correct_std**2 + incorrect_std**2) / 2)
                )
                if (correct_std > 0 or incorrect_std > 0)
                else 0.0,
                "separability": "high"
                if abs(incorrect_mean - correct_mean) > 0.1
                else "medium"
                if abs(incorrect_mean - correct_mean) > 0.05
                else "low",
            },
        }

    # Add overall summary
    total_correct = len(all_correct_data[metrics_list[0]]) if metrics_list else 0
    total_incorrect = len(all_incorrect_data[metrics_list[0]]) if metrics_list else 0
    total_points = total_correct + total_incorrect

    export_data["summary"] = {
        "total_questions": total_points,
        "correct_answers": total_correct,
        "incorrect_answers": total_incorrect,
        "overall_accuracy": float(100.0 * total_correct / total_points)
        if total_points > 0
        else 0.0,
        "selection_method": "percentile",
        "percentile_used": percentile,
        "best_metrics": sorted(
            [
                (metric, data["uncertainty_effectiveness"]["effect_size"])
                for metric, data in export_data["metrics"].items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:3],
    }

    # Export to JSON
    json_file = output_file.with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Percentile-based data exported to: {json_file}")


def analyze_and_plot_percentile(run_ids: List[str], percentile: float = 15.0) -> None:
    """Analyze uncertainty metrics and create violin plots with percentile-based selection."""
    console = Console()
    console.print(
        f"[bold cyan]Creating percentile-based violin plots ({percentile}%)[/bold cyan]"
    )
    console.print(f"Analyzing {len(run_ids)} runs: {', '.join(run_ids)}")

    # All available uncertainty metrics
    metrics_list = [
        "agreement_ratio",
        "entropy_freq",
        "entropy_weighted",
        "consensus_support",
        "prm_max",
        "prm_mean",
        "prm_std",
        "prm_margin",
        "prm_top_frac",
    ]

    # Collect all data across runs
    all_correct_data = {metric: [] for metric in metrics_list}
    all_incorrect_data = {metric: [] for metric in metrics_list}
    all_data = {metric: [] for metric in metrics_list}

    for run_id in run_ids:
        out_file = Path("./output") / run_id / "inference_output.jsonl"
        records = load_jsonl(out_file)

        print(f"Processing {run_id}: {len(records)} questions")

        for rec in records:
            qa = _get_question_answer_from_record(rec)
            metrics = _compute_uncertainty_metrics(rec)

            # Extract all available metrics
            for metric in metrics_list:
                metric_value = metrics.get(metric, 0.0)

                # Add to all_data for threshold calculation
                all_data[metric].append(metric_value)

                # Separate by correctness
                if qa.is_correct:
                    all_correct_data[metric].append(metric_value)
                else:
                    all_incorrect_data[metric].append(metric_value)

    # Calculate percentile-based thresholds
    thresholds = calculate_percentile_thresholds(
        all_data, all_correct_data, all_incorrect_data, percentile
    )
    correctness_analysis = analyze_percentile_correctness(
        all_correct_data, all_incorrect_data, thresholds
    )

    # Calculate F1 scores for ordering
    f1_scores = calculate_f1_scores_percentile(
        all_correct_data, all_incorrect_data, all_data, percentile
    )

    # Create split violin plots
    output_file = create_percentile_violin_plots(
        all_correct_data,
        all_incorrect_data,
        run_ids,
        thresholds,
        correctness_analysis,
        percentile,
        f1_scores,
    )

    # Export data
    export_percentile_data(
        output_file,
        run_ids,
        thresholds,
        correctness_analysis,
        all_correct_data,
        all_incorrect_data,
        list(thresholds.keys()),
        percentile,
    )

    # Print analysis
    console.print(
        f"\n[bold green]Percentile-Based Threshold Analysis ({percentile}% selection)[/bold green]"
    )
    console.print("[bold cyan]Metrics ordered by F1 score (descending):[/bold cyan]")

    # Sort metrics for display by F1 score
    sorted_metrics = sorted(f1_scores.keys(), key=lambda x: f1_scores[x], reverse=True)
    total_points = len(all_data["agreement_ratio"])
    for metric in sorted_metrics:
        if metric not in thresholds:
            continue

        direction = (
            "≤"
            if is_low_uncertainty_metric(metric, all_correct_data, all_incorrect_data)
            else "≥"
        )
        correct_mean = (
            np.mean(all_correct_data[metric]) if all_correct_data[metric] else 0.0
        )
        incorrect_mean = (
            np.mean(all_incorrect_data[metric]) if all_incorrect_data[metric] else 0.0
        )

        analysis = correctness_analysis[metric]
        selection_accuracy = (
            100 * analysis["incorrect_selected"] / analysis["total_selected"]
            if analysis["total_selected"] > 0
            else 0
        )

        f1_score = f1_scores.get(metric, 0.0)
        console.print(
            f"\n[cyan]{metric.replace('_', ' ').title()} (F1: {f1_score:.3f})[/cyan]:"
        )
        console.print(f"  Threshold: {direction} {thresholds[metric]:.4f}")
        console.print(
            f"  Means: Correct={correct_mean:.3f}, Incorrect={incorrect_mean:.3f}"
        )
        console.print(
            f"  Selected: {analysis['total_selected']} ({100 * analysis['total_selected'] / total_points:.1f}%)"
        )
        console.print(f"  Accuracy of selection: {selection_accuracy:.1f}% incorrect")

    # Print summary
    total_correct = len(all_correct_data["agreement_ratio"])
    total_incorrect = len(all_incorrect_data["agreement_ratio"])
    total = total_correct + total_incorrect
    accuracy = 100.0 * total_correct / total

    console.print("\n[bold]Overall Summary:[/bold]")
    console.print(
        f"  Total: {total} questions ({total_correct} correct, {total_incorrect} incorrect)"
    )
    console.print(f"  Overall accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    # Use the same runs as the original analysis
    run_ids = ["gfw8x07r", "77pyab58", "tqfyvf5w"]
    run_ids = [run_ids[0]]

    # Create plots with the configured percentile selection
    analyze_and_plot_percentile(run_ids, percentile=PERCENTILE)
