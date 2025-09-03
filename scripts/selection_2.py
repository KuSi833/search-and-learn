#!/usr/bin/env python3
"""
Simple script to count True/False answers for the fusion base runs.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from sal.utils.runs import fusion_base_runs_best
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    load_jsonl,
)


def calculate_thresholds(all_data: Dict[str, List[float]]) -> Dict[str, float]:
    """Calculate thresholds for each metric to capture 10% of data points.

    Low values indicate high uncertainty: agreement_ratio, group_top_frac, prm_max, prm_mean, prm_margin, prm_top_frac
    High values indicate high uncertainty: entropy_freq, entropy_weighted, prm_std
    """
    thresholds = {}

    # Combine correct and incorrect data for threshold calculation
    combined_data = {}
    for metric in all_data.keys():
        combined_data[metric] = all_data[metric]

    # Metrics where LOW values indicate HIGH uncertainty (use 10th percentile)
    low_uncertainty_metrics = [
        "agreement_ratio",
        "group_top_frac",
        "prm_max",
        "prm_mean",
        "prm_margin",
        "prm_top_frac",
    ]

    # Metrics where HIGH values indicate HIGH uncertainty (use 90th percentile)
    high_uncertainty_metrics = ["entropy_freq", "entropy_weighted", "prm_std"]

    # Calculate thresholds
    for metric in combined_data.keys():
        if metric in low_uncertainty_metrics:
            thresholds[metric] = np.percentile(combined_data[metric], 10)
        elif metric in high_uncertainty_metrics:
            thresholds[metric] = np.percentile(combined_data[metric], 90)
        else:
            # Default to low uncertainty behavior for unknown metrics
            thresholds[metric] = np.percentile(combined_data[metric], 10)

    return thresholds


def analyze_threshold_correctness(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, int]]:
    """Analyze how many correct vs incorrect answers fall within each threshold."""
    analysis = {}

    # Metrics where LOW values indicate HIGH uncertainty (use <= threshold)
    low_uncertainty_metrics = [
        "agreement_ratio",
        "group_top_frac",
        "prm_max",
        "prm_mean",
        "prm_margin",
        "prm_top_frac",
    ]

    # Metrics where HIGH values indicate HIGH uncertainty (use >= threshold)
    high_uncertainty_metrics = ["entropy_freq", "entropy_weighted", "prm_std"]

    for metric in thresholds.keys():
        if metric in low_uncertainty_metrics:
            # Count values <= threshold
            correct_selected = sum(
                1 for x in correct_data[metric] if x <= thresholds[metric]
            )
            incorrect_selected = sum(
                1 for x in incorrect_data[metric] if x <= thresholds[metric]
            )
        elif metric in high_uncertainty_metrics:
            # Count values >= threshold
            correct_selected = sum(
                1 for x in correct_data[metric] if x >= thresholds[metric]
            )
            incorrect_selected = sum(
                1 for x in incorrect_data[metric] if x >= thresholds[metric]
            )
        else:
            # Default to low uncertainty behavior
            correct_selected = sum(
                1 for x in correct_data[metric] if x <= thresholds[metric]
            )
            incorrect_selected = sum(
                1 for x in incorrect_data[metric] if x <= thresholds[metric]
            )

        analysis[metric] = {
            "correct_selected": correct_selected,
            "incorrect_selected": incorrect_selected,
            "total_selected": correct_selected + incorrect_selected,
        }

    return analysis


def analyze_and_plot(run_ids: List[str]) -> None:
    """Analyze uncertainty metrics and create violin plots."""
    print(f"Analyzing {len(run_ids)} runs: {', '.join(run_ids)}")

    # All available uncertainty metrics
    metrics_list = [
        "agreement_ratio",
        "entropy_freq",
        "group_top_frac",
        "entropy_weighted",
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

    # Calculate thresholds for 10% selection
    thresholds = calculate_thresholds(all_data)
    correctness_analysis = analyze_threshold_correctness(
        all_correct_data, all_incorrect_data, thresholds
    )

    # Create cumulative line charts with threshold lines
    create_cumulative_line_charts(
        all_correct_data, all_incorrect_data, run_ids, thresholds, correctness_analysis
    )

    # Create split violin plots
    create_split_violin_plots(
        all_correct_data, all_incorrect_data, run_ids, thresholds, correctness_analysis
    )

    # Print threshold analysis
    total_points = len(all_data["agreement_ratio"])
    print(f"\nThreshold Analysis (targeting 10% of {total_points} data points):")

    # Metrics where LOW values indicate HIGH uncertainty (use <= threshold)
    low_uncertainty_metrics = [
        "agreement_ratio",
        "group_top_frac",
        "prm_max",
        "prm_mean",
        "prm_margin",
        "prm_top_frac",
    ]

    for metric in metrics_list:
        if metric not in thresholds:
            continue

        direction = "<=" if metric in low_uncertainty_metrics else ">="
        analysis = correctness_analysis[metric]
        correct_pct = (
            100 * analysis["correct_selected"] / analysis["total_selected"]
            if analysis["total_selected"] > 0
            else 0
        )

        print(
            f"  {metric.replace('_', ' ').title()}: {direction} {thresholds[metric]:.4f}"
        )
        print(
            f"    Selected: {analysis['total_selected']} ({100 * analysis['total_selected'] / total_points:.1f}%)"
        )
        print(
            f"    Correct: {analysis['correct_selected']} ({correct_pct:.1f}% of selected)"
        )
        print(
            f"    Incorrect: {analysis['incorrect_selected']} ({100 - correct_pct:.1f}% of selected)"
        )
        print()

    # Print summary
    total_correct = len(all_correct_data["agreement_ratio"])
    total_incorrect = len(all_incorrect_data["agreement_ratio"])
    total = total_correct + total_incorrect
    accuracy = 100.0 * total_correct / total

    print("\nOverall Summary:")
    print(f"  Correct: {total_correct}, Incorrect: {total_incorrect}")
    print(f"  Accuracy: {accuracy:.1f}%")


def create_cumulative_line_charts(
    correct_data, incorrect_data, run_ids, thresholds, correctness_analysis
):
    """Create cumulative line charts for all uncertainty metrics."""
    metrics = list(thresholds.keys())
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

    # Metrics where LOW values indicate HIGH uncertainty
    low_uncertainty_metrics = [
        "agreement_ratio",
        "group_top_frac",
        "prm_max",
        "prm_mean",
        "prm_margin",
        "prm_top_frac",
    ]

    for i, metric in enumerate(metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]

        # Combine and sort data with labels
        combined_data = []
        for val in correct_data[metric]:
            combined_data.append((val, True))  # True = correct
        for val in incorrect_data[metric]:
            combined_data.append((val, False))  # False = incorrect

        # Sort data based on metric direction
        if metric in low_uncertainty_metrics:
            # For these metrics, we accumulate from low to high (low → high)
            combined_data.sort(key=lambda x: x[0])
            x_label = f"{metric.replace('_', ' ').title()} (low → high)"
        else:
            # For high uncertainty metrics, we accumulate from high to low (high → low)
            combined_data.sort(key=lambda x: x[0], reverse=True)
            x_label = f"{metric.replace('_', ' ').title()} (high → low)"

        # Calculate cumulative counts
        total_points = len(combined_data)
        metric_values = []  # Actual metric values for x-axis
        correct_counts = []
        incorrect_counts = []

        correct_cum = 0
        incorrect_cum = 0

        # Create cumulative data points
        for idx, (value, is_correct) in enumerate(combined_data):
            if is_correct:
                correct_cum += 1
            else:
                incorrect_cum += 1

            # Store points at regular intervals
            if idx % max(1, total_points // 100) == 0 or idx == total_points - 1:
                metric_values.append(value)  # Use actual metric value
                correct_counts.append(correct_cum / total_points)
                incorrect_counts.append(incorrect_cum / total_points)

        # Create separate line plots for correct and incorrect
        ax.plot(
            metric_values,
            correct_counts,
            color="#2ca02c",
            linewidth=2.5,
            label=f"Correct (n={len(correct_data[metric])})",
            alpha=0.9,
        )
        ax.plot(
            metric_values,
            incorrect_counts,
            color="#d62728",
            linewidth=2.5,
            label=f"Incorrect (n={len(incorrect_data[metric])})",
            alpha=0.9,
        )

        # Add threshold line
        threshold_value = thresholds[metric]
        analysis = correctness_analysis[metric]

        ax.axvline(
            x=threshold_value,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="10% Threshold",
        )

        # Customize plot with proper axis limits
        if metric_values:
            if metric in low_uncertainty_metrics:
                # For these metrics: low → high (left to right)
                ax.set_xlim(min(metric_values), max(metric_values))
            else:
                # For high uncertainty metrics: high → low (left to right, but values go from high to low)
                ax.set_xlim(max(metric_values), min(metric_values))

        ax.set_ylim(0, 1)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cumulative Fraction of Data")

        # Enhanced title with threshold info
        direction = "<=" if metric in low_uncertainty_metrics else ">="
        analysis = correctness_analysis[metric]
        selected_correct = analysis["correct_selected"]
        selected_incorrect = analysis["incorrect_selected"]
        total_selected = selected_correct + selected_incorrect

        title = f"{metric.replace('_', ' ').title()}\nThreshold {direction} {threshold_value:.3f}: {selected_correct}C + {selected_incorrect}I = {total_selected}"
        ax.set_title(title, fontsize=10)

        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    # Hide any unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row][col].set_visible(False)
        else:
            axes[col].set_visible(False)

    plt.suptitle(
        f"Cumulative Selection: Correct vs Incorrect Answers\nRuns: {', '.join(run_ids)}"
    )
    plt.tight_layout()

    # Save the plot
    output_dir = Path("figures/selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"cumulative_lines_{'_'.join(run_ids)}.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")

    print(f"Cumulative line charts saved to: {output_file}")
    plt.close()


def create_split_violin_plots(
    correct_data, incorrect_data, run_ids, thresholds, correctness_analysis
):
    """Create split violin plots with correct on left side, incorrect on right side."""
    metrics = list(thresholds.keys())
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

    # Metrics where LOW values indicate HIGH uncertainty
    low_uncertainty_metrics = [
        "agreement_ratio",
        "group_top_frac",
        "prm_max",
        "prm_mean",
        "prm_margin",
        "prm_top_frac",
    ]

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

            # Create two separate patches for inside and outside threshold

            # Split vertices based on threshold
            if metric in low_uncertainty_metrics:
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

            # Create two separate patches for inside and outside threshold

            # Split vertices based on threshold
            if metric in low_uncertainty_metrics:
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
        if metric in low_uncertainty_metrics:
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

        title = f"{metric.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=10)

        # Create enhanced legend with selection counts
        direction = "<=" if metric in low_uncertainty_metrics else ">="
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
            handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9
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

    plt.suptitle(
        f"Split Violin Plots: Correct vs Incorrect Distribution\nRuns: {', '.join(run_ids)}"
    )
    plt.tight_layout()

    # Save the plot
    output_dir = Path("figures/selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"split_violins_{'_'.join(run_ids)}.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")

    print(f"Split violin plots saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    # run_ids = fusion_base_runs_best()
    run_ids = [fusion_base_runs_best()[0]]
    analyze_and_plot(run_ids)
