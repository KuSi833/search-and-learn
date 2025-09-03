#!/usr/bin/env python3
"""
Simple script to count True/False answers for the fusion base runs.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from sal.utils.runs import fusion_base_runs_best
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    load_jsonl,
)


def calculate_thresholds(all_data: Dict[str, List[float]]) -> Dict[str, float]:
    """Calculate thresholds for each metric to capture 10% of data points.

    For agreement_ratio and group_top_frac: use 10th percentile (lower threshold)
    For entropy_freq: use 90th percentile (upper threshold)
    """
    thresholds = {}

    # Combine correct and incorrect data for threshold calculation
    combined_data = {}
    for metric in all_data.keys():
        combined_data[metric] = all_data[metric]

    # Calculate thresholds
    thresholds["agreement_ratio"] = np.percentile(combined_data["agreement_ratio"], 10)
    thresholds["entropy_freq"] = np.percentile(combined_data["entropy_freq"], 90)
    thresholds["group_top_frac"] = np.percentile(combined_data["group_top_frac"], 10)

    return thresholds


def analyze_threshold_correctness(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, int]]:
    """Analyze how many correct vs incorrect answers fall within each threshold."""
    analysis = {}

    for metric in thresholds.keys():
        if metric in ["agreement_ratio", "group_top_frac"]:
            # Count values <= threshold
            correct_selected = sum(
                1 for x in correct_data[metric] if x <= thresholds[metric]
            )
            incorrect_selected = sum(
                1 for x in incorrect_data[metric] if x <= thresholds[metric]
            )
        else:  # entropy_freq
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


def analyze_and_plot(run_ids: List[str]) -> None:
    """Analyze uncertainty metrics and create violin plots."""
    print(f"Analyzing {len(run_ids)} runs: {', '.join(run_ids)}")

    # Collect all data across runs
    all_correct_data = {"agreement_ratio": [], "entropy_freq": [], "group_top_frac": []}
    all_incorrect_data = {
        "agreement_ratio": [],
        "entropy_freq": [],
        "group_top_frac": [],
    }
    all_data = {"agreement_ratio": [], "entropy_freq": [], "group_top_frac": []}

    for run_id in run_ids:
        out_file = Path("./output") / run_id / "inference_output.jsonl"
        records = load_jsonl(out_file)

        print(f"Processing {run_id}: {len(records)} questions")

        for rec in records:
            qa = _get_question_answer_from_record(rec)
            metrics = _compute_uncertainty_metrics(rec)

            # Extract the 3 core metrics
            agreement_ratio = metrics["agreement_ratio"]
            entropy_freq = metrics["entropy_freq"]
            group_top_frac = metrics["group_top_frac"]

            # Add to all_data for threshold calculation
            all_data["agreement_ratio"].append(agreement_ratio)
            all_data["entropy_freq"].append(entropy_freq)
            all_data["group_top_frac"].append(group_top_frac)

            # Separate by correctness
            if qa.is_correct:
                all_correct_data["agreement_ratio"].append(agreement_ratio)
                all_correct_data["entropy_freq"].append(entropy_freq)
                all_correct_data["group_top_frac"].append(group_top_frac)
            else:
                all_incorrect_data["agreement_ratio"].append(agreement_ratio)
                all_incorrect_data["entropy_freq"].append(entropy_freq)
                all_incorrect_data["group_top_frac"].append(group_top_frac)

    # Calculate thresholds for 10% selection
    thresholds = calculate_thresholds(all_data)
    correctness_analysis = analyze_threshold_correctness(
        all_correct_data, all_incorrect_data, thresholds
    )

    # Create violin plots with threshold lines
    create_violin_plots(
        all_correct_data, all_incorrect_data, run_ids, thresholds, correctness_analysis
    )

    # Print threshold analysis
    total_points = len(all_data["agreement_ratio"])
    print(f"\nThreshold Analysis (targeting 10% of {total_points} data points):")

    for metric in ["agreement_ratio", "entropy_freq", "group_top_frac"]:
        direction = "<=" if metric in ["agreement_ratio", "group_top_frac"] else ">="
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


def create_violin_plots(
    correct_data, incorrect_data, run_ids, thresholds, correctness_analysis
):
    """Create violin plots for the 3 uncertainty metrics with threshold lines."""
    metrics = ["agreement_ratio", "entropy_freq", "group_top_frac"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Create violin plot with proportional widths
        correct_count = len(correct_data[metric])
        incorrect_count = len(incorrect_data[metric])
        total_count = correct_count + incorrect_count

        # Scale widths by proportion of data
        correct_width = 0.8 * (correct_count / total_count)
        incorrect_width = 0.8 * (incorrect_count / total_count)

        parts = ax.violinplot(
            [correct_data[metric]], positions=[1], widths=[correct_width]
        )
        parts["bodies"][0].set_facecolor("#2ca02c")
        parts["bodies"][0].set_alpha(0.7)

        parts2 = ax.violinplot(
            [incorrect_data[metric]], positions=[2], widths=[incorrect_width]
        )
        parts2["bodies"][0].set_facecolor("#d62728")
        parts2["bodies"][0].set_alpha(0.7)

        # Add threshold line
        threshold_value = thresholds[metric]
        ax.axhline(
            y=threshold_value,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Threshold: {threshold_value:.3f}",
        )

        # Customize plot with counts and threshold info
        correct_count = len(correct_data[metric])
        incorrect_count = len(incorrect_data[metric])
        analysis = correctness_analysis[metric]

        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"Correct\n(n={correct_count})", f"Incorrect\n(n={incorrect_count})"]
        )
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}")

        # Enhanced title with threshold info
        direction = "<=" if metric in ["agreement_ratio", "group_top_frac"] else ">="
        selected_correct = analysis["correct_selected"]
        selected_incorrect = analysis["incorrect_selected"]
        title = f"{metric.replace('_', ' ').title()} Distribution\nThreshold {direction} {threshold_value:.3f}: {selected_correct}C + {selected_incorrect}I = {selected_correct + selected_incorrect} total"
        ax.set_title(title, fontsize=10)

        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle(
        f"Uncertainty Metrics: Correct vs Incorrect Answers\nRuns: {', '.join(run_ids)}"
    )
    plt.tight_layout()

    # Save the plot
    output_dir = Path("figures/selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"violin_metrics_{'_'.join(run_ids)}.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")

    print(f"Violin plots saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    # run_ids = fusion_base_runs_best()
    run_ids = [fusion_base_runs_best()[0]]
    analyze_and_plot(run_ids)
