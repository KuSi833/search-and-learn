#!/usr/bin/env python3
"""
Simple script to count True/False answers for the fusion base runs.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from sal.utils.runs import fusion_base_runs_best
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    load_jsonl,
)


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

            # Separate by correctness
            if qa.is_correct:
                all_correct_data["agreement_ratio"].append(agreement_ratio)
                all_correct_data["entropy_freq"].append(entropy_freq)
                all_correct_data["group_top_frac"].append(group_top_frac)
            else:
                all_incorrect_data["agreement_ratio"].append(agreement_ratio)
                all_incorrect_data["entropy_freq"].append(entropy_freq)
                all_incorrect_data["group_top_frac"].append(group_top_frac)

    # Create violin plots
    create_violin_plots(all_correct_data, all_incorrect_data, run_ids)

    # Print summary
    total_correct = len(all_correct_data["agreement_ratio"])
    total_incorrect = len(all_incorrect_data["agreement_ratio"])
    total = total_correct + total_incorrect
    accuracy = 100.0 * total_correct / total

    print(f"\nOverall Summary:")
    print(f"  Correct: {total_correct}, Incorrect: {total_incorrect}")
    print(f"  Accuracy: {accuracy:.1f}%")


def create_violin_plots(correct_data, incorrect_data, run_ids):
    """Create violin plots for the 3 uncertainty metrics."""
    metrics = ["agreement_ratio", "entropy_freq", "group_top_frac"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Prepare data for violin plot
        data_to_plot = [correct_data[metric], incorrect_data[metric]]

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

        # Customize plot with counts
        correct_count = len(correct_data[metric])
        incorrect_count = len(incorrect_data[metric])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(
            [f"Correct\n(n={correct_count})", f"Incorrect\n(n={incorrect_count})"]
        )
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
        ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")
        ax.grid(True, alpha=0.3)

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
