#!/usr/bin/env python3
"""
Ensemble analysis of 3 specific uncertainty metrics with fixed thresholds.
Focus on recall performance for incorrect answers.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from rich.console import Console
from rich.table import Table

from sal.utils.runs import fusion_base_runs_best
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    load_jsonl,
)

# Fixed thresholds as specified
FIXED_THRESHOLDS = {
    "consensus_support": {"threshold": 0.5006, "direction": "<="},
    "agreement_ratio": {"threshold": 0.5000, "direction": "<="},
    "entropy_freq": {"threshold": 1.0000, "direction": ">="},
}

METRICS = list(FIXED_THRESHOLDS.keys())


def load_data_from_runs(
    run_ids: List[str],
) -> tuple[
    Dict[str, List[float]],
    Dict[str, List[float]],
    List[str],
    List[str],
    List[str],
]:
    """Load data from runs and separate by correctness.

    Returns:
        correct_data, incorrect_data: metric -> values
        correct_ids, incorrect_ids: original question ids (run_id_index) per subset
        question_ids: all question ids (unused by analysis but kept for reference)
    """
    correct_data = {metric: [] for metric in METRICS}
    incorrect_data = {metric: [] for metric in METRICS}
    question_ids: List[str] = []
    correct_ids: List[str] = []
    incorrect_ids: List[str] = []

    console = Console()
    console.print(f"[bold cyan]Loading data from {len(run_ids)} runs[/bold cyan]")

    for run_id in run_ids:
        out_file = Path("./output") / run_id / "inference_output.jsonl"
        records = load_jsonl(out_file)
        console.print(f"Processing {run_id}: {len(records)} questions")

        for idx, rec in enumerate(records):
            qa = _get_question_answer_from_record(rec)
            metrics = _compute_uncertainty_metrics(rec)

            question_id = f"{run_id}_{idx}"
            question_ids.append(question_id)

            for metric in METRICS:
                metric_value = metrics.get(metric, 0.0)

                if qa.is_correct:
                    correct_data[metric].append(metric_value)
                else:
                    incorrect_data[metric].append(metric_value)

            if qa.is_correct:
                correct_ids.append(question_id)
            else:
                incorrect_ids.append(question_id)

    return correct_data, incorrect_data, correct_ids, incorrect_ids, question_ids


def apply_threshold(metric: str, values: List[float]) -> List[bool]:
    """Apply fixed threshold to metric values."""
    threshold_info = FIXED_THRESHOLDS[metric]
    threshold = threshold_info["threshold"]
    direction = threshold_info["direction"]

    if direction == "<=":
        return [val <= threshold for val in values]
    else:  # direction == ">="
        return [val >= threshold for val in values]


def get_selected_questions(
    metric: str,
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    correct_ids: List[str],
    incorrect_ids: List[str],
) -> tuple[Set[str], int, int]:
    """Get questions selected by a metric using fixed threshold."""
    selected_indices = set()

    # Apply threshold to correct answers
    correct_selected = apply_threshold(metric, correct_data[metric])
    for idx, is_selected in enumerate(correct_selected):
        if is_selected:
            selected_indices.add(f"correct_{correct_ids[idx]}")

    # Apply threshold to incorrect answers
    incorrect_selected = apply_threshold(metric, incorrect_data[metric])
    for idx, is_selected in enumerate(incorrect_selected):
        if is_selected:
            selected_indices.add(f"incorrect_{incorrect_ids[idx]}")

    correct_count = sum(correct_selected)
    incorrect_count = sum(incorrect_selected)

    return selected_indices, correct_count, incorrect_count


def calculate_recall_metrics(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    correct_ids: List[str],
    incorrect_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Calculate recall and other metrics for each individual metric."""
    total_correct = len(correct_data[METRICS[0]])
    total_incorrect = len(incorrect_data[METRICS[0]])

    results = {}

    for metric in METRICS:
        _, correct_selected, incorrect_selected = get_selected_questions(
            metric, correct_data, incorrect_data, correct_ids, incorrect_ids
        )

        # Calculate confusion matrix
        tp = incorrect_selected  # True positives: incorrect answers correctly flagged
        fp = correct_selected  # False positives: correct answers incorrectly flagged
        fn = (
            total_incorrect - incorrect_selected
        )  # False negatives: incorrect answers missed
        tn = (
            total_correct - correct_selected
        )  # True negatives: correct answers not flagged

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[metric] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_selected": tp + fp,
        }

    return results


def analyze_ensemble_combinations(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    correct_ids: List[str],
    incorrect_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Analyze all possible combinations of the 3 metrics."""
    total_correct = len(correct_data[METRICS[0]])
    total_incorrect = len(incorrect_data[METRICS[0]])

    # Get individual metric selections
    individual_selections = {}
    for metric in METRICS:
        selections, _, _ = get_selected_questions(
            metric, correct_data, incorrect_data, correct_ids, incorrect_ids
        )
        individual_selections[metric] = selections

    # Define all combinations
    combinations = {
        "consensus_support": [individual_selections["consensus_support"]],
        "agreement_ratio": [individual_selections["agreement_ratio"]],
        "entropy_freq": [individual_selections["entropy_freq"]],
        "consensus_support ∩ agreement_ratio": [
            individual_selections["consensus_support"]
            & individual_selections["agreement_ratio"]
        ],
        "consensus_support ∩ entropy_freq": [
            individual_selections["consensus_support"]
            & individual_selections["entropy_freq"]
        ],
        "agreement_ratio ∩ entropy_freq": [
            individual_selections["agreement_ratio"]
            & individual_selections["entropy_freq"]
        ],
        "all_three_intersection": [
            individual_selections["consensus_support"]
            & individual_selections["agreement_ratio"]
            & individual_selections["entropy_freq"]
        ],
        "any_one_union": [
            individual_selections["consensus_support"]
            | individual_selections["agreement_ratio"]
            | individual_selections["entropy_freq"]
        ],
    }

    results = {}

    for combo_name, selections_list in combinations.items():
        selected_questions = selections_list[0]  # Get the set of selected questions

        # Count correct vs incorrect in selection
        correct_selected = len(
            [q for q in selected_questions if q.startswith("correct_")]
        )
        incorrect_selected = len(
            [q for q in selected_questions if q.startswith("incorrect_")]
        )

        # Calculate confusion matrix
        tp = incorrect_selected  # True positives: incorrect answers correctly flagged
        fp = correct_selected  # False positives: correct answers incorrectly flagged
        fn = (
            total_incorrect - incorrect_selected
        )  # False negatives: incorrect answers missed
        tn = (
            total_correct - correct_selected
        )  # True negatives: correct answers not flagged

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[combo_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_selected": tp + fp,
        }

    return results


def create_detailed_analysis_table(
    individual_results: Dict[str, Dict[str, float]],
    ensemble_results: Dict[str, Dict[str, float]],
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
):
    """Create a comprehensive analysis table."""
    console = Console()

    total_correct = len(correct_data[METRICS[0]])
    total_incorrect = len(incorrect_data[METRICS[0]])
    total_questions = total_correct + total_incorrect

    console.print(f"\n[bold cyan]FIXED THRESHOLD ENSEMBLE ANALYSIS[/bold cyan]")
    console.print(
        f"Dataset: [green]{total_correct}[/green] correct, [red]{total_incorrect}[/red] incorrect, [blue]{total_questions}[/blue] total"
    )
    console.print(
        f"Baseline accuracy: [yellow]{total_correct / total_questions * 100:.1f}%[/yellow]\n"
    )

    # Print fixed thresholds
    console.print("[bold]Fixed Thresholds:[/bold]")
    for metric, info in FIXED_THRESHOLDS.items():
        console.print(
            f"• {metric.replace('_', ' ').title()}: [yellow]{info['direction']} {info['threshold']:.4f}[/yellow]"
        )
    console.print()

    # Create comprehensive table
    table = Table(
        title="Individual Metrics and Ensemble Combinations Analysis",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Selected", justify="right", style="blue")
    table.add_column("% of Total", justify="right", style="dim blue")
    table.add_column("Correct Sel.", justify="right", style="red")
    table.add_column("Incorrect Sel.", justify="right", style="green")
    table.add_column("Precision", justify="right", style="bright_blue")
    table.add_column("Recall", justify="right", style="bright_green")
    table.add_column("F1", justify="right", style="bright_magenta")

    # Combine all results for sorting
    all_results = {}
    all_results.update(individual_results)
    all_results.update(ensemble_results)

    # Sort by recall (descending)
    sorted_methods = sorted(
        all_results.items(), key=lambda x: x[1]["recall"], reverse=True
    )

    for method_name, result in sorted_methods:
        # Color-code F1 and recall scores
        f1_value = result["f1"]
        recall_value = result["recall"]

        if f1_value >= 0.5:
            f1_style = "[bright_green]"
        elif f1_value >= 0.3:
            f1_style = "[yellow]"
        else:
            f1_style = "[red]"

        if recall_value >= 0.7:
            recall_style = "[bright_green]"
        elif recall_value >= 0.5:
            recall_style = "[yellow]"
        else:
            recall_style = "[red]"

        # Format method name
        if method_name in METRICS:
            display_name = f"[bold]{method_name.replace('_', ' ').title()}[/bold]"
        else:
            display_name = method_name.replace("_", " ").title()

        percentage = (result["total_selected"] / total_questions) * 100

        table.add_row(
            display_name,
            str(result["total_selected"]),
            f"{percentage:.1f}%",
            str(result["fp"]),  # fp = correct selected
            str(result["tp"]),  # tp = incorrect selected
            f"{result['precision']:.3f}",
            f"{recall_style}{result['recall']:.3f}[/{recall_style.strip('[]')}]",
            f"{f1_style}{result['f1']:.3f}[/{f1_style.strip('[]')}]",
        )

    console.print(table)

    # Add detailed recall analysis
    console.print("\n[bold]Recall Analysis Summary:[/bold]")
    console.print(f"Total incorrect answers to detect: [red]{total_incorrect}[/red]")
    console.print()

    best_recall = max(result["recall"] for result in all_results.values())
    best_method = next(
        method
        for method, result in all_results.items()
        if result["recall"] == best_recall
    )

    console.print(
        f"🏆 [bold]Best Recall:[/bold] [bright_green]{best_method.replace('_', ' ').title()}[/bright_green] ({best_recall:.3f})"
    )
    console.print(
        f"   Detects [green]{all_results[best_method]['tp']}[/green] out of [red]{total_incorrect}[/red] incorrect answers"
    )
    console.print()

    # Compare individual vs ensemble performance
    individual_recalls = [individual_results[metric]["recall"] for metric in METRICS]
    best_individual_recall = max(individual_recalls)
    union_recall = ensemble_results["any_one_union"]["recall"]
    intersection_recall = ensemble_results["all_three_intersection"]["recall"]

    console.print("[bold]Individual vs Ensemble Comparison:[/bold]")
    console.print(
        f"• Best individual recall: [bright_blue]{best_individual_recall:.3f}[/bright_blue]"
    )
    console.print(
        f"• Union (any one) recall: [bright_green]{union_recall:.3f}[/bright_green] ({union_recall - best_individual_recall:+.3f})"
    )
    console.print(
        f"• Intersection (all three) recall: [bright_red]{intersection_recall:.3f}[/bright_red] ({intersection_recall - best_individual_recall:+.3f})"
    )

    # Recommendation
    if union_recall > best_individual_recall + 0.05:
        console.print(
            f"\n[bold]Recommendation:[/bold] [green]Use union ensemble[/green] - significant recall improvement"
        )
    elif union_recall > best_individual_recall + 0.02:
        console.print(
            f"\n[bold]Recommendation:[/bold] [yellow]Consider union ensemble[/yellow] - moderate recall improvement"
        )
    else:
        console.print(
            f"\n[bold]Recommendation:[/bold] [red]Stick with best individual metric[/red] - minimal ensemble benefit"
        )


def analyze_overlap_patterns(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    correct_ids: List[str],
    incorrect_ids: List[str],
):
    """Analyze overlap patterns between the three metrics."""
    console = Console()

    # Get individual metric selections
    individual_selections = {}
    for metric in METRICS:
        selections, correct_count, incorrect_count = get_selected_questions(
            metric, correct_data, incorrect_data, correct_ids, incorrect_ids
        )
        individual_selections[metric] = {
            "selections": selections,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
        }

    console.print("\n[bold cyan]OVERLAP PATTERN ANALYSIS[/bold cyan]")

    # Create overlap table
    overlap_table = Table(
        title="Pairwise Overlap Analysis",
        show_header=True,
        header_style="bold magenta",
    )

    overlap_table.add_column("Metric Pair", style="cyan")
    overlap_table.add_column("Overlap Count", justify="right", style="green")
    overlap_table.add_column("Overlap %", justify="right", style="yellow")
    overlap_table.add_column("Jaccard Index", justify="right", style="bright_magenta")
    overlap_table.add_column("Correct in Overlap", justify="right", style="red")
    overlap_table.add_column("Incorrect in Overlap", justify="right", style="green")

    # Calculate pairwise overlaps
    metric_pairs = [
        ("consensus_support", "agreement_ratio"),
        ("consensus_support", "entropy_freq"),
        ("agreement_ratio", "entropy_freq"),
    ]

    for metric_a, metric_b in metric_pairs:
        set_a = individual_selections[metric_a]["selections"]
        set_b = individual_selections[metric_b]["selections"]

        overlap = set_a & set_b
        union = set_a | set_b

        overlap_count = len(overlap)
        union_count = len(union)
        jaccard = overlap_count / union_count if union_count > 0 else 0

        # Count correct vs incorrect in overlap
        correct_in_overlap = len([q for q in overlap if q.startswith("correct_")])
        incorrect_in_overlap = len([q for q in overlap if q.startswith("incorrect_")])

        # Calculate overlap percentage relative to smaller set
        smaller_set_size = min(len(set_a), len(set_b))
        overlap_pct = (
            (overlap_count / smaller_set_size) * 100 if smaller_set_size > 0 else 0
        )

        # Color code Jaccard index
        if jaccard >= 0.7:
            jaccard_style = "[red]"  # High overlap
        elif jaccard >= 0.4:
            jaccard_style = "[yellow]"  # Medium overlap
        else:
            jaccard_style = "[green]"  # Low overlap

        pair_name = f"{metric_a.replace('_', ' ').title()[:12]} × {metric_b.replace('_', ' ').title()[:12]}"

        overlap_table.add_row(
            pair_name,
            str(overlap_count),
            f"{overlap_pct:.1f}%",
            f"{jaccard_style}{jaccard:.3f}[/{jaccard_style.strip('[]')}]",
            str(correct_in_overlap),
            str(incorrect_in_overlap),
        )

    console.print(overlap_table)

    # Three-way analysis
    all_three = (
        individual_selections["consensus_support"]["selections"]
        & individual_selections["agreement_ratio"]["selections"]
        & individual_selections["entropy_freq"]["selections"]
    )

    console.print(f"\n[bold]Three-way Overlap:[/bold]")
    console.print(
        f"• Questions selected by ALL three metrics: [blue]{len(all_three)}[/blue]"
    )

    if len(all_three) > 0:
        correct_all_three = len([q for q in all_three if q.startswith("correct_")])
        incorrect_all_three = len([q for q in all_three if q.startswith("incorrect_")])
        precision_all_three = incorrect_all_three / len(all_three)

        console.print(f"• Correct in three-way overlap: [red]{correct_all_three}[/red]")
        console.print(
            f"• Incorrect in three-way overlap: [green]{incorrect_all_three}[/green]"
        )
        console.print(
            f"• Three-way precision: [bright_blue]{precision_all_three:.3f}[/bright_blue]"
        )


def main():
    """Main analysis function."""
    console = Console()

    # Get the same runs as selection_2.py
    run_ids = fusion_base_runs_best()
    console.print(f"[bold]Analyzing runs:[/bold] {', '.join(run_ids)}")

    # Load data
    correct_data, incorrect_data, correct_ids, incorrect_ids, question_ids = (
        load_data_from_runs(run_ids)
    )

    # Calculate individual metric results
    individual_results = calculate_recall_metrics(
        correct_data, incorrect_data, correct_ids, incorrect_ids
    )

    # Calculate ensemble combinations
    ensemble_results = analyze_ensemble_combinations(
        correct_data, incorrect_data, correct_ids, incorrect_ids
    )

    # Create detailed analysis table
    create_detailed_analysis_table(
        individual_results, ensemble_results, correct_data, incorrect_data
    )

    # Analyze overlap patterns
    analyze_overlap_patterns(correct_data, incorrect_data, correct_ids, incorrect_ids)

    console.print("\n[bold green]Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
