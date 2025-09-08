#!/usr/bin/env python3
"""
Simple script to count True/False answers for the fusion base runs.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from sal.utils.runs import fusion_base_runs_best
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    load_jsonl,
)


def calculate_thresholds(
    all_data: Dict[str, List[float]],
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
) -> Dict[str, float]:
    """Calculate thresholds for each metric to capture 10% of data points.

    Direction is determined by comparing means: if incorrect answers have higher mean values,
    then high values indicate uncertainty (use 90th percentile). If incorrect answers have
    lower mean values, then low values indicate uncertainty (use 10th percentile).
    """
    thresholds = {}

    # Combine correct and incorrect data for threshold calculation
    combined_data = {}
    for metric in all_data.keys():
        combined_data[metric] = all_data[metric]

    # Calculate thresholds based on data-driven direction
    for metric in combined_data.keys():
        correct_mean = np.mean(correct_data[metric]) if correct_data[metric] else 0.0
        incorrect_mean = (
            np.mean(incorrect_data[metric]) if incorrect_data[metric] else 0.0
        )

        # If incorrect answers have higher mean values, high values indicate uncertainty
        if incorrect_mean > correct_mean:
            thresholds[metric] = np.percentile(combined_data[metric], 90)
        else:
            # If incorrect answers have lower mean values, low values indicate uncertainty
            thresholds[metric] = np.percentile(combined_data[metric], 10)

    return thresholds


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


def analyze_threshold_correctness(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, int]]:
    """Analyze how many correct vs incorrect answers fall within each threshold."""
    analysis = {}

    for metric in thresholds.keys():
        # Determine direction based on data
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


def calculate_f1_score_for_coverage(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    coverage_percent: float,
) -> Dict[str, Dict[str, float]]:
    """Calculate F1 scores for each metric at a given coverage percentage."""
    results = {}

    for metric in all_data.keys():
        # Calculate threshold for this coverage
        if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
            threshold = np.percentile(all_data[metric], coverage_percent)
            # Count values <= threshold (uncertain)
            correct_selected = sum(1 for x in correct_data[metric] if x <= threshold)
            incorrect_selected = sum(
                1 for x in incorrect_data[metric] if x <= threshold
            )
        else:
            threshold = np.percentile(all_data[metric], 100 - coverage_percent)
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

        # Calculate accuracy of selection (what % of selected items are incorrect)
        selection_accuracy = (
            incorrect_selected / (correct_selected + incorrect_selected)
            if (correct_selected + incorrect_selected) > 0
            else 0.0
        )

        results[metric] = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "selection_accuracy": selection_accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_selected": correct_selected + incorrect_selected,
        }

    return results


def calculate_f1_score_for_fixed_count(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    selection_count: int,
) -> Dict[str, Dict[str, float]]:
    """Calculate F1 scores for each metric with fixed count selection."""
    results = {}

    for metric in all_data.keys():
        # Get selected questions using fixed count
        selected_questions, threshold_used = get_selected_questions_fixed_count(
            metric, correct_data, incorrect_data, all_data, selection_count
        )

        # Count correct vs incorrect in selection
        correct_selected = len(
            [q for q in selected_questions if q.startswith("correct_")]
        )
        incorrect_selected = len(
            [q for q in selected_questions if q.startswith("incorrect_")]
        )

        # Use the actual threshold returned by the function
        threshold = threshold_used

        # Calculate confusion matrix values
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

        # Calculate accuracy of selection (what % of selected items are incorrect)
        selection_accuracy = (
            incorrect_selected / (correct_selected + incorrect_selected)
            if (correct_selected + incorrect_selected) > 0
            else 0.0
        )

        results[metric] = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "selection_accuracy": selection_accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total_selected": correct_selected + incorrect_selected,
        }

    return results


def print_coverage_analysis_table(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
):
    """Print a comprehensive table showing raw counts, precision, recall, and F1 scores for different coverage levels."""
    console = Console()
    coverage_levels = [10, 15, 20, 25, 30]

    # Calculate total counts for context
    total_correct = len(correct_data[list(correct_data.keys())[0]])
    total_incorrect = len(incorrect_data[list(incorrect_data.keys())[0]])
    total_all = total_correct + total_incorrect

    console.print("\n[bold cyan]COMPREHENSIVE METRIC ANALYSIS[/bold cyan]")
    console.print(
        f"Dataset: [green]{total_correct}[/green] correct, [red]{total_incorrect}[/red] incorrect, [blue]{total_all}[/blue] total answers\n"
    )

    # Calculate results for all coverage levels
    all_results = {}
    for coverage in coverage_levels:
        all_results[coverage] = calculate_f1_score_for_coverage(
            correct_data, incorrect_data, all_data, coverage
        )

    # Create rich table
    table = Table(
        title="Raw Counts, Precision, Recall, F1 by Selection Rate",
        show_header=True,
        header_style="bold magenta",
    )

    # Add columns
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Select%", justify="center", style="yellow")
    table.add_column("Direction", justify="center", style="white")
    table.add_column("Threshold", justify="right", style="white")
    table.add_column("Corr.Sel", justify="right", style="red")
    table.add_column("Incor.Sel", justify="right", style="green")
    table.add_column("Corr.NotSel", justify="right", style="dim red")
    table.add_column("Incor.NotSel", justify="right", style="dim green")
    table.add_column("Precision", justify="right", style="bright_blue")
    table.add_column("Recall", justify="right", style="bright_green")
    table.add_column("F1", justify="right", style="bright_magenta")

    # Get all metrics and sort them by best F1 score at 10% coverage
    metrics = list(all_results[10].keys())
    metrics_with_f1 = [(metric, all_results[10][metric]["f1"]) for metric in metrics]
    metrics_sorted = sorted(metrics_with_f1, key=lambda x: x[1], reverse=True)

    # Add rows to table
    for metric, best_f1 in metrics_sorted:
        for i, coverage in enumerate(coverage_levels):
            result = all_results[coverage][metric]

            metric_name = metric.replace("_", " ").title() if i == 0 else ""

            # Raw counts
            correct_selected = result["fp"]  # FP = correct answers flagged as uncertain
            incorrect_selected = result[
                "tp"
            ]  # TP = incorrect answers flagged as uncertain
            correct_not_selected = result["tn"]  # TN = correct answers not flagged
            incorrect_not_selected = result["fn"]  # FN = incorrect answers not flagged

            # Get direction for this metric
            direction = get_metric_direction(metric, correct_data, incorrect_data)

            # Color-code F1 scores
            f1_value = result["f1"]
            if f1_value >= 0.5:
                f1_style = "[bright_green]"
            elif f1_value >= 0.3:
                f1_style = "[yellow]"
            else:
                f1_style = "[red]"

            table.add_row(
                metric_name,
                f"{coverage}%",
                direction if i == 0 else "",
                f"{result['threshold']:.4f}",
                str(correct_selected),
                str(incorrect_selected),
                str(correct_not_selected),
                str(incorrect_not_selected),
                f"{result['precision']:.3f}",
                f"{result['recall']:.3f}",
                f"{f1_style}{result['f1']:.3f}[/{f1_style.strip('[]')}]",
            )

        # Add separator between metrics (empty row)
        if metric != metrics_sorted[-1][0]:  # Don't add after last metric
            table.add_row("", "", "", "", "", "", "", "", "", "")

    console.print(table)

    # Add legend
    console.print("\n[bold]Legend:[/bold]")
    console.print(
        "• [yellow]Select%[/yellow]: Percentage of data selected as 'uncertain' (based on percentile threshold)"
    )
    console.print(
        "• [red]Corr.Sel[/red]: Correct answers selected as uncertain (False Positives)"
    )
    console.print(
        "• [green]Incor.Sel[/green]: Incorrect answers selected as uncertain (True Positives)"
    )
    console.print(
        "• [dim red]Corr.NotSel[/dim red]: Correct answers not selected (True Negatives)"
    )
    console.print(
        "• [dim green]Incor.NotSel[/dim green]: Incorrect answers not selected (False Negatives)"
    )
    console.print(
        "• [bright_blue]Precision[/bright_blue]: TP/(TP+FP) = Incor.Sel/(Incor.Sel+Corr.Sel)"
    )
    console.print(
        "• [bright_green]Recall[/bright_green]: TP/(TP+FN) = Incor.Sel/(Incor.Sel+Incor.NotSel)"
    )
    console.print(
        "• [bright_magenta]F1[/bright_magenta]: Harmonic mean of precision and recall"
    )
    console.print(
        "• F1 Color coding: [bright_green]≥0.5 (Good)[/bright_green], [yellow]≥0.3 (Fair)[/yellow], [red]<0.3 (Poor)[/red]"
    )


def print_fixed_count_analysis_table(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
):
    """Print a comprehensive table showing raw counts, precision, recall, and F1 scores for fixed count selections."""
    console = Console()
    selection_counts = [50, 75, 100, 125, 150]

    # Calculate total counts for context
    total_correct = len(correct_data[list(correct_data.keys())[0]])
    total_incorrect = len(incorrect_data[list(incorrect_data.keys())[0]])
    total_all = total_correct + total_incorrect

    console.print("\n[bold cyan]FIXED COUNT METRIC ANALYSIS[/bold cyan]")
    console.print(
        f"Dataset: [green]{total_correct}[/green] correct, [red]{total_incorrect}[/red] incorrect, [blue]{total_all}[/blue] total answers\n"
    )

    # Calculate results for all selection counts
    all_results = {}
    for count in selection_counts:
        all_results[count] = calculate_f1_score_for_fixed_count(
            correct_data, incorrect_data, all_data, count
        )

    # Create rich table
    table = Table(
        title="Raw Counts, Precision, Recall, F1 by Fixed Selection Count",
        show_header=True,
        header_style="bold magenta",
    )

    # Add columns
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Count", justify="center", style="yellow")
    table.add_column("Direction", justify="center", style="white")
    table.add_column("Threshold", justify="right", style="white")
    table.add_column("Corr.Sel", justify="right", style="red")
    table.add_column("Incor.Sel", justify="right", style="green")
    table.add_column("Corr.NotSel", justify="right", style="dim red")
    table.add_column("Incor.NotSel", justify="right", style="dim green")
    table.add_column("Precision", justify="right", style="bright_blue")
    table.add_column("Recall", justify="right", style="bright_green")
    table.add_column("F1", justify="right", style="bright_magenta")

    # Get all metrics and sort them by best F1 score at 100 count
    metrics = list(all_results[100].keys())
    metrics_with_f1 = [(metric, all_results[100][metric]["f1"]) for metric in metrics]
    metrics_sorted = sorted(metrics_with_f1, key=lambda x: x[1], reverse=True)

    # Add rows to table
    for metric, best_f1 in metrics_sorted:
        for i, count in enumerate(selection_counts):
            result = all_results[count][metric]

            metric_name = metric.replace("_", " ").title() if i == 0 else ""

            # Raw counts
            correct_selected = result["fp"]  # FP = correct answers flagged as uncertain
            incorrect_selected = result[
                "tp"
            ]  # TP = incorrect answers flagged as uncertain
            correct_not_selected = result["tn"]  # TN = correct answers not flagged
            incorrect_not_selected = result["fn"]  # FN = incorrect answers not flagged

            # Color-code F1 scores
            f1_value = result["f1"]
            if f1_value >= 0.5:
                f1_style = "[bright_green]"
            elif f1_value >= 0.3:
                f1_style = "[yellow]"
            else:
                f1_style = "[red]"

            table.add_row(
                metric_name,
                str(count),
                f"{result['threshold']:.4f}",
                str(correct_selected),
                str(incorrect_selected),
                str(correct_not_selected),
                str(incorrect_not_selected),
                f"{result['precision']:.3f}",
                f"{result['recall']:.3f}",
                f"{f1_style}{result['f1']:.3f}[/{f1_style.strip('[]')}]",
            )

        # Add separator between metrics (empty row)
        if metric != metrics_sorted[-1][0]:  # Don't add after last metric
            table.add_row("", "", "", "", "", "", "", "", "", "")

    console.print(table)

    # Add legend
    console.print("\n[bold]Legend:[/bold]")
    console.print(
        "• [yellow]Count[/yellow]: Fixed number of questions selected as 'uncertain'"
    )
    console.print(
        "• [red]Corr.Sel[/red]: Correct answers selected as uncertain (False Positives)"
    )
    console.print(
        "• [green]Incor.Sel[/green]: Incorrect answers selected as uncertain (True Positives)"
    )
    console.print(
        "• [dim red]Corr.NotSel[/dim red]: Correct answers not selected (True Negatives)"
    )
    console.print(
        "• [dim green]Incor.NotSel[/dim green]: Incorrect answers not selected (False Negatives)"
    )
    console.print(
        "• [bright_blue]Precision[/bright_blue]: TP/(TP+FP) = Incor.Sel/(Incor.Sel+Corr.Sel)"
    )
    console.print(
        "• [bright_green]Recall[/bright_green]: TP/(TP+FN) = Incor.Sel/(Incor.Sel+Incor.NotSel)"
    )
    console.print(
        "• [bright_magenta]F1[/bright_magenta]: Harmonic mean of precision and recall"
    )
    console.print(
        "• F1 Color coding: [bright_green]≥0.5 (Good)[/bright_green], [yellow]≥0.3 (Fair)[/yellow], [red]<0.3 (Poor)[/red]"
    )


def get_selected_questions(
    metric: str,
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    selection_rate: float,
) -> tuple[set, float]:
    """Get the set of question indices selected by a metric at a given selection rate.

    Returns:
        tuple: (selected_indices, threshold_used)
    """
    # Calculate threshold for this selection rate
    if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
        threshold = np.percentile(all_data[metric], selection_rate)
        direction = "<="
        # Select values <= threshold (uncertain)
        selected_indices = set()
        idx = 0
        for val in correct_data[metric]:
            if val <= threshold:
                selected_indices.add(f"correct_{idx}")
            idx += 1
        idx = 0
        for val in incorrect_data[metric]:
            if val <= threshold:
                selected_indices.add(f"incorrect_{idx}")
            idx += 1
    else:
        threshold = np.percentile(all_data[metric], 100 - selection_rate)
        direction = ">="
        # Select values >= threshold (uncertain)
        selected_indices = set()
        idx = 0
        for val in correct_data[metric]:
            if val >= threshold:
                selected_indices.add(f"correct_{idx}")
            idx += 1
        idx = 0
        for val in incorrect_data[metric]:
            if val >= threshold:
                selected_indices.add(f"incorrect_{idx}")
            idx += 1

    print(
        f"    {metric}: {direction} {threshold:.4f} → {len(selected_indices)} questions selected"
    )
    return selected_indices, threshold


def get_selected_questions_fixed_count(
    metric: str,
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    selection_count: int,
) -> tuple[set, float]:
    """Get the set of question indices selected by a metric with fixed count selection.

    Returns:
        tuple: (selected_indices, threshold_used)
    """
    # Create list of (value, index, is_correct) tuples
    all_values_with_indices = []

    # Add correct data with indices
    for idx, val in enumerate(correct_data[metric]):
        all_values_with_indices.append((val, f"correct_{idx}", True))

    # Add incorrect data with indices
    for idx, val in enumerate(incorrect_data[metric]):
        all_values_with_indices.append((val, f"incorrect_{idx}", False))

    # Sort based on uncertainty direction
    if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
        # For these metrics, low values indicate high uncertainty - sort ascending
        all_values_with_indices.sort(key=lambda x: x[0])
        direction = "<="
    else:
        # For high uncertainty metrics, high values indicate high uncertainty - sort descending
        all_values_with_indices.sort(key=lambda x: x[0], reverse=True)
        direction = ">="

    # Select top N most uncertain questions
    selected_count = min(selection_count, len(all_values_with_indices))
    selected_indices = set()

    for i in range(selected_count):
        _, question_id, _ = all_values_with_indices[i]
        selected_indices.add(question_id)

    # The threshold is the value of the last selected item
    threshold = (
        all_values_with_indices[selected_count - 1][0] if selected_count > 0 else 0.0
    )

    print(
        f"    {metric}: {direction} {threshold:.4f} (rank {selected_count}) → {len(selected_indices)} questions selected"
    )
    return selected_indices, threshold


def export_venn_diagram_data(
    count: int,
    metric_names: List[str],
    selected_by_metric: Dict[str, set],
    three_way_overlap: set,
    overlaps: Dict,
):
    """Export detailed Venn diagram data broken down by true/false predictions."""
    console = Console()

    console.print(
        f"\n[bold green]═══ VENN DIAGRAM DATA EXPORT (N={count}) ═══[/bold green]"
    )
    console.print("Detailed breakdown for True/False prediction overlaps:\n")

    # Get the three metrics (assuming order: A, B, C)
    metric_A, metric_B, metric_C = metric_names[0], metric_names[1], metric_names[2]
    set_A, set_B, set_C = (
        selected_by_metric[metric_A],
        selected_by_metric[metric_B],
        selected_by_metric[metric_C],
    )

    # Calculate all 7 Venn diagram regions
    # Three-way overlap (111)
    ABC = three_way_overlap

    # Two-way overlaps only (110, 101, 011)
    AB_only = (set_A & set_B) - set_C  # A∩B - C
    AC_only = (set_A & set_C) - set_B  # A∩C - B
    BC_only = (set_B & set_C) - set_A  # B∩C - A

    # Individual only (100, 010, 001)
    A_only = set_A - set_B - set_C  # A - B - C
    B_only = set_B - set_A - set_C  # B - A - C
    C_only = set_C - set_A - set_B  # C - A - B

    regions = {
        "Three-way (ABC)": ABC,
        f"{metric_A} ∩ {metric_B} only": AB_only,
        f"{metric_A} ∩ {metric_C} only": AC_only,
        f"{metric_B} ∩ {metric_C} only": BC_only,
        f"{metric_A} only": A_only,
        f"{metric_B} only": B_only,
        f"{metric_C} only": C_only,
    }

    console.print(f"[bold]Metrics:[/bold] A={metric_A}, B={metric_B}, C={metric_C}")
    console.print(f"[bold]Selection Count:[/bold] {count} each\n")

    # Create detailed breakdown table
    breakdown_table = Table(
        title=f"Venn Diagram Regions Breakdown (N={count})",
        show_header=True,
        header_style="bold magenta",
    )

    breakdown_table.add_column("Region", style="cyan", no_wrap=True)
    breakdown_table.add_column("Total", justify="right", style="blue")
    breakdown_table.add_column("True Pred", justify="right", style="green")
    breakdown_table.add_column("False Pred", justify="right", style="red")
    breakdown_table.add_column("True %", justify="right", style="bright_green")
    breakdown_table.add_column("False %", justify="right", style="bright_red")

    # Calculate breakdowns for each region
    total_true_preds = 0
    total_false_preds = 0

    for region_name, question_set in regions.items():
        total_count = len(question_set)
        true_preds = len(
            [q for q in question_set if q.startswith("incorrect_")]
        )  # True predictions = correctly identified incorrect
        false_preds = len(
            [q for q in question_set if q.startswith("correct_")]
        )  # False predictions = incorrectly identified correct

        true_pct = (true_preds / total_count * 100) if total_count > 0 else 0
        false_pct = (false_preds / total_count * 100) if total_count > 0 else 0

        total_true_preds += true_preds
        total_false_preds += false_preds

        breakdown_table.add_row(
            region_name,
            str(total_count),
            str(true_preds),
            str(false_preds),
            f"{true_pct:.1f}%",
            f"{false_pct:.1f}%",
        )

    console.print(breakdown_table)

    # Summary statistics
    console.print(f"\n[bold]Summary for N={count}:[/bold]")
    console.print(f"• Total True Predictions: [green]{total_true_preds}[/green]")
    console.print(f"• Total False Predictions: [red]{total_false_preds}[/red]")
    console.print(
        f"• True/False Ratio: [yellow]{total_true_preds / total_false_preds:.2f}[/yellow]"
        if total_false_preds > 0
        else "• True/False Ratio: [yellow]∞[/yellow]"
    )

    # Three-way overlap analysis
    three_way_true = len([q for q in ABC if q.startswith("incorrect_")])
    three_way_false = len([q for q in ABC if q.startswith("correct_")])
    three_way_total = len(ABC)

    if three_way_total > 0:
        console.print(f"\n[bold]Three-way Overlap Analysis:[/bold]")
        console.print(f"• Total questions: [blue]{three_way_total}[/blue]")
        console.print(
            f"• True predictions: [green]{three_way_true}[/green] ({three_way_true / three_way_total * 100:.1f}%)"
        )
        console.print(
            f"• False predictions: [red]{three_way_false}[/red] ({three_way_false / three_way_total * 100:.1f}%)"
        )
        console.print(
            f"• Three-way True/False ratio: [yellow]{three_way_true / three_way_false:.2f}[/yellow]"
            if three_way_false > 0
            else "• Three-way True/False ratio: [yellow]∞[/yellow]"
        )

    # Raw data for copy-paste into visualization script
    console.print(f"\n[bold cyan]RAW DATA FOR VENN DIAGRAM (N={count}):[/bold cyan]")
    console.print("[dim]# Copy this data into your visualization script[/dim]")
    console.print(f"# Count: {count}")
    console.print(f"# Metrics: {metric_A}, {metric_B}, {metric_C}")
    console.print("")
    console.print("# True predictions (correctly identified uncertain)")
    console.print(f"true_ABC = {three_way_true}  # All three methods")
    console.print(
        f"true_AB_only = {len([q for q in AB_only if q.startswith('incorrect_')])}  # {metric_A} ∩ {metric_B} only"
    )
    console.print(
        f"true_AC_only = {len([q for q in AC_only if q.startswith('incorrect_')])}  # {metric_A} ∩ {metric_C} only"
    )
    console.print(
        f"true_BC_only = {len([q for q in BC_only if q.startswith('incorrect_')])}  # {metric_B} ∩ {metric_C} only"
    )
    console.print(
        f"true_A_only = {len([q for q in A_only if q.startswith('incorrect_')])}  # {metric_A} only"
    )
    console.print(
        f"true_B_only = {len([q for q in B_only if q.startswith('incorrect_')])}  # {metric_B} only"
    )
    console.print(
        f"true_C_only = {len([q for q in C_only if q.startswith('incorrect_')])}  # {metric_C} only"
    )
    console.print("")
    console.print("# False predictions (incorrectly identified uncertain)")
    console.print(f"false_ABC = {three_way_false}  # All three methods")
    console.print(
        f"false_AB_only = {len([q for q in AB_only if q.startswith('correct_')])}  # {metric_A} ∩ {metric_B} only"
    )
    console.print(
        f"false_AC_only = {len([q for q in AC_only if q.startswith('correct_')])}  # {metric_A} ∩ {metric_C} only"
    )
    console.print(
        f"false_BC_only = {len([q for q in BC_only if q.startswith('correct_')])}  # {metric_B} ∩ {metric_C} only"
    )
    console.print(
        f"false_A_only = {len([q for q in A_only if q.startswith('correct_')])}  # {metric_A} only"
    )
    console.print(
        f"false_B_only = {len([q for q in B_only if q.startswith('correct_')])}  # {metric_B} only"
    )
    console.print(
        f"false_C_only = {len([q for q in C_only if q.startswith('correct_')])}  # {metric_C} only"
    )


def analyze_ensemble_potential(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
):
    """Analyze overlap between top 3 metrics to determine ensemble potential."""
    console = Console()

    # Get top 3 metrics by F1 score at 10%
    results_10 = calculate_f1_score_for_coverage(
        correct_data, incorrect_data, all_data, 10
    )
    metrics_with_f1 = [
        (metric, results_10[metric]["f1"]) for metric in results_10.keys()
    ]
    top_3_metrics = sorted(metrics_with_f1, key=lambda x: x[1], reverse=True)[:3]

    console.print("\n[bold cyan]ENSEMBLE ANALYSIS - TOP 3 METRICS[/bold cyan]")
    console.print("Analyzing overlap between top 3 performing metrics:\n")

    for i, (metric, f1_score) in enumerate(top_3_metrics, 1):
        console.print(
            f"{i}. [cyan]{metric.replace('_', ' ').title()}[/cyan] (F1: {f1_score:.3f})"
        )

    # Analyze at 10% and 15% selection rates
    selection_rates = [10, 15]

    for rate in selection_rates:
        console.print(
            f"\n[bold yellow]═══ {rate}% Selection Rate Analysis ═══[/bold yellow]"
        )

        # Get selected questions for each metric
        selected_by_metric = {}
        thresholds_used = {}
        print(f"\n  Thresholds used for {rate}% selection:")
        for metric, _ in top_3_metrics:
            selected_by_metric[metric], thresholds_used[metric] = (
                get_selected_questions(
                    metric, correct_data, incorrect_data, all_data, rate
                )
            )

        # Calculate pairwise overlaps
        metric_names = [metric for metric, _ in top_3_metrics]

        # Create overlap table
        overlap_table = Table(
            title=f"Pairwise Overlap at {rate}% Selection",
            show_header=True,
            header_style="bold magenta",
        )
        overlap_table.add_column("Metric Pair", style="cyan")
        overlap_table.add_column("Overlap Count", justify="right", style="green")
        overlap_table.add_column("Overlap %", justify="right", style="yellow")
        overlap_table.add_column("Union Count", justify="right", style="blue")
        overlap_table.add_column(
            "Jaccard Index", justify="right", style="bright_magenta"
        )

        total_questions = len(correct_data[metric_names[0]]) + len(
            incorrect_data[metric_names[0]]
        )
        expected_selected = int(total_questions * rate / 100)

        # Calculate all pairwise overlaps
        overlaps = {}
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric_a, metric_b = metric_names[i], metric_names[j]
                set_a = selected_by_metric[metric_a]
                set_b = selected_by_metric[metric_b]

                overlap = set_a & set_b
                union = set_a | set_b
                overlap_count = len(overlap)
                union_count = len(union)
                overlap_pct = (
                    (overlap_count / expected_selected) * 100
                    if expected_selected > 0
                    else 0
                )
                jaccard = overlap_count / union_count if union_count > 0 else 0

                overlaps[(metric_a, metric_b)] = {
                    "overlap_count": overlap_count,
                    "overlap_pct": overlap_pct,
                    "union_count": union_count,
                    "jaccard": jaccard,
                }

                # Color code Jaccard index
                if jaccard >= 0.7:
                    jaccard_style = "[red]"  # High overlap - bad for ensemble
                elif jaccard >= 0.5:
                    jaccard_style = "[yellow]"  # Medium overlap
                else:
                    jaccard_style = "[green]"  # Low overlap - good for ensemble

                pair_name = f"{metric_a.replace('_', ' ').title()[:12]} × {metric_b.replace('_', ' ').title()[:12]}"

                overlap_table.add_row(
                    pair_name,
                    str(overlap_count),
                    f"{overlap_pct:.1f}%",
                    str(union_count),
                    f"{jaccard_style}{jaccard:.3f}[/{jaccard_style.strip('[]')}]",
                )

        console.print(overlap_table)

        # Calculate three-way overlap
        set_all = [selected_by_metric[metric] for metric, _ in top_3_metrics]
        three_way_overlap = set_all[0] & set_all[1] & set_all[2]
        three_way_union = set_all[0] | set_all[1] | set_all[2]

        console.print("\n[bold]Three-way Analysis:[/bold]")
        console.print(
            f"• Questions selected by ALL 3 metrics: [green]{len(three_way_overlap)}[/green]"
        )
        console.print(
            f"• Questions selected by ANY metric: [blue]{len(three_way_union)}[/blue]"
        )
        console.print(
            f"• Three-way overlap rate: [yellow]{len(three_way_overlap) / expected_selected * 100:.1f}%[/yellow]"
        )

        # Analyze incorrect vs correct in overlaps
        incorrect_in_overlap = len(
            [q for q in three_way_overlap if q.startswith("incorrect_")]
        )
        correct_in_overlap = len(
            [q for q in three_way_overlap if q.startswith("correct_")]
        )

        if len(three_way_overlap) > 0:
            console.print(
                f"• In 3-way overlap: [red]{incorrect_in_overlap}[/red] incorrect, [green]{correct_in_overlap}[/green] correct"
            )
            console.print(
                f"• 3-way overlap precision: [bright_blue]{incorrect_in_overlap / len(three_way_overlap) * 100:.1f}%[/bright_blue]"
            )

    # Ensemble recommendation
    console.print("\n[bold cyan]═══ ENSEMBLE RECOMMENDATION ═══[/bold cyan]")

    # Calculate average Jaccard index across all pairs at both rates
    all_jaccards = []
    for rate in [10, 15]:
        selected_by_metric = {}
        for metric, _ in top_3_metrics:
            selected_by_metric[metric], _ = get_selected_questions(
                metric, correct_data, incorrect_data, all_data, rate
            )

        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric_a, metric_b = metric_names[i], metric_names[j]
                set_a = selected_by_metric[metric_a]
                set_b = selected_by_metric[metric_b]
                overlap = len(set_a & set_b)
                union = len(set_a | set_b)
                jaccard = overlap / union if union > 0 else 0
                all_jaccards.append(jaccard)

    avg_jaccard = np.mean(all_jaccards)

    if avg_jaccard < 0.5:
        recommendation = "[green]HIGHLY RECOMMENDED[/green]"
        reason = "Low overlap suggests metrics capture different uncertainty patterns"
    elif avg_jaccard < 0.7:
        recommendation = "[yellow]POTENTIALLY BENEFICIAL[/yellow]"
        reason = "Moderate overlap - ensemble may provide some benefit"
    else:
        recommendation = "[red]NOT RECOMMENDED[/red]"
        reason = "High overlap suggests metrics are redundant"

    console.print(
        f"Average Jaccard Index: [bright_blue]{avg_jaccard:.3f}[/bright_blue]"
    )
    console.print(f"Ensemble Recommendation: {recommendation}")
    console.print(f"Reasoning: {reason}")

    console.print(
        "\n[dim]Note: Jaccard Index measures overlap (intersection/union). Lower values indicate better ensemble potential.[/dim]"
    )

    # Compare ensemble vs individual metrics
    compare_ensemble_vs_individual(
        correct_data, incorrect_data, all_data, top_3_metrics
    )


def analyze_ensemble_potential_fixed_count(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    selection_count: int = 100,
):
    """Analyze overlap between top 3 metrics using fixed count selection."""
    console = Console()

    # Get top 3 metrics by F1 score at fixed count
    results_fixed = calculate_f1_score_for_fixed_count(
        correct_data, incorrect_data, all_data, selection_count
    )
    metrics_with_f1 = [
        (metric, results_fixed[metric]["f1"]) for metric in results_fixed.keys()
    ]
    top_3_metrics = sorted(metrics_with_f1, key=lambda x: x[1], reverse=True)[:3]

    console.print(
        f"\n[bold cyan]FIXED COUNT ENSEMBLE ANALYSIS - TOP 3 METRICS (N={selection_count})[/bold cyan]"
    )
    console.print(
        "Analyzing overlap between top 3 performing metrics with fixed count selection:\n"
    )

    for i, (metric, f1_score) in enumerate(top_3_metrics, 1):
        console.print(
            f"{i}. [cyan]{metric.replace('_', ' ').title()}[/cyan] (F1: {f1_score:.3f})"
        )

    # Analyze at different fixed counts
    selection_counts = [75, 100, 125]

    for count in selection_counts:
        console.print(
            f"\n[bold yellow]═══ Fixed Count {count} Analysis ═══[/bold yellow]"
        )

        # Get selected questions for each metric
        selected_by_metric = {}
        thresholds_used = {}
        print(f"\n  Thresholds used for count {count} selection:")
        for metric, _ in top_3_metrics:
            selected_by_metric[metric], thresholds_used[metric] = (
                get_selected_questions_fixed_count(
                    metric, correct_data, incorrect_data, all_data, count
                )
            )

        # Calculate pairwise overlaps
        metric_names = [metric for metric, _ in top_3_metrics]

        # Create overlap table
        overlap_table = Table(
            title=f"Pairwise Overlap at Count {count}",
            show_header=True,
            header_style="bold magenta",
        )
        overlap_table.add_column("Metric Pair", style="cyan")
        overlap_table.add_column("Overlap Count", justify="right", style="green")
        overlap_table.add_column("Overlap %", justify="right", style="yellow")
        overlap_table.add_column("Union Count", justify="right", style="blue")
        overlap_table.add_column(
            "Jaccard Index", justify="right", style="bright_magenta"
        )

        # Calculate all pairwise overlaps
        overlaps = {}
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric_a, metric_b = metric_names[i], metric_names[j]
                set_a = selected_by_metric[metric_a]
                set_b = selected_by_metric[metric_b]

                overlap = set_a & set_b
                union = set_a | set_b
                overlap_count = len(overlap)
                union_count = len(union)
                overlap_pct = (overlap_count / count) * 100  # Percentage of fixed count
                jaccard = overlap_count / union_count if union_count > 0 else 0

                overlaps[(metric_a, metric_b)] = {
                    "overlap_count": overlap_count,
                    "overlap_pct": overlap_pct,
                    "union_count": union_count,
                    "jaccard": jaccard,
                }

                # Color code Jaccard index
                if jaccard >= 0.7:
                    jaccard_style = "[red]"  # High overlap - bad for ensemble
                elif jaccard >= 0.5:
                    jaccard_style = "[yellow]"  # Medium overlap
                else:
                    jaccard_style = "[green]"  # Low overlap - good for ensemble

                pair_name = f"{metric_a.replace('_', ' ').title()[:12]} × {metric_b.replace('_', ' ').title()[:12]}"

                overlap_table.add_row(
                    pair_name,
                    str(overlap_count),
                    f"{overlap_pct:.1f}%",
                    str(union_count),
                    f"{jaccard_style}{jaccard:.3f}[/{jaccard_style.strip('[]')}]",
                )

        console.print(overlap_table)

        # Calculate three-way overlap
        set_all = [selected_by_metric[metric] for metric, _ in top_3_metrics]
        three_way_overlap = set_all[0] & set_all[1] & set_all[2]
        three_way_union = set_all[0] | set_all[1] | set_all[2]

        console.print("\n[bold]Three-way Analysis:[/bold]")
        console.print(
            f"• Questions selected by ALL 3 metrics: [green]{len(three_way_overlap)}[/green]"
        )
        console.print(
            f"• Questions selected by ANY metric: [blue]{len(three_way_union)}[/blue]"
        )
        console.print(
            f"• Three-way overlap rate: [yellow]{len(three_way_overlap) / count * 100:.1f}%[/yellow] of {count}"
        )

        # Analyze incorrect vs correct in overlaps
        incorrect_in_overlap = len(
            [q for q in three_way_overlap if q.startswith("incorrect_")]
        )
        correct_in_overlap = len(
            [q for q in three_way_overlap if q.startswith("correct_")]
        )

        if len(three_way_overlap) > 0:
            console.print(
                f"• In 3-way overlap: [red]{incorrect_in_overlap}[/red] incorrect, [green]{correct_in_overlap}[/green] correct"
            )
            console.print(
                f"• 3-way overlap precision: [bright_blue]{incorrect_in_overlap / len(three_way_overlap) * 100:.1f}%[/bright_blue]"
            )

        # Export detailed Venn diagram data
        export_venn_diagram_data(
            count, metric_names, selected_by_metric, three_way_overlap, overlaps
        )

    # Compare ensemble vs individual metrics with fixed count
    compare_ensemble_vs_individual_fixed_count(
        correct_data, incorrect_data, all_data, top_3_metrics, selection_count
    )


def compare_ensemble_vs_individual(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    top_3_metrics: List[tuple],
):
    """Compare individual metrics vs ensemble approaches at 10% selection rate."""
    console = Console()

    console.print("\n[bold cyan]═══ ENSEMBLE vs INDIVIDUAL COMPARISON ═══[/bold cyan]")
    console.print(
        "Comparing individual metrics with ensemble (OR logic) at 10% selection rate\n"
    )

    selection_rate = 10

    # Get individual metric performance at 10%
    individual_results = {}
    selected_sets = {}

    for metric, f1_score in top_3_metrics:
        # Calculate individual performance
        individual_results[metric] = calculate_f1_score_for_coverage(
            correct_data, incorrect_data, all_data, selection_rate
        )[metric]

        # Get selected questions for this metric
        selected_sets[metric], _ = get_selected_questions(
            metric, correct_data, incorrect_data, all_data, selection_rate
        )

    # Calculate ensemble performance (OR logic - select if ANY metric flags as uncertain)
    ensemble_selected = set()
    for metric_set in selected_sets.values():
        ensemble_selected |= metric_set

    # Debug: Show individual selections vs ensemble
    console.print("\n[dim]DEBUG: Individual metric selections at 10%:[/dim]")
    for i, (metric, _) in enumerate(top_3_metrics):
        selected_count = len(selected_sets[metric])
        console.print(f"[dim]• {metric}: {selected_count} questions[/dim]")

    console.print(f"[dim]• Ensemble (OR): {len(ensemble_selected)} questions[/dim]")

    # Calculate overlaps for debugging
    metric_names = [metric for metric, _ in top_3_metrics]
    if len(metric_names) >= 2:
        overlap_01 = len(
            selected_sets[metric_names[0]] & selected_sets[metric_names[1]]
        )
        console.print(
            f"[dim]• Overlap {metric_names[0][:8]} & {metric_names[1][:8]}: {overlap_01}[/dim]"
        )
    if len(metric_names) >= 3:
        overlap_02 = len(
            selected_sets[metric_names[0]] & selected_sets[metric_names[2]]
        )
        overlap_12 = len(
            selected_sets[metric_names[1]] & selected_sets[metric_names[2]]
        )
        three_way = len(
            selected_sets[metric_names[0]]
            & selected_sets[metric_names[1]]
            & selected_sets[metric_names[2]]
        )
        console.print(
            f"[dim]• Overlap {metric_names[0][:8]} & {metric_names[2][:8]}: {overlap_02}[/dim]"
        )
        console.print(
            f"[dim]• Overlap {metric_names[1][:8]} & {metric_names[2][:8]}: {overlap_12}[/dim]"
        )
        console.print(f"[dim]• Three-way overlap: {three_way}[/dim]")

        # Check if metrics are selecting identical sets
        if selected_sets[metric_names[0]] == selected_sets[metric_names[1]]:
            console.print(
                f"[dim red]WARNING: {metric_names[0]} and {metric_names[1]} select IDENTICAL questions![/dim red]"
            )
        if selected_sets[metric_names[0]] == selected_sets[metric_names[2]]:
            console.print(
                f"[dim red]WARNING: {metric_names[0]} and {metric_names[2]} select IDENTICAL questions![/dim red]"
            )
        if selected_sets[metric_names[1]] == selected_sets[metric_names[2]]:
            console.print(
                f"[dim red]WARNING: {metric_names[1]} and {metric_names[2]} select IDENTICAL questions![/dim red]"
            )

        # Show some examples of unique selections
        unique_to_0 = (
            selected_sets[metric_names[0]]
            - selected_sets[metric_names[1]]
            - selected_sets[metric_names[2]]
        )
        unique_to_1 = (
            selected_sets[metric_names[1]]
            - selected_sets[metric_names[0]]
            - selected_sets[metric_names[2]]
        )
        unique_to_2 = (
            selected_sets[metric_names[2]]
            - selected_sets[metric_names[0]]
            - selected_sets[metric_names[1]]
        )

        console.print(
            f"[dim]• Unique to {metric_names[0][:8]}: {len(unique_to_0)}[/dim]"
        )
        console.print(
            f"[dim]• Unique to {metric_names[1][:8]}: {len(unique_to_1)}[/dim]"
        )
        console.print(
            f"[dim]• Unique to {metric_names[2][:8]}: {len(unique_to_2)}[/dim]"
        )

    # Calculate ensemble confusion matrix
    total_correct = len(correct_data[list(correct_data.keys())[0]])
    total_incorrect = len(incorrect_data[list(incorrect_data.keys())[0]])

    # Count correct/incorrect in ensemble selection
    ensemble_correct_selected = len(
        [q for q in ensemble_selected if q.startswith("correct_")]
    )
    ensemble_incorrect_selected = len(
        [q for q in ensemble_selected if q.startswith("incorrect_")]
    )

    # Calculate ensemble metrics
    tp = ensemble_incorrect_selected  # True positives: incorrect answers flagged as uncertain
    fp = ensemble_correct_selected  # False positives: correct answers flagged as uncertain
    fn = (
        total_incorrect - ensemble_incorrect_selected
    )  # False negatives: incorrect answers not flagged

    ensemble_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ensemble_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ensemble_f1 = (
        2
        * (ensemble_precision * ensemble_recall)
        / (ensemble_precision + ensemble_recall)
        if (ensemble_precision + ensemble_recall) > 0
        else 0.0
    )

    # Create comparison table
    comparison_table = Table(
        title="Individual vs Ensemble Performance Comparison (10% Selection Rate)",
        show_header=True,
        header_style="bold magenta",
    )

    comparison_table.add_column("Method", style="cyan", no_wrap=True)
    comparison_table.add_column("Selected", justify="right", style="blue")
    comparison_table.add_column("Corr.Sel", justify="right", style="red")
    comparison_table.add_column("Incor.Sel", justify="right", style="green")
    comparison_table.add_column("Precision", justify="right", style="bright_blue")
    comparison_table.add_column("Recall", justify="right", style="bright_green")
    comparison_table.add_column("F1", justify="right", style="bright_magenta")
    comparison_table.add_column("Improvement", justify="right", style="yellow")

    # Add individual metrics to table
    baseline_f1 = 0
    for i, (metric, _) in enumerate(top_3_metrics):
        result = individual_results[metric]

        # Color-code F1 scores
        f1_value = result["f1"]
        if f1_value >= 0.5:
            f1_style = "[bright_green]"
        elif f1_value >= 0.3:
            f1_style = "[yellow]"
        else:
            f1_style = "[red]"

        if i == 0:  # Use first metric as baseline
            baseline_f1 = f1_value
            improvement = "baseline"
        else:
            improvement = (
                f"{((f1_value - baseline_f1) / baseline_f1 * 100):+.1f}%"
                if baseline_f1 > 0
                else "N/A"
            )

        comparison_table.add_row(
            metric.replace("_", " ").title(),
            str(result["total_selected"]),
            str(result["fp"]),  # correct selected
            str(result["tp"]),  # incorrect selected
            f"{result['precision']:.3f}",
            f"{result['recall']:.3f}",
            f"{f1_style}{result['f1']:.3f}[/{f1_style.strip('[]')}]",
            improvement,
        )

    # Add ensemble row
    ensemble_f1_style = (
        "[bright_green]"
        if ensemble_f1 >= 0.5
        else "[yellow]"
        if ensemble_f1 >= 0.3
        else "[red]"
    )
    ensemble_improvement = (
        f"{((ensemble_f1 - baseline_f1) / baseline_f1 * 100):+.1f}%"
        if baseline_f1 > 0
        else "N/A"
    )

    comparison_table.add_row(
        "[bold]Ensemble (OR)[/bold]",
        str(len(ensemble_selected)),
        str(ensemble_correct_selected),
        str(ensemble_incorrect_selected),
        f"{ensemble_precision:.3f}",
        f"{ensemble_recall:.3f}",
        f"{ensemble_f1_style}{ensemble_f1:.3f}[/{ensemble_f1_style.strip('[]')}]",
        f"[bold]{ensemble_improvement}[/bold]",
    )

    console.print(comparison_table)

    # Summary analysis
    console.print("\n[bold]Ensemble Analysis Summary:[/bold]")
    console.print(
        f"• Ensemble selects [blue]{len(ensemble_selected)}[/blue] total questions ({len(ensemble_selected) / (total_correct + total_incorrect) * 100:.1f}% of dataset)"
    )
    console.print(
        f"• Individual metrics select ~[blue]{individual_results[top_3_metrics[0][0]]['total_selected']}[/blue] questions each (~10% of dataset)"
    )
    console.print(
        f"• Ensemble precision: [bright_blue]{ensemble_precision:.3f}[/bright_blue] vs best individual: [bright_blue]{max(r['precision'] for r in individual_results.values()):.3f}[/bright_blue]"
    )
    console.print(
        f"• Ensemble recall: [bright_green]{ensemble_recall:.3f}[/bright_green] vs best individual: [bright_green]{max(r['recall'] for r in individual_results.values()):.3f}[/bright_green]"
    )
    console.print(
        f"• Ensemble F1: [bright_magenta]{ensemble_f1:.3f}[/bright_magenta] vs best individual: [bright_magenta]{max(r['f1'] for r in individual_results.values()):.3f}[/bright_magenta]"
    )

    # Recommendation
    best_individual_f1 = max(r["f1"] for r in individual_results.values())
    f1_improvement = ensemble_f1 - best_individual_f1

    if f1_improvement > 0.05:
        recommendation = "[green]ENSEMBLE RECOMMENDED[/green]"
        reason = f"Significant F1 improvement: +{f1_improvement:.3f}"
    elif f1_improvement > 0.02:
        recommendation = "[yellow]ENSEMBLE BENEFICIAL[/yellow]"
        reason = f"Moderate F1 improvement: +{f1_improvement:.3f}"
    elif f1_improvement > -0.02:
        recommendation = "[yellow]MARGINAL BENEFIT[/yellow]"
        reason = f"Small F1 change: {f1_improvement:+.3f}"
    else:
        recommendation = "[red]INDIVIDUAL BETTER[/red]"
        reason = f"F1 decrease: {f1_improvement:+.3f}"

    console.print(f"\n[bold]Final Recommendation:[/bold] {recommendation}")
    console.print(f"[bold]Reasoning:[/bold] {reason}")


def compare_ensemble_vs_individual_fixed_count(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    top_3_metrics: List[tuple],
    selection_count: int = 100,
):
    """Compare individual metrics vs ensemble approaches with fixed count selection."""
    console = Console()

    console.print(
        f"\n[bold cyan]═══ FIXED COUNT ENSEMBLE vs INDIVIDUAL COMPARISON (N={selection_count}) ═══[/bold cyan]"
    )
    console.print(
        "Comparing individual metrics with ensemble (OR logic) using fixed count selection\n"
    )

    # Get individual metric performance with fixed count
    individual_results = {}
    selected_sets = {}

    for metric, _ in top_3_metrics:
        # Calculate individual performance
        individual_results[metric] = calculate_f1_score_for_fixed_count(
            correct_data, incorrect_data, all_data, selection_count
        )[metric]

        # Get selected questions for this metric
        selected_sets[metric], _ = get_selected_questions_fixed_count(
            metric, correct_data, incorrect_data, all_data, selection_count
        )

    # Calculate ensemble performance (OR logic - select if ANY metric flags as uncertain)
    ensemble_selected = set()
    for metric_set in selected_sets.values():
        ensemble_selected |= metric_set

    # Debug: Show individual selections vs ensemble
    console.print(
        f"\n[dim]DEBUG: Individual metric selections at count {selection_count}:[/dim]"
    )
    for i, (metric, _) in enumerate(top_3_metrics):
        selected_count = len(selected_sets[metric])
        console.print(f"[dim]• {metric}: {selected_count} questions[/dim]")

    console.print(f"[dim]• Ensemble (OR): {len(ensemble_selected)} questions[/dim]")

    # Calculate overlaps for debugging
    metric_names = [metric for metric, _ in top_3_metrics]
    if len(metric_names) >= 2:
        overlap_01 = len(
            selected_sets[metric_names[0]] & selected_sets[metric_names[1]]
        )
        console.print(
            f"[dim]• Overlap {metric_names[0][:8]} & {metric_names[1][:8]}: {overlap_01}[/dim]"
        )
    if len(metric_names) >= 3:
        overlap_02 = len(
            selected_sets[metric_names[0]] & selected_sets[metric_names[2]]
        )
        overlap_12 = len(
            selected_sets[metric_names[1]] & selected_sets[metric_names[2]]
        )
        three_way = len(
            selected_sets[metric_names[0]]
            & selected_sets[metric_names[1]]
            & selected_sets[metric_names[2]]
        )
        console.print(
            f"[dim]• Overlap {metric_names[0][:8]} & {metric_names[2][:8]}: {overlap_02}[/dim]"
        )
        console.print(
            f"[dim]• Overlap {metric_names[1][:8]} & {metric_names[2][:8]}: {overlap_12}[/dim]"
        )
        console.print(f"[dim]• Three-way overlap: {three_way}[/dim]")

        # Show unique selections
        unique_to_0 = (
            selected_sets[metric_names[0]]
            - selected_sets[metric_names[1]]
            - selected_sets[metric_names[2]]
        )
        unique_to_1 = (
            selected_sets[metric_names[1]]
            - selected_sets[metric_names[0]]
            - selected_sets[metric_names[2]]
        )
        unique_to_2 = (
            selected_sets[metric_names[2]]
            - selected_sets[metric_names[0]]
            - selected_sets[metric_names[1]]
        )

        console.print(
            f"[dim]• Unique to {metric_names[0][:8]}: {len(unique_to_0)}[/dim]"
        )
        console.print(
            f"[dim]• Unique to {metric_names[1][:8]}: {len(unique_to_1)}[/dim]"
        )
        console.print(
            f"[dim]• Unique to {metric_names[2][:8]}: {len(unique_to_2)}[/dim]"
        )

    # Calculate ensemble confusion matrix
    total_correct = len(correct_data[list(correct_data.keys())[0]])
    total_incorrect = len(incorrect_data[list(incorrect_data.keys())[0]])

    # Count correct/incorrect in ensemble selection
    ensemble_correct_selected = len(
        [q for q in ensemble_selected if q.startswith("correct_")]
    )
    ensemble_incorrect_selected = len(
        [q for q in ensemble_selected if q.startswith("incorrect_")]
    )

    # Calculate ensemble metrics
    tp = ensemble_incorrect_selected  # True positives: incorrect answers flagged as uncertain
    fp = ensemble_correct_selected  # False positives: correct answers flagged as uncertain
    fn = (
        total_incorrect - ensemble_incorrect_selected
    )  # False negatives: incorrect answers not flagged

    ensemble_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ensemble_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ensemble_f1 = (
        2
        * (ensemble_precision * ensemble_recall)
        / (ensemble_precision + ensemble_recall)
        if (ensemble_precision + ensemble_recall) > 0
        else 0.0
    )

    # Create comparison table
    comparison_table = Table(
        title=f"Individual vs Ensemble Performance Comparison (Fixed Count {selection_count})",
        show_header=True,
        header_style="bold magenta",
    )

    comparison_table.add_column("Method", style="cyan", no_wrap=True)
    comparison_table.add_column("Selected", justify="right", style="blue")
    comparison_table.add_column("Corr.Sel", justify="right", style="red")
    comparison_table.add_column("Incor.Sel", justify="right", style="green")
    comparison_table.add_column("Precision", justify="right", style="bright_blue")
    comparison_table.add_column("Recall", justify="right", style="bright_green")
    comparison_table.add_column("F1", justify="right", style="bright_magenta")
    comparison_table.add_column("Improvement", justify="right", style="yellow")

    # Add individual metrics to table
    baseline_f1 = 0
    for i, (metric, _) in enumerate(top_3_metrics):
        result = individual_results[metric]

        # Color-code F1 scores
        f1_value = result["f1"]
        if f1_value >= 0.5:
            f1_style = "[bright_green]"
        elif f1_value >= 0.3:
            f1_style = "[yellow]"
        else:
            f1_style = "[red]"

        if i == 0:  # Use first metric as baseline
            baseline_f1 = f1_value
            improvement = "baseline"
        else:
            improvement = (
                f"{((f1_value - baseline_f1) / baseline_f1 * 100):+.1f}%"
                if baseline_f1 > 0
                else "N/A"
            )

        comparison_table.add_row(
            metric.replace("_", " ").title(),
            str(result["total_selected"]),
            str(result["fp"]),  # correct selected
            str(result["tp"]),  # incorrect selected
            f"{result['precision']:.3f}",
            f"{result['recall']:.3f}",
            f"{f1_style}{result['f1']:.3f}[/{f1_style.strip('[]')}]",
            improvement,
        )

    # Add ensemble row
    ensemble_f1_style = (
        "[bright_green]"
        if ensemble_f1 >= 0.5
        else "[yellow]"
        if ensemble_f1 >= 0.3
        else "[red]"
    )
    ensemble_improvement = (
        f"{((ensemble_f1 - baseline_f1) / baseline_f1 * 100):+.1f}%"
        if baseline_f1 > 0
        else "N/A"
    )

    comparison_table.add_row(
        "[bold]Ensemble (OR)[/bold]",
        str(len(ensemble_selected)),
        str(ensemble_correct_selected),
        str(ensemble_incorrect_selected),
        f"{ensemble_precision:.3f}",
        f"{ensemble_recall:.3f}",
        f"{ensemble_f1_style}{ensemble_f1:.3f}[/{ensemble_f1_style.strip('[]')}]",
        f"[bold]{ensemble_improvement}[/bold]",
    )

    console.print(comparison_table)

    # Summary analysis
    console.print("\n[bold]Fixed Count Ensemble Analysis Summary:[/bold]")
    console.print(
        f"• Ensemble selects [blue]{len(ensemble_selected)}[/blue] total questions ({len(ensemble_selected) / (total_correct + total_incorrect) * 100:.1f}% of dataset)"
    )
    console.print(
        f"• Individual metrics each select exactly [blue]{selection_count}[/blue] questions"
    )
    console.print(
        f"• Ensemble precision: [bright_blue]{ensemble_precision:.3f}[/bright_blue] vs best individual: [bright_blue]{max(r['precision'] for r in individual_results.values()):.3f}[/bright_blue]"
    )
    console.print(
        f"• Ensemble recall: [bright_green]{ensemble_recall:.3f}[/bright_green] vs best individual: [bright_green]{max(r['recall'] for r in individual_results.values()):.3f}[/bright_green]"
    )
    console.print(
        f"• Ensemble F1: [bright_magenta]{ensemble_f1:.3f}[/bright_magenta] vs best individual: [bright_magenta]{max(r['f1'] for r in individual_results.values()):.3f}[/bright_magenta]"
    )

    # Recommendation
    best_individual_f1 = max(r["f1"] for r in individual_results.values())
    f1_improvement = ensemble_f1 - best_individual_f1

    if f1_improvement > 0.05:
        recommendation = "[green]ENSEMBLE RECOMMENDED[/green]"
        reason = f"Significant F1 improvement: +{f1_improvement:.3f}"
    elif f1_improvement > 0.02:
        recommendation = "[yellow]ENSEMBLE BENEFICIAL[/yellow]"
        reason = f"Moderate F1 improvement: +{f1_improvement:.3f}"
    elif f1_improvement > -0.02:
        recommendation = "[yellow]MARGINAL BENEFIT[/yellow]"
        reason = f"Small F1 change: {f1_improvement:+.3f}"
    else:
        recommendation = "[red]INDIVIDUAL BETTER[/red]"
        reason = f"F1 decrease: {f1_improvement:+.3f}"

    console.print(f"\n[bold]Final Recommendation:[/bold] {recommendation}")
    console.print(f"[bold]Reasoning:[/bold] {reason}")

    # Calculate ensemble expansion ratio
    total_individual_selections = sum(
        len(selected_sets[metric]) for metric in selected_sets.keys()
    )
    expansion_ratio = len(ensemble_selected) / (
        total_individual_selections / len(selected_sets)
    )

    console.print("\n[bold]Ensemble Expansion Analysis:[/bold]")
    console.print(
        f"• Expected if no overlap: [blue]{total_individual_selections}[/blue] questions"
    )
    console.print(
        f"• Actual ensemble size: [blue]{len(ensemble_selected)}[/blue] questions"
    )
    console.print(
        f"• Expansion ratio: [yellow]{expansion_ratio:.2f}x[/yellow] (lower = more overlap)"
    )

    if expansion_ratio < 1.5:
        console.print(
            "• [green]High overlap detected - metrics are complementary[/green]"
        )
    elif expansion_ratio < 2.0:
        console.print("• [yellow]Moderate overlap - some complementarity[/yellow]")
    else:
        console.print("• [red]Low overlap - metrics may be redundant[/red]")


def compare_ensemble_across_counts(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    selection_counts: List[int],
):
    """Compare ensemble performance across different fixed selection counts."""
    console = Console()

    console.print(
        "\n[bold cyan]═══ ENSEMBLE PERFORMANCE ACROSS SELECTION COUNTS ═══[/bold cyan]"
    )
    console.print(
        "Comparing ensemble vs individual performance across different selection counts\n"
    )

    # Get top 3 metrics based on F1 score at count=100
    results_100 = calculate_f1_score_for_fixed_count(
        correct_data, incorrect_data, all_data, 100
    )
    metrics_with_f1 = [
        (metric, results_100[metric]["f1"]) for metric in results_100.keys()
    ]
    top_3_metrics = sorted(metrics_with_f1, key=lambda x: x[1], reverse=True)[:3]

    console.print("Top 3 metrics (based on F1 at count=100):")
    for i, (metric, f1_score) in enumerate(top_3_metrics, 1):
        console.print(
            f"{i}. [cyan]{metric.replace('_', ' ').title()}[/cyan] (F1: {f1_score:.3f})"
        )

    # Create comparison table
    comparison_table = Table(
        title="Ensemble vs Individual Performance Comparison Across Counts",
        show_header=True,
        header_style="bold magenta",
    )

    comparison_table.add_column("Count", justify="center", style="yellow")
    comparison_table.add_column("Method", style="cyan", no_wrap=True)
    comparison_table.add_column("Selected", justify="right", style="blue")
    comparison_table.add_column("Incor.Sel", justify="right", style="green")
    comparison_table.add_column("Precision", justify="right", style="bright_blue")
    comparison_table.add_column("Recall", justify="right", style="bright_green")
    comparison_table.add_column("F1", justify="right", style="bright_magenta")
    comparison_table.add_column("Improvement", justify="right", style="yellow")

    # Calculate results for all counts
    all_results = {}
    all_ensemble_results = {}

    for count in selection_counts:
        # Get individual metric performance
        individual_results = calculate_f1_score_for_fixed_count(
            correct_data, incorrect_data, all_data, count
        )

        # Get selected sets for ensemble calculation
        selected_sets = {}
        for metric, _ in top_3_metrics:
            selected_sets[metric], _ = get_selected_questions_fixed_count(
                metric, correct_data, incorrect_data, all_data, count
            )

        # Calculate ensemble performance
        ensemble_selected = set()
        for metric_set in selected_sets.values():
            ensemble_selected |= metric_set

        # Calculate ensemble metrics
        total_correct = len(correct_data[list(correct_data.keys())[0]])
        total_incorrect = len(incorrect_data[list(incorrect_data.keys())[0]])

        ensemble_correct_selected = len(
            [q for q in ensemble_selected if q.startswith("correct_")]
        )
        ensemble_incorrect_selected = len(
            [q for q in ensemble_selected if q.startswith("incorrect_")]
        )

        tp = ensemble_incorrect_selected
        fp = ensemble_correct_selected
        fn = total_incorrect - ensemble_incorrect_selected

        ensemble_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        ensemble_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ensemble_f1 = (
            2
            * (ensemble_precision * ensemble_recall)
            / (ensemble_precision + ensemble_recall)
            if (ensemble_precision + ensemble_recall) > 0
            else 0.0
        )

        all_results[count] = individual_results
        all_ensemble_results[count] = {
            "total_selected": len(ensemble_selected),
            "tp": tp,
            "fp": fp,
            "precision": ensemble_precision,
            "recall": ensemble_recall,
            "f1": ensemble_f1,
        }

    # Add rows to table for each count
    for count in selection_counts:
        individual_results = all_results[count]
        ensemble_result = all_ensemble_results[count]

        # Find best individual metric for this count
        best_individual_metric = max(
            top_3_metrics, key=lambda x: individual_results[x[0]]["f1"]
        )[0]
        best_individual_f1 = individual_results[best_individual_metric]["f1"]

        # Add best individual metric row
        best_result = individual_results[best_individual_metric]
        f1_value = best_result["f1"]
        if f1_value >= 0.5:
            f1_style = "[bright_green]"
        elif f1_value >= 0.3:
            f1_style = "[yellow]"
        else:
            f1_style = "[red]"

        comparison_table.add_row(
            str(count),
            f"Best Individual\n({best_individual_metric.replace('_', ' ').title()})",
            str(best_result["total_selected"]),
            str(best_result["tp"]),
            f"{best_result['precision']:.3f}",
            f"{best_result['recall']:.3f}",
            f"{f1_style}{best_result['f1']:.3f}[/{f1_style.strip('[]')}]",
            "baseline",
        )

        # Add ensemble row
        ensemble_f1 = ensemble_result["f1"]
        if ensemble_f1 >= 0.5:
            ensemble_f1_style = "[bright_green]"
        elif ensemble_f1 >= 0.3:
            ensemble_f1_style = "[yellow]"
        else:
            ensemble_f1_style = "[red]"

        f1_improvement = ensemble_f1 - best_individual_f1
        improvement_str = (
            f"{((f1_improvement / best_individual_f1) * 100):+.1f}%"
            if best_individual_f1 > 0
            else "N/A"
        )

        comparison_table.add_row(
            "",
            "[bold]Ensemble (OR)[/bold]",
            str(ensemble_result["total_selected"]),
            str(ensemble_result["tp"]),
            f"{ensemble_result['precision']:.3f}",
            f"{ensemble_result['recall']:.3f}",
            f"{ensemble_f1_style}{ensemble_result['f1']:.3f}[/{ensemble_f1_style.strip('[]')}]",
            f"[bold]{improvement_str}[/bold]",
        )

        # Add separator between counts
        if count != selection_counts[-1]:
            comparison_table.add_row("", "", "", "", "", "", "", "")

    console.print(comparison_table)

    # Summary analysis
    console.print("\n[bold]Summary Analysis Across Counts:[/bold]")

    best_ensemble_count = max(
        selection_counts, key=lambda c: all_ensemble_results[c]["f1"]
    )
    best_ensemble_f1 = all_ensemble_results[best_ensemble_count]["f1"]

    best_individual_count = max(
        selection_counts,
        key=lambda c: max(all_results[c][m[0]]["f1"] for m in top_3_metrics),
    )
    best_individual_f1 = max(
        all_results[best_individual_count][m[0]]["f1"] for m in top_3_metrics
    )

    console.print(
        f"• Best ensemble F1: [bright_magenta]{best_ensemble_f1:.3f}[/bright_magenta] at count {best_ensemble_count}"
    )
    console.print(
        f"• Best individual F1: [bright_magenta]{best_individual_f1:.3f}[/bright_magenta] at count {best_individual_count}"
    )

    # Show ensemble size trends
    console.print("\n[bold]Ensemble Size Analysis:[/bold]")
    for count in selection_counts:
        ensemble_size = all_ensemble_results[count]["total_selected"]
        expansion_ratio = (
            ensemble_size / count
        )  # How much bigger than individual selections
        console.print(
            f"• Count {count}: Ensemble selects [blue]{ensemble_size}[/blue] questions ([yellow]{expansion_ratio:.2f}x[/yellow] expansion)"
        )

    # Final recommendation
    overall_best_f1 = max(best_ensemble_f1, best_individual_f1)
    if overall_best_f1 == best_ensemble_f1:
        console.print(
            f"\n[bold]Recommendation:[/bold] [green]Use ensemble at count {best_ensemble_count}[/green]"
        )
        console.print(
            f"[bold]Reasoning:[/bold] Best overall F1 score ({best_ensemble_f1:.3f})"
        )
    else:
        console.print(
            f"\n[bold]Recommendation:[/bold] [yellow]Use individual metric at count {best_individual_count}[/yellow]"
        )
        console.print(
            f"[bold]Reasoning:[/bold] Individual metric achieves best F1 score ({best_individual_f1:.3f})"
        )


def train_linear_model_for_uncertainty(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    top_3_metrics: List[tuple],
    selection_counts: List[int],
):
    """Train a simple logistic regression model using top 3 metrics to predict uncertainty."""
    console = Console()

    console.print(
        "\n[bold cyan]═══ LINEAR MODEL UNCERTAINTY PREDICTION ═══[/bold cyan]"
    )
    console.print(
        "Training logistic regression using top 3 metrics to predict incorrect answers\n"
    )

    # Prepare data for the model
    metric_names = [metric for metric, _ in top_3_metrics]
    console.print("Using metrics:")
    for i, metric in enumerate(metric_names, 1):
        console.print(f"{i}. [cyan]{metric.replace('_', ' ').title()}[/cyan]")

    # Create feature matrix and labels
    X = []
    y = []
    question_ids = []

    # Add correct answers (label = 0, certain)
    for idx in range(len(correct_data[metric_names[0]])):
        features = [correct_data[metric][idx] for metric in metric_names]
        X.append(features)
        y.append(0)  # 0 = correct/certain
        question_ids.append(f"correct_{idx}")

    # Add incorrect answers (label = 1, uncertain)
    for idx in range(len(incorrect_data[metric_names[0]])):
        features = [incorrect_data[metric][idx] for metric in metric_names]
        X.append(features)
        y.append(1)  # 1 = incorrect/uncertain
        question_ids.append(f"incorrect_{idx}")

    X = np.array(X)
    y = np.array(y)

    # Split data for training and validation
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, question_ids, test_size=0.3, random_state=42, stratify=y
    )

    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Get model coefficients
    console.print("\n[bold]Model Coefficients:[/bold]")
    for i, metric in enumerate(metric_names):
        coef = model.coef_[0][i]
        direction = "↑" if coef > 0 else "↓"
        console.print(
            f"• {metric.replace('_', ' ').title()}: [yellow]{coef:.4f}[/yellow] {direction}"
        )

    console.print(f"• Intercept: [yellow]{model.intercept_[0]:.4f}[/yellow]")

    # Predict on full dataset to get uncertainty scores
    uncertainty_scores = model.predict_proba(X)[:, 1]  # Probability of being uncertain

    # Create results for different selection counts
    console.print(
        "\n[bold]Linear Model Performance at Different Selection Counts:[/bold]"
    )

    # Create comparison table
    model_table = Table(
        title="Linear Model vs Top Individual Metric Performance",
        show_header=True,
        header_style="bold magenta",
    )

    model_table.add_column("Count", justify="center", style="yellow")
    model_table.add_column("Method", style="cyan", no_wrap=True)
    model_table.add_column("Selected", justify="right", style="blue")
    model_table.add_column("Incor.Sel", justify="right", style="green")
    model_table.add_column("Precision", justify="right", style="bright_blue")
    model_table.add_column("Recall", justify="right", style="bright_green")
    model_table.add_column("F1", justify="right", style="bright_magenta")
    model_table.add_column("Improvement", justify="right", style="yellow")

    for count in selection_counts:
        # Get top N most uncertain according to linear model
        top_uncertain_indices = np.argsort(uncertainty_scores)[-count:]
        model_selected_ids = set(question_ids[i] for i in top_uncertain_indices)

        # Count correct vs incorrect in model selection
        model_correct_selected = len(
            [q for q in model_selected_ids if q.startswith("correct_")]
        )
        model_incorrect_selected = len(
            [q for q in model_selected_ids if q.startswith("incorrect_")]
        )

        # Calculate model metrics
        total_correct = len(correct_data[metric_names[0]])
        total_incorrect = len(incorrect_data[metric_names[0]])

        tp_model = model_incorrect_selected
        fp_model = model_correct_selected
        fn_model = total_incorrect - model_incorrect_selected

        precision_model = (
            tp_model / (tp_model + fp_model) if (tp_model + fp_model) > 0 else 0.0
        )
        recall_model = (
            tp_model / (tp_model + fn_model) if (tp_model + fn_model) > 0 else 0.0
        )
        f1_model = (
            2 * (precision_model * recall_model) / (precision_model + recall_model)
            if (precision_model + recall_model) > 0
            else 0.0
        )

        # Get best individual metric performance for comparison
        individual_results = calculate_f1_score_for_fixed_count(
            correct_data, incorrect_data, all_data, count
        )
        best_individual_metric = max(
            top_3_metrics, key=lambda x: individual_results[x[0]]["f1"]
        )[0]
        best_individual_result = individual_results[best_individual_metric]
        best_individual_f1 = best_individual_result["f1"]

        # Color-code F1 scores
        if best_individual_f1 >= 0.5:
            individual_f1_style = "[bright_green]"
        elif best_individual_f1 >= 0.3:
            individual_f1_style = "[yellow]"
        else:
            individual_f1_style = "[red]"

        if f1_model >= 0.5:
            model_f1_style = "[bright_green]"
        elif f1_model >= 0.3:
            model_f1_style = "[yellow]"
        else:
            model_f1_style = "[red]"

        # Calculate improvement
        f1_improvement = f1_model - best_individual_f1
        improvement_str = (
            f"{((f1_improvement / best_individual_f1) * 100):+.1f}%"
            if best_individual_f1 > 0
            else "N/A"
        )

        # Add rows to table
        model_table.add_row(
            str(count),
            f"Best Individual\n({best_individual_metric.replace('_', ' ').title()})",
            str(best_individual_result["total_selected"]),
            str(best_individual_result["tp"]),
            f"{best_individual_result['precision']:.3f}",
            f"{best_individual_result['recall']:.3f}",
            f"{individual_f1_style}{best_individual_result['f1']:.3f}[/{individual_f1_style.strip('[]')}]",
            "baseline",
        )

        model_table.add_row(
            "",
            "[bold]Linear Model[/bold]",
            str(count),
            str(model_incorrect_selected),
            f"{precision_model:.3f}",
            f"{recall_model:.3f}",
            f"{model_f1_style}{f1_model:.3f}[/{model_f1_style.strip('[]')}]",
            f"[bold]{improvement_str}[/bold]",
        )

        # Add separator between counts
        if count != selection_counts[-1]:
            model_table.add_row("", "", "", "", "", "", "", "")

    console.print(model_table)

    # Feature importance analysis
    console.print("\n[bold]Feature Importance Analysis:[/bold]")
    feature_importance = np.abs(model.coef_[0])
    sorted_indices = np.argsort(feature_importance)[::-1]

    for i, idx in enumerate(sorted_indices, 1):
        metric = metric_names[idx]
        importance = feature_importance[idx]
        coef = model.coef_[0][idx]
        direction = (
            "higher values → uncertain" if coef > 0 else "lower values → uncertain"
        )
        console.print(
            f"{i}. [cyan]{metric.replace('_', ' ').title()}[/cyan]: importance [yellow]{importance:.4f}[/yellow] ({direction})"
        )

    # Model validation metrics
    console.print("\n[bold]Model Validation (30% holdout):[/bold]")
    y_pred = model.predict(X_test)
    val_precision = precision_score(y_test, y_pred)
    val_recall = recall_score(y_test, y_pred)
    val_f1 = f1_score(y_test, y_pred)

    console.print(
        f"• Validation Precision: [bright_blue]{val_precision:.3f}[/bright_blue]"
    )
    console.print(f"• Validation Recall: [bright_green]{val_recall:.3f}[/bright_green]")
    console.print(f"• Validation F1: [bright_magenta]{val_f1:.3f}[/bright_magenta]")

    return model, uncertainty_scores, question_ids


def create_final_method_comparison(
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    top_3_metrics: List[tuple],
    model,
    uncertainty_scores: np.ndarray,
    question_ids: List[str],
    selection_counts: List[int],
):
    """Create a comprehensive comparison of all methods: individual, ensemble, and linear model."""
    console = Console()

    console.print("\n[bold cyan]═══ COMPREHENSIVE METHOD COMPARISON ═══[/bold cyan]")
    console.print(
        "Comparing all uncertainty detection methods across selection counts\n"
    )

    # Create master comparison table
    master_table = Table(
        title="All Methods Performance Comparison",
        show_header=True,
        header_style="bold magenta",
    )

    master_table.add_column("Count", justify="center", style="yellow")
    master_table.add_column("Method", style="cyan", no_wrap=True)
    master_table.add_column("Selected", justify="right", style="blue")
    master_table.add_column("Incor.Sel", justify="right", style="green")
    master_table.add_column("Precision", justify="right", style="bright_blue")
    master_table.add_column("Recall", justify="right", style="bright_green")
    master_table.add_column("F1", justify="right", style="bright_magenta")
    master_table.add_column("Rank", justify="center", style="white")

    all_method_results = {}

    for count in selection_counts:
        method_results = []

        # 1. Get individual metric results
        individual_results = calculate_f1_score_for_fixed_count(
            correct_data, incorrect_data, all_data, count
        )

        for metric, _ in top_3_metrics:
            result = individual_results[metric]
            method_results.append(
                {
                    "method": f"Individual: {metric.replace('_', ' ').title()}",
                    "selected": result["total_selected"],
                    "incorrect_selected": result["tp"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "type": "individual",
                }
            )

        # 2. Get ensemble results
        selected_sets = {}
        for metric, _ in top_3_metrics:
            selected_sets[metric], _ = get_selected_questions_fixed_count(
                metric, correct_data, incorrect_data, all_data, count
            )

        ensemble_selected = set()
        for metric_set in selected_sets.values():
            ensemble_selected |= metric_set

        total_correct = len(correct_data[list(correct_data.keys())[0]])
        total_incorrect = len(incorrect_data[list(incorrect_data.keys())[0]])

        ensemble_correct_selected = len(
            [q for q in ensemble_selected if q.startswith("correct_")]
        )
        ensemble_incorrect_selected = len(
            [q for q in ensemble_selected if q.startswith("incorrect_")]
        )

        tp_ensemble = ensemble_incorrect_selected
        fp_ensemble = ensemble_correct_selected
        fn_ensemble = total_incorrect - ensemble_incorrect_selected

        precision_ensemble = (
            tp_ensemble / (tp_ensemble + fp_ensemble)
            if (tp_ensemble + fp_ensemble) > 0
            else 0.0
        )
        recall_ensemble = (
            tp_ensemble / (tp_ensemble + fn_ensemble)
            if (tp_ensemble + fn_ensemble) > 0
            else 0.0
        )
        f1_ensemble = (
            2
            * (precision_ensemble * recall_ensemble)
            / (precision_ensemble + recall_ensemble)
            if (precision_ensemble + recall_ensemble) > 0
            else 0.0
        )

        method_results.append(
            {
                "method": "Ensemble (OR)",
                "selected": len(ensemble_selected),
                "incorrect_selected": tp_ensemble,
                "precision": precision_ensemble,
                "recall": recall_ensemble,
                "f1": f1_ensemble,
                "type": "ensemble",
            }
        )

        # 3. Get linear model results
        top_uncertain_indices = np.argsort(uncertainty_scores)[-count:]
        model_selected_ids = set(question_ids[i] for i in top_uncertain_indices)

        model_correct_selected = len(
            [q for q in model_selected_ids if q.startswith("correct_")]
        )
        model_incorrect_selected = len(
            [q for q in model_selected_ids if q.startswith("incorrect_")]
        )

        tp_model = model_incorrect_selected
        fp_model = model_correct_selected
        fn_model = total_incorrect - model_incorrect_selected

        precision_model = (
            tp_model / (tp_model + fp_model) if (tp_model + fp_model) > 0 else 0.0
        )
        recall_model = (
            tp_model / (tp_model + fn_model) if (tp_model + fn_model) > 0 else 0.0
        )
        f1_model = (
            2 * (precision_model * recall_model) / (precision_model + recall_model)
            if (precision_model + recall_model) > 0
            else 0.0
        )

        method_results.append(
            {
                "method": "Linear Model",
                "selected": count,
                "incorrect_selected": tp_model,
                "precision": precision_model,
                "recall": recall_model,
                "f1": f1_model,
                "type": "model",
            }
        )

        # Sort by F1 score and assign ranks
        method_results.sort(key=lambda x: x["f1"], reverse=True)

        all_method_results[count] = method_results

        # Add to master table
        for rank, result in enumerate(method_results, 1):
            # Color-code F1 scores
            f1_value = result["f1"]
            if f1_value >= 0.5:
                f1_style = "[bright_green]"
            elif f1_value >= 0.3:
                f1_style = "[yellow]"
            else:
                f1_style = "[red]"

            # Color-code method type
            if result["type"] == "individual":
                method_style = "[cyan]"
            elif result["type"] == "ensemble":
                method_style = "[magenta]"
            else:  # model
                method_style = "[green]"

            # Color-code rank
            if rank == 1:
                rank_style = "[bold bright_yellow]🥇"
            elif rank == 2:
                rank_style = "[bold white]🥈"
            elif rank == 3:
                rank_style = "[bold yellow]🥉"
            else:
                rank_style = f"[dim]{rank}[/dim]"

            master_table.add_row(
                str(count) if rank == 1 else "",
                f"{method_style}{result['method']}[/{method_style.strip('[]')}]",
                str(result["selected"]),
                str(result["incorrect_selected"]),
                f"{result['precision']:.3f}",
                f"{result['recall']:.3f}",
                f"{f1_style}{result['f1']:.3f}[/{f1_style.strip('[]')}]",
                rank_style,
            )

        # Add separator between counts
        if count != selection_counts[-1]:
            master_table.add_row("", "", "", "", "", "", "", "")

    console.print(master_table)

    # Summary analysis
    console.print("\n[bold]Overall Performance Summary:[/bold]")

    # Find best method overall
    best_overall = None
    best_f1 = 0

    method_type_wins = {"individual": 0, "ensemble": 0, "model": 0}

    for count in selection_counts:
        results = all_method_results[count]
        winner = results[0]  # First in sorted list
        method_type_wins[winner["type"]] += 1

        if winner["f1"] > best_f1:
            best_f1 = winner["f1"]
            best_overall = (count, winner)

        console.print(
            f"• Count {count}: [bright_yellow]{winner['method']}[/bright_yellow] wins with F1 = [bright_magenta]{winner['f1']:.3f}[/bright_magenta]"
        )

    console.print("\n[bold]Method Type Performance:[/bold]")
    console.print(
        f"• Individual metrics won: [cyan]{method_type_wins['individual']}[/cyan] times"
    )
    console.print(
        f"• Ensemble won: [magenta]{method_type_wins['ensemble']}[/magenta] times"
    )
    console.print(
        f"• Linear model won: [green]{method_type_wins['model']}[/green] times"
    )

    if best_overall:
        count, winner = best_overall
        console.print("\n[bold]🏆 Overall Best Method:[/bold]")
        console.print(
            f"[bright_yellow]{winner['method']}[/bright_yellow] at count {count}"
        )
        console.print(f"F1 Score: [bright_magenta]{winner['f1']:.3f}[/bright_magenta]")
        console.print(
            f"Precision: [bright_blue]{winner['precision']:.3f}[/bright_blue], Recall: [bright_green]{winner['recall']:.3f}[/bright_green]"
        )

    # Method insights
    console.print("\n[bold]Key Insights:[/bold]")

    # Calculate average performance by type
    type_performance = {"individual": [], "ensemble": [], "model": []}
    for count_results in all_method_results.values():
        for result in count_results:
            type_performance[result["type"]].append(result["f1"])

    for method_type, f1_scores in type_performance.items():
        if f1_scores:
            avg_f1 = np.mean(f1_scores)
            console.print(
                f"• {method_type.title()} methods average F1: [bright_magenta]{avg_f1:.3f}[/bright_magenta]"
            )

    # Linear model feature insights
    console.print("\n[bold]Linear Model Feature Weights:[/bold]")
    metric_names = [metric for metric, _ in top_3_metrics]
    for i, metric in enumerate(metric_names):
        coef = model.coef_[0][i]
        direction = "↑" if coef > 0 else "↓"
        console.print(
            f"• {metric.replace('_', ' ').title()}: [yellow]{coef:.4f}[/yellow] {direction}"
        )


def export_llm_readable_data(
    output_file: Path,
    run_ids: List[str],
    thresholds: Dict[str, float],
    correctness_analysis: Dict[str, Dict[str, int]],
    all_correct_data: Dict[str, List[float]],
    all_incorrect_data: Dict[str, List[float]],
    metrics_list: List[str],
) -> None:
    """Export all plot data in LLM-readable JSON format."""

    # Calculate summary statistics for each metric
    export_data = {
        "metadata": {
            "description": "Uncertainty metrics analysis for LLM consumption",
            "run_ids": run_ids,
            "total_metrics": len(metrics_list),
            "export_timestamp": str(Path().resolve()),
        },
        "metrics": {},
    }

    for metric in metrics_list:
        if metric not in thresholds:
            continue

        correct_values = all_correct_data[metric]
        incorrect_values = all_incorrect_data[metric]
        all_values = correct_values + incorrect_values

        # Calculate statistics
        correct_mean = np.mean(correct_values) if correct_values else 0.0
        incorrect_mean = np.mean(incorrect_values) if incorrect_values else 0.0
        correct_std = np.std(correct_values) if correct_values else 0.0
        incorrect_std = np.std(incorrect_values) if incorrect_values else 0.0

        # Determine direction and reasoning
        direction = get_metric_direction(metric, all_correct_data, all_incorrect_data)
        uncertainty_interpretation = (
            "low_values_uncertain" if direction == "<=" else "high_values_uncertain"
        )

        # Get selection analysis
        analysis = correctness_analysis[metric]

        export_data["metrics"][metric] = {
            "threshold": {
                "value": float(thresholds[metric]),
                "direction": direction,
                "interpretation": uncertainty_interpretation,
                "reasoning": f"Incorrect mean ({incorrect_mean:.3f}) {'>' if incorrect_mean > correct_mean else '<='} Correct mean ({correct_mean:.3f})",
            },
            "statistics": {
                "correct": {
                    "count": len(correct_values),
                    "mean": float(correct_mean),
                    "std": float(correct_std),
                    "min": float(min(correct_values)) if correct_values else 0.0,
                    "max": float(max(correct_values)) if correct_values else 0.0,
                },
                "incorrect": {
                    "count": len(incorrect_values),
                    "mean": float(incorrect_mean),
                    "std": float(incorrect_std),
                    "min": float(min(incorrect_values)) if incorrect_values else 0.0,
                    "max": float(max(incorrect_values)) if incorrect_values else 0.0,
                },
                "combined": {
                    "count": len(all_values),
                    "mean": float(np.mean(all_values)) if all_values else 0.0,
                    "std": float(np.std(all_values)) if all_values else 0.0,
                    "min": float(min(all_values)) if all_values else 0.0,
                    "max": float(max(all_values)) if all_values else 0.0,
                },
            },
            "selection_analysis": {
                "correct_selected": analysis["correct_selected"],
                "incorrect_selected": analysis["incorrect_selected"],
                "total_selected": analysis["total_selected"],
                "selection_accuracy": float(
                    100 * analysis["correct_selected"] / analysis["total_selected"]
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
        "overall_accuracy": float(100 * total_correct / total_points)
        if total_points > 0
        else 0.0,
        "best_metrics": sorted(
            [
                (
                    m,
                    export_data["metrics"][m]["uncertainty_effectiveness"][
                        "effect_size"
                    ],
                )
                for m in export_data["metrics"].keys()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        if export_data["metrics"]
        else [],
    }

    # Write to JSON file
    json_file = output_file.with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"LLM-readable data exported to: {json_file}")


def analyze_and_plot(run_ids: List[str]) -> None:
    """Analyze uncertainty metrics and create violin plots."""
    print(f"Analyzing {len(run_ids)} runs: {', '.join(run_ids)}")

    # All available uncertainty metrics
    metrics_list = [
        "agreement_ratio",
        "entropy_freq",
        "consensus_support",
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
    thresholds = calculate_thresholds(all_data, all_correct_data, all_incorrect_data)
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

    for metric in metrics_list:
        if metric not in thresholds:
            continue

        # Determine direction based on actual data
        direction = get_metric_direction(metric, all_correct_data, all_incorrect_data)
        correct_mean = (
            np.mean(all_correct_data[metric]) if all_correct_data[metric] else 0.0
        )
        incorrect_mean = (
            np.mean(all_incorrect_data[metric]) if all_incorrect_data[metric] else 0.0
        )

        analysis = correctness_analysis[metric]
        correct_pct = (
            100 * analysis["correct_selected"] / analysis["total_selected"]
            if analysis["total_selected"] > 0
            else 0
        )

        print(
            f"  {metric.replace('_', ' ').title()}: {direction} {thresholds[metric]:.4f}"
        )
        print(f"    Means: Correct={correct_mean:.3f}, Incorrect={incorrect_mean:.3f}")
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

    # Print comprehensive coverage analysis table
    print_coverage_analysis_table(all_correct_data, all_incorrect_data, all_data)

    # Print fixed count analysis table
    print_fixed_count_analysis_table(all_correct_data, all_incorrect_data, all_data)

    # Analyze ensemble potential with top 3 metrics (both approaches)
    analyze_ensemble_potential(all_correct_data, all_incorrect_data, all_data)

    # Run fixed count ensemble analysis for multiple selection counts
    for count in [75, 100, 125]:
        analyze_ensemble_potential_fixed_count(
            all_correct_data, all_incorrect_data, all_data, selection_count=count
        )

    # Create summary comparison across all counts
    compare_ensemble_across_counts(
        all_correct_data, all_incorrect_data, all_data, [75, 100, 125]
    )

    # Train and evaluate linear model
    results_100 = calculate_f1_score_for_fixed_count(
        all_correct_data, all_incorrect_data, all_data, 100
    )
    metrics_with_f1 = [
        (metric, results_100[metric]["f1"]) for metric in results_100.keys()
    ]
    top_3_metrics = sorted(metrics_with_f1, key=lambda x: x[1], reverse=True)[:3]

    model, uncertainty_scores, question_ids = train_linear_model_for_uncertainty(
        all_correct_data, all_incorrect_data, all_data, top_3_metrics, [75, 100, 125]
    )

    # Create final comprehensive comparison
    create_final_method_comparison(
        all_correct_data,
        all_incorrect_data,
        all_data,
        top_3_metrics,
        model,
        uncertainty_scores,
        question_ids,
        [75, 100, 125],
    )


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

        # Sort data based on data-driven metric direction
        if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
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
            if is_low_uncertainty_metric(metric, correct_data, incorrect_data):
                # For these metrics: low → high (left to right)
                ax.set_xlim(min(metric_values), max(metric_values))
            else:
                # For high uncertainty metrics: high → low (left to right, but values go from high to low)
                ax.set_xlim(max(metric_values), min(metric_values))

        ax.set_ylim(0, 1)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cumulative Fraction of Data")

        # Enhanced title with threshold info
        direction = get_metric_direction(metric, correct_data, incorrect_data)
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
        "Cumulative Selection: Correct vs Incorrect Answers", fontsize=14, y=0.98
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # Save the plot
    output_dir = Path("figures/selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"cumulative_lines_{'_'.join(run_ids)}.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")

    print(f"Cumulative line charts saved to: {output_file}")

    # Export LLM-readable data
    export_llm_readable_data(
        output_file,
        run_ids,
        thresholds,
        correctness_analysis,
        correct_data,
        incorrect_data,
        list(thresholds.keys()),
    )

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

            # Create two separate patches for inside and outside threshold

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

        title = f"{metric.replace('_', ' ').title()}"
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

    # plt.suptitle(
    #     "Split Violin Plots: Correct vs Incorrect Distribution", fontsize=14, y=0.98
    # )
    # plt.tight_layout(rect=(0, 0, 1, 0.95))

    # Save the plot
    output_dir = Path("figures/selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"split_violins_{'_'.join(run_ids)}.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")

    print(f"Split violin plots saved to: {output_file}")

    # Export LLM-readable data
    export_llm_readable_data(
        output_file,
        run_ids,
        thresholds,
        correctness_analysis,
        correct_data,
        incorrect_data,
        list(thresholds.keys()),
    )

    plt.close()


if __name__ == "__main__":
    run_ids = fusion_base_runs_best()
    # run_ids = [fusion_base_runs_best()[1]]
    analyze_and_plot(run_ids)
