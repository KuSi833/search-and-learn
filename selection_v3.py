#!/usr/bin/env python3
"""
Per-run analysis to find highest possible recall for detecting incorrect answers.
Focus on individual run performance with the 3 specific metrics.
Search for thresholds that make all metrics select similar amounts.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from sal.utils.runs import fusion_base_runs_best
from scripts.inference_visualiser import (
    _compute_uncertainty_metrics,
    _get_question_answer_from_record,
    load_jsonl,
)

# Metrics to analyze
METRICS = ["consensus_support", "agreement_ratio", "entropy_freq"]

# Speed/verbosity controls
FAST_MODE = True  # If True: analyze only 15% target and skip heavy tables
DEBUG_EXPORT = False  # If True: verbose export diagnostics

# Target selection percentages to test
TARGET_SELECTION_RATES = [10, 15, 20, 25, 30]  # percentages
ACTIVE_SELECTION_RATES = [15] if FAST_MODE else TARGET_SELECTION_RATES


def load_single_run_data(
    run_id: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[str]]:
    """Load data from a single run and separate by correctness."""
    correct_data = {metric: [] for metric in METRICS}
    incorrect_data = {metric: [] for metric in METRICS}
    all_data = {metric: [] for metric in METRICS}
    question_ids = []

    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)

    for idx, rec in enumerate(records):
        qa = _get_question_answer_from_record(rec)
        metrics = _compute_uncertainty_metrics(rec)

        question_id = f"{run_id}_{idx}"
        question_ids.append(question_id)

        for metric in METRICS:
            metric_value = metrics.get(metric, 0.0)
            all_data[metric].append(metric_value)

            if qa.is_correct:
                correct_data[metric].append(metric_value)
            else:
                incorrect_data[metric].append(metric_value)

    return correct_data, incorrect_data, all_data, question_ids


def determine_metric_direction(
    metric: str,
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
) -> str:
    """Determine if metric uses <= or >= threshold based on mean comparison."""
    correct_mean = np.mean(correct_data[metric]) if correct_data[metric] else 0.0
    incorrect_mean = np.mean(incorrect_data[metric]) if incorrect_data[metric] else 0.0

    # If incorrect answers have higher mean values, high values indicate uncertainty
    if incorrect_mean > correct_mean:
        return ">="
    else:
        # If incorrect answers have lower mean values, low values indicate uncertainty
        return "<="


def calculate_threshold_for_selection_rate(
    metric: str,
    all_data: Dict[str, List[float]],
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    target_rate: float,
) -> float:
    """Calculate threshold to achieve target selection rate."""
    direction = determine_metric_direction(metric, correct_data, incorrect_data)

    if direction == "<=":
        # Low values indicate uncertainty - use target percentile
        threshold = np.percentile(all_data[metric], target_rate)
    else:
        # High values indicate uncertainty - use (100 - target) percentile
        threshold = np.percentile(all_data[metric], 100 - target_rate)

    return threshold


def apply_threshold(
    metric: str, values: List[float], threshold: float, direction: str
) -> List[bool]:
    """Apply threshold to metric values."""
    if direction == "<=":
        return [val <= threshold for val in values]
    else:  # direction == ">="
        return [val >= threshold for val in values]


def get_selected_questions_with_threshold(
    metric: str,
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    threshold: float,
    direction: str,
) -> Tuple[Set[str], int, int]:
    """Get questions selected by a metric using specific threshold."""
    selected_indices = set()

    # Apply threshold to correct answers
    correct_selected = apply_threshold(
        metric, correct_data[metric], threshold, direction
    )
    for idx, is_selected in enumerate(correct_selected):
        if is_selected:
            selected_indices.add(f"correct_{idx}")

    # Apply threshold to incorrect answers
    incorrect_selected = apply_threshold(
        metric, incorrect_data[metric], threshold, direction
    )
    for idx, is_selected in enumerate(incorrect_selected):
        if is_selected:
            selected_indices.add(f"incorrect_{idx}")

    correct_count = sum(correct_selected)
    incorrect_count = sum(incorrect_selected)

    return selected_indices, correct_count, incorrect_count


def analyze_single_run_with_balanced_thresholds(
    run_id: str,
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
    all_data: Dict[str, List[float]],
    target_rate: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tuple[float, str]]]:
    """Analyze recall performance for a single run with balanced selection rates."""
    total_correct = len(correct_data[METRICS[0]])
    total_incorrect = len(incorrect_data[METRICS[0]])

    if total_incorrect == 0:
        return {}, {}

    # Calculate balanced thresholds for target selection rate
    thresholds = {}
    for metric in METRICS:
        threshold = calculate_threshold_for_selection_rate(
            metric, all_data, correct_data, incorrect_data, target_rate
        )
        direction = determine_metric_direction(metric, correct_data, incorrect_data)
        thresholds[metric] = (threshold, direction)

    # Get individual metric selections
    individual_selections = {}
    for metric in METRICS:
        threshold, direction = thresholds[metric]
        selections, correct_count, incorrect_count = (
            get_selected_questions_with_threshold(
                metric, correct_data, incorrect_data, threshold, direction
            )
        )
        individual_selections[metric] = {
            "selections": selections,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
        }

    # Define all combinations to test
    combinations = {
        "consensus_support": individual_selections["consensus_support"]["selections"],
        "agreement_ratio": individual_selections["agreement_ratio"]["selections"],
        "entropy_freq": individual_selections["entropy_freq"]["selections"],
        "consensus_support ∩ agreement_ratio": (
            individual_selections["consensus_support"]["selections"]
            & individual_selections["agreement_ratio"]["selections"]
        ),
        "consensus_support ∩ entropy_freq": (
            individual_selections["consensus_support"]["selections"]
            & individual_selections["entropy_freq"]["selections"]
        ),
        "agreement_ratio ∩ entropy_freq": (
            individual_selections["agreement_ratio"]["selections"]
            & individual_selections["entropy_freq"]["selections"]
        ),
        "all_three_intersection": (
            individual_selections["consensus_support"]["selections"]
            & individual_selections["agreement_ratio"]["selections"]
            & individual_selections["entropy_freq"]["selections"]
        ),
        "any_one_union": (
            individual_selections["consensus_support"]["selections"]
            | individual_selections["agreement_ratio"]["selections"]
            | individual_selections["entropy_freq"]["selections"]
        ),
    }

    results = {}

    for combo_name, selected_questions in combinations.items():
        # Count correct vs incorrect in selection
        correct_selected = len(
            [q for q in selected_questions if q.startswith("correct_")]
        )
        incorrect_selected = len(
            [q for q in selected_questions if q.startswith("incorrect_")]
        )

        # Calculate recall (most important metric)
        recall = incorrect_selected / total_incorrect if total_incorrect > 0 else 0.0

        # Calculate other metrics
        tp = incorrect_selected  # True positives: incorrect answers correctly flagged
        fp = correct_selected  # False positives: correct answers incorrectly flagged
        fn = (
            total_incorrect - incorrect_selected
        )  # False negatives: incorrect answers missed
        tn = (
            total_correct - correct_selected
        )  # True negatives: correct answers not flagged

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[combo_name] = {
            "run_id": run_id,
            "target_rate": target_rate,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_selected": tp + fp,
            "total_incorrect": total_incorrect,
            "total_correct": total_correct,
        }

    return results, thresholds


def create_threshold_analysis_table(
    run_id: str,
    all_results: Dict[
        float, Tuple[Dict[str, Dict[str, float]], Dict[str, Tuple[float, str]]]
    ],
    correct_data: Dict[str, List[float]],
    incorrect_data: Dict[str, List[float]],
):
    """Create analysis table showing threshold search results."""
    console = Console()

    total_questions = len(correct_data[METRICS[0]]) + len(incorrect_data[METRICS[0]])
    total_correct = len(correct_data[METRICS[0]])
    total_incorrect = len(incorrect_data[METRICS[0]])

    console.print(
        f"\n[bold yellow]═══ THRESHOLD SEARCH RESULTS: {run_id} ═══[/bold yellow]"
    )
    console.print(
        f"Total questions: [blue]{total_questions}[/blue], Correct: [green]{total_correct}[/green], Incorrect: [red]{total_incorrect}[/red]\n"
    )

    # Show thresholds for each target rate (FAST_MODE only shows active rate)
    threshold_table = Table(
        title=f"Balanced Thresholds - {run_id}",
        show_header=True,
        header_style="bold magenta",
    )

    threshold_table.add_column("Target %", justify="center", style="yellow")
    threshold_table.add_column("Consensus Support", justify="right", style="cyan")
    threshold_table.add_column("Agreement Ratio", justify="right", style="cyan")
    threshold_table.add_column("Entropy Freq", justify="right", style="cyan")
    threshold_table.add_column("Actual Selections", justify="right", style="dim blue")

    for target_rate in sorted(all_results.keys()):
        results, thresholds = all_results[target_rate]

        # Get actual selection counts for individual metrics
        cs_selected = results["consensus_support"]["total_selected"]
        ar_selected = results["agreement_ratio"]["total_selected"]
        ef_selected = results["entropy_freq"]["total_selected"]

        # Format thresholds with direction
        cs_threshold, cs_direction = thresholds["consensus_support"]
        ar_threshold, ar_direction = thresholds["agreement_ratio"]
        ef_threshold, ef_direction = thresholds["entropy_freq"]

        threshold_table.add_row(
            f"{target_rate:.0f}%",
            f"{cs_direction} {cs_threshold:.4f}",
            f"{ar_direction} {ar_threshold:.4f}",
            f"{ef_direction} {ef_threshold:.4f}",
            f"{cs_selected}, {ar_selected}, {ef_selected}",
        )

    console.print(threshold_table)

    # Find best performance for each target rate (both recall and F1)
    if not FAST_MODE:
        console.print(f"\n[bold]Best Performance by Target Rate:[/bold]")

    performance_table = None
    if not FAST_MODE:
        performance_table = Table(
            title=f"Best Performance by Selection Rate - {run_id}",
            show_header=True,
            header_style="bold magenta",
        )
        performance_table.add_column("Target %", justify="center", style="yellow")
        performance_table.add_column("Best Recall Method", style="cyan")
        performance_table.add_column("Recall", justify="right", style="bright_green")
        performance_table.add_column("Best F1 Method", style="magenta")
        performance_table.add_column("F1", justify="right", style="bright_magenta")
        performance_table.add_column("Detected", justify="right", style="green")
        performance_table.add_column("Selected", justify="right", style="blue")

    best_overall_recall = 0
    best_overall_f1 = 0
    best_recall_method = ""
    best_f1_method = ""
    best_recall_rate = 0
    best_f1_rate = 0

    for target_rate in sorted(all_results.keys()):
        results, _ = all_results[target_rate]

        # Find best recall method for this rate
        best_recall_method_rate = max(
            results.keys(), key=lambda k: results[k]["recall"]
        )
        best_recall_result = results[best_recall_method_rate]

        # Find best F1 method for this rate
        best_f1_method_rate = max(results.keys(), key=lambda k: results[k]["f1"])
        best_f1_result = results[best_f1_method_rate]

        # Track overall best
        recall_value = best_recall_result["recall"]
        f1_value = best_f1_result["f1"]

        if recall_value > best_overall_recall:
            best_overall_recall = recall_value
            best_recall_method = best_recall_method_rate
            best_recall_rate = target_rate

        if f1_value > best_overall_f1:
            best_overall_f1 = f1_value
            best_f1_method = best_f1_method_rate
            best_f1_rate = target_rate

        # Color code recall
        if recall_value >= 0.8:
            recall_style = "[bright_green]"
        elif recall_value >= 0.6:
            recall_style = "[yellow]"
        else:
            recall_style = "[red]"

        # Color code F1
        if f1_value >= 0.5:
            f1_style = "[bright_green]"
        elif f1_value >= 0.3:
            f1_style = "[yellow]"
        else:
            f1_style = "[red]"

        # Use recall result for detected/selected counts (could be different methods)
        if performance_table is not None:
            performance_table.add_row(
                f"{target_rate:.0f}%",
                best_recall_method_rate.replace("_", " ").title(),
                f"{recall_style}{recall_value:.3f}[/{recall_style.strip('[]')}]",
                best_f1_method_rate.replace("_", " ").title(),
                f"{f1_style}{f1_value:.3f}[/{f1_style.strip('[]')}]",
                f"{best_recall_result['tp']}/{total_incorrect}",
                str(best_recall_result["total_selected"]),
            )

    if performance_table is not None:
        console.print(performance_table)

    # Highlight overall best for both metrics
    console.print(f"\n[bold]🏆 Overall Best for {run_id}:[/bold]")
    console.print(f"[bold green]Best Recall:[/bold green]")
    console.print(
        f"  Method: [bright_green]{best_recall_method.replace('_', ' ').title()}[/bright_green]"
    )
    console.print(f"  Target Rate: [yellow]{best_recall_rate:.0f}%[/yellow]")
    console.print(f"  Recall: [bright_green]{best_overall_recall:.3f}[/bright_green]")

    console.print(f"[bold magenta]Best F1:[/bold magenta]")
    console.print(
        f"  Method: [bright_magenta]{best_f1_method.replace('_', ' ').title()}[/bright_magenta]"
    )
    console.print(f"  Target Rate: [yellow]{best_f1_rate:.0f}%[/yellow]")
    console.print(f"  F1: [bright_magenta]{best_overall_f1:.3f}[/bright_magenta]")

    return (
        best_overall_recall,
        best_recall_method,
        best_recall_rate,
        best_overall_f1,
        best_f1_method,
        best_f1_rate,
    )


def create_cross_run_summary_balanced(
    all_run_best_results: Dict[str, Tuple[float, str, float, float, str, float]],
):
    """Create summary comparing best balanced performance across runs."""
    console = Console()

    console.print(f"\n[bold cyan]CROSS-RUN SUMMARY (BALANCED THRESHOLDS)[/bold cyan]")
    console.print(
        "Best recall and F1 performance for each run with balanced selection rates\n"
    )

    # Create summary table
    summary_table = Table(
        title="Best Balanced Performance Per Run",
        show_header=True,
        header_style="bold magenta",
    )

    summary_table.add_column("Run ID", style="cyan")
    summary_table.add_column("Best Recall Method", style="green")
    summary_table.add_column("Recall", justify="right", style="bright_green")
    summary_table.add_column("Rate", justify="right", style="dim green")
    summary_table.add_column("Best F1 Method", style="magenta")
    summary_table.add_column("F1", justify="right", style="bright_magenta")
    summary_table.add_column("Rate", justify="right", style="dim magenta")

    recalls = []
    f1_scores = []
    recall_rates = []
    f1_rates = []
    recall_method_counts = {}
    f1_method_counts = {}

    for run_id, (
        recall,
        recall_method,
        recall_rate,
        f1,
        f1_method,
        f1_rate,
    ) in all_run_best_results.items():
        recalls.append(recall)
        f1_scores.append(f1)
        recall_rates.append(recall_rate)
        f1_rates.append(f1_rate)

        recall_method_counts[recall_method] = (
            recall_method_counts.get(recall_method, 0) + 1
        )
        f1_method_counts[f1_method] = f1_method_counts.get(f1_method, 0) + 1

        # Color code recall
        if recall >= 0.8:
            recall_style = "[bright_green]"
        elif recall >= 0.6:
            recall_style = "[yellow]"
        else:
            recall_style = "[red]"

        # Color code F1
        if f1 >= 0.5:
            f1_style = "[bright_green]"
        elif f1 >= 0.3:
            f1_style = "[yellow]"
        else:
            f1_style = "[red]"

        summary_table.add_row(
            run_id,
            recall_method.replace("_", " ").title(),
            f"{recall_style}{recall:.3f}[/{recall_style.strip('[]')}]",
            f"{recall_rate:.0f}%",
            f1_method.replace("_", " ").title(),
            f"{f1_style}{f1:.3f}[/{f1_style.strip('[]')}]",
            f"{f1_rate:.0f}%",
        )

    console.print(summary_table)

    # Overall statistics
    if recalls and f1_scores:
        avg_recall = np.mean(recalls)
        best_recall = max(recalls)
        worst_recall = min(recalls)
        avg_recall_rate = np.mean(recall_rates)

        avg_f1 = np.mean(f1_scores)
        best_f1 = max(f1_scores)
        worst_f1 = min(f1_scores)
        avg_f1_rate = np.mean(f1_rates)

        console.print(f"\n[bold]Overall Statistics:[/bold]")

        console.print(f"[bold green]Recall Performance:[/bold green]")
        console.print(
            f"• Average best recall: [bright_blue]{avg_recall:.3f}[/bright_blue]"
        )
        console.print(
            f"• Best recall achieved: [bright_green]{best_recall:.3f}[/bright_green]"
        )
        console.print(f"• Worst recall: [red]{worst_recall:.3f}[/red]")
        console.print(
            f"• Average optimal recall rate: [yellow]{avg_recall_rate:.1f}%[/yellow]"
        )

        console.print(f"\n[bold magenta]F1 Performance:[/bold magenta]")
        console.print(f"• Average best F1: [bright_blue]{avg_f1:.3f}[/bright_blue]")
        console.print(f"• Best F1 achieved: [bright_green]{best_f1:.3f}[/bright_green]")
        console.print(f"• Worst F1: [red]{worst_f1:.3f}[/red]")
        console.print(f"• Average optimal F1 rate: [yellow]{avg_f1_rate:.1f}%[/yellow]")

        console.print(f"\n[bold]Best Recall Method Frequency:[/bold]")
        for method, count in sorted(
            recall_method_counts.items(), key=lambda x: x[1], reverse=True
        ):
            console.print(
                f"• {method.replace('_', ' ').title()}: [yellow]{count}[/yellow] runs"
            )

        console.print(f"\n[bold]Best F1 Method Frequency:[/bold]")
        for method, count in sorted(
            f1_method_counts.items(), key=lambda x: x[1], reverse=True
        ):
            console.print(
                f"• {method.replace('_', ' ').title()}: [yellow]{count}[/yellow] runs"
            )


def export_union_15_percent_dataset(
    run_ids: List[str], output_file: str = "union_15pct_dataset.jsonl"
):
    """Export dataset with Any One Union at 15% selection rate using the EXACT same analysis logic."""
    console = Console()

    console.print(f"[bold cyan]EXPORTING UNION 15% DATASET[/bold cyan]")
    console.print(f"Target: Any One Union at 15% selection rate")
    console.print(f"Output file: [yellow]{output_file}[/yellow]\n")

    console.print(
        "[bold red]DEBUG: Let me first re-run the analysis to get the exact performance...[/bold red]"
    )

    all_selected_questions = []
    total_questions = 0
    total_selected = 0
    total_correct_selected = 0
    total_incorrect_selected = 0

    for run_id in run_ids:
        console.print(f"Processing run: [cyan]{run_id}[/cyan]")

        # Load data for this run
        correct_data, incorrect_data, all_data, question_ids = load_single_run_data(
            run_id
        )

        console.print(f"  [bold blue]Data loading check:[/bold blue]")
        console.print(
            f"    Correct questions in analysis: {len(correct_data[METRICS[0]])}"
        )
        console.print(
            f"    Incorrect questions in analysis: {len(incorrect_data[METRICS[0]])}"
        )

        # Re-run the EXACT analysis for 15% to see what we get
        target_rate = 15.0
        results, thresholds = analyze_single_run_with_balanced_thresholds(
            run_id, correct_data, incorrect_data, all_data, target_rate
        )

        # Show what the analysis says for "any_one_union"
        union_result = results.get("any_one_union", {})
        console.print(f"  [bold]Analysis says Union at 15% should give:[/bold]")
        console.print(f"    Recall: {union_result.get('recall', 0):.3f}")
        console.print(f"    Precision: {union_result.get('precision', 0):.3f}")
        console.print(f"    Selected: {union_result.get('total_selected', 0)}")
        console.print(f"    TP (incorrect detected): {union_result.get('tp', 0)}")
        console.print(f"    FP (correct selected): {union_result.get('fp', 0)}")

        console.print(f"  Thresholds for {run_id}:")
        for metric, (threshold, direction) in thresholds.items():
            console.print(f"    {metric}: {direction} {threshold:.4f}")

        # Load original records to get question details
        out_file = Path("./output") / run_id / "inference_output.jsonl"
        records = load_jsonl(out_file)

        console.print(f"  [bold blue]Record loading check:[/bold blue]")
        console.print(f"    Total records in JSONL: {len(records)}")

        # Debug: Count actual correct/incorrect in records
        record_correct_count = 0
        record_incorrect_count = 0
        for rec in records:
            qa = _get_question_answer_from_record(rec)
            if qa.is_correct:
                record_correct_count += 1
            else:
                record_incorrect_count += 1

        console.print(f"    Correct in records: {record_correct_count}")
        console.print(f"    Incorrect in records: {record_incorrect_count}")

        # Check if there's a mismatch in data loading
        if record_correct_count != len(correct_data[METRICS[0]]):
            console.print(f"  [bold red]ERROR: Correct count mismatch![/bold red]")
            console.print(f"    Analysis data: {len(correct_data[METRICS[0]])}")
            console.print(f"    Record data: {record_correct_count}")

        if record_incorrect_count != len(incorrect_data[METRICS[0]]):
            console.print(f"  [bold red]ERROR: Incorrect count mismatch![/bold red]")
            console.print(f"    Analysis data: {len(incorrect_data[METRICS[0]])}")
            console.print(f"    Record data: {record_incorrect_count}")

        run_selected = 0
        run_correct_selected = 0
        run_incorrect_selected = 0

        # Debug: Track what we're checking
        debug_checks = {
            "correct_checked": 0,
            "incorrect_checked": 0,
            "correct_selected": 0,
            "incorrect_selected": 0,
        }

        for idx, rec in enumerate(records):
            qa = _get_question_answer_from_record(rec)
            metrics = _compute_uncertainty_metrics(rec)

            # Debug counts
            if qa.is_correct:
                debug_checks["correct_checked"] += 1
            else:
                debug_checks["incorrect_checked"] += 1

            # Apply thresholds directly per record and metric
            per_metric_selected = {}
            for metric in METRICS:
                value = metrics.get(metric, 0.0)
                thresh, direction = thresholds[metric]
                if direction == "<=":
                    per_metric_selected[metric] = value <= thresh
                else:
                    per_metric_selected[metric] = value >= thresh

            is_selected = any(per_metric_selected.values())

            if is_selected:
                if qa.is_correct:
                    debug_checks["correct_selected"] += 1
                else:
                    debug_checks["incorrect_selected"] += 1

                # Add selection metadata
                selected_record = rec.copy()
                selected_record["selection_metadata"] = {
                    "run_id": run_id,
                    "question_index": idx,
                    "selection_method": "any_one_union",
                    "target_rate": target_rate,
                    "is_correct": bool(qa.is_correct),
                    "thresholds_used": {
                        metric: {
                            "threshold": float(thresh),
                            "direction": str(direction),
                        }
                        for metric, (thresh, direction) in thresholds.items()
                    },
                    "metric_values": {
                        metric: float(metrics.get(metric, 0.0)) for metric in METRICS
                    },
                    "individual_selections": {
                        metric: bool(per_metric_selected.get(metric, False))
                        for metric in METRICS
                    },
                }

                all_selected_questions.append(selected_record)
                run_selected += 1

                if qa.is_correct:
                    run_correct_selected += 1
                else:
                    run_incorrect_selected += 1

        total_questions += len(records)
        total_selected += run_selected
        total_correct_selected += run_correct_selected
        total_incorrect_selected += run_incorrect_selected

        console.print(f"  [bold blue]Debug check results:[/bold blue]")
        console.print(
            f"    Correct questions checked: {debug_checks['correct_checked']}"
        )
        console.print(
            f"    Incorrect questions checked: {debug_checks['incorrect_checked']}"
        )
        console.print(
            f"    Correct questions selected: {debug_checks['correct_selected']}"
        )
        console.print(
            f"    Incorrect questions selected: {debug_checks['incorrect_selected']}"
        )

        console.print(f"  [bold]ACTUAL export results:[/bold]")
        console.print(
            f"    Selected: {run_selected}/{len(records)} ({run_selected / len(records) * 100:.1f}%)"
        )
        console.print(
            f"    Correct: {run_correct_selected}, Incorrect: {run_incorrect_selected}"
        )

        if run_incorrect_selected + run_correct_selected > 0:
            precision = run_incorrect_selected / (
                run_incorrect_selected + run_correct_selected
            )
            console.print(f"    Precision: {precision:.3f}")

        total_incorrect_in_run = len(incorrect_data[METRICS[0]])
        if total_incorrect_in_run > 0:
            recall = run_incorrect_selected / total_incorrect_in_run
            console.print(f"    Recall: {recall:.3f}")

        # Compare with analysis prediction
        expected_tp = union_result.get("tp", 0)
        expected_fp = union_result.get("fp", 0)
        console.print(f"  [bold red]MISMATCH CHECK:[/bold red]")
        console.print(f"    Expected TP: {expected_tp}, Got: {run_incorrect_selected}")
        console.print(f"    Expected FP: {expected_fp}, Got: {run_correct_selected}")

        console.print()

    # Write to JSONL file
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        for record in all_selected_questions:
            f.write(json.dumps(record) + "\n")

    # Print summary
    console.print(f"\n[bold green]EXPORT COMPLETE![/bold green]")
    console.print(f"Output file: [yellow]{output_path.absolute()}[/yellow]")
    console.print(f"Total questions processed: [blue]{total_questions}[/blue]")
    console.print(
        f"Total selected: [blue]{total_selected}[/blue] ({total_selected / total_questions * 100:.1f}%)"
    )
    console.print(f"Selected breakdown:")
    console.print(f"  Correct: [green]{total_correct_selected}[/green]")
    console.print(f"  Incorrect: [red]{total_incorrect_selected}[/red]")

    if total_selected > 0:
        overall_precision = total_incorrect_selected / total_selected
        console.print(
            f"Overall precision: [bright_blue]{overall_precision:.3f}[/bright_blue]"
        )

    console.print(f"\n[bold]Each record includes:[/bold]")
    console.print("• Original question and response data")
    console.print("• selection_metadata with:")
    console.print("  - Selection method and parameters")
    console.print("  - Threshold values used")
    console.print("  - Individual metric values")
    console.print("  - Which individual metrics selected it")
    console.print("  - Correctness information")


def main():
    """Main analysis function."""
    console = Console()

    # Get the same runs as selection_2.py
    run_ids = fusion_base_runs_best()
    console.print(
        f"[bold]Analyzing runs with balanced thresholds:[/bold] {', '.join(run_ids)}"
    )
    console.print(
        f"[bold]Goal:[/bold] Find optimal recall with balanced selection rates\n"
    )
    console.print(
        f"[bold]Target selection rates to test:[/bold] {ACTIVE_SELECTION_RATES}%\n"
    )

    all_run_best_results = {}

    # Analyze each run separately
    for run_id in run_ids:
        console.print(f"Processing run: [cyan]{run_id}[/cyan]")

        # Load data for this run only
        correct_data, incorrect_data, all_data, question_ids = load_single_run_data(
            run_id
        )

        total_questions = len(correct_data[METRICS[0]]) + len(
            incorrect_data[METRICS[0]]
        )
        console.print(f"  Questions: {total_questions}")
        console.print(
            f"  Correct: {len(correct_data[METRICS[0]])}, Incorrect: {len(incorrect_data[METRICS[0]])}"
        )

        # Test different target selection rates (FAST_MODE limits to 15%)
        all_results = {}
        for target_rate in ACTIVE_SELECTION_RATES:
            results, thresholds = analyze_single_run_with_balanced_thresholds(
                run_id, correct_data, incorrect_data, all_data, target_rate
            )
            all_results[target_rate] = (results, thresholds)

        # Create detailed analysis for this run
        (
            best_recall,
            best_recall_method,
            best_recall_rate,
            best_f1,
            best_f1_method,
            best_f1_rate,
        ) = create_threshold_analysis_table(
            run_id, all_results, correct_data, incorrect_data
        )

        all_run_best_results[run_id] = (
            best_recall,
            best_recall_method,
            best_recall_rate,
            best_f1,
            best_f1_method,
            best_f1_rate,
        )

    # Create cross-run summary (skip in FAST_MODE)
    if not FAST_MODE:
        create_cross_run_summary_balanced(all_run_best_results)

    console.print("\n[bold green]Balanced threshold analysis complete![/bold green]")

    # In FAST_MODE export immediately; otherwise ask
    if FAST_MODE:
        export_union_15_percent_dataset(run_ids)
    else:
        console.print(f"\n[bold yellow]Export Union 15% Dataset?[/bold yellow]")
        console.print(
            "Based on your results, Any One Union at 15% shows excellent performance."
        )
        console.print("Would you like to export this dataset? (y/n): ", end="")

        try:
            response = input().strip().lower()
            if response in ["y", "yes"]:
                export_union_15_percent_dataset(run_ids)
        except KeyboardInterrupt:
            console.print("\n[yellow]Export cancelled.[/yellow]")


if __name__ == "__main__":
    main()
