#!/usr/bin/env python3
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sal.evaluation.grader import math_equal
from sal.evaluation.parser import extract_answer, find_box
from sal.utils.constants import BENCHMARK_SUBSETS_ROOT, Benchmark, Benchmarks
from sal.utils.data import BenchmarkMapping
from sal.utils.logging import setup_logging

setup_logging()


@dataclass
class QuestionAnswer:
    unique_id: str
    answer_extracted: str
    pred_extracted: str
    is_correct: bool
    level: str


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Best effort: skip malformed lines
                continue
    return records


def _shorten(text: str, max_len: int = 280) -> str:
    text = text.replace("\n\n", "\n").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_float_list(xs: List[float], limit: int = 16) -> str:
    if not xs:
        return "[]"
    view = xs if len(xs) <= limit else xs[: limit - 1] + [xs[-1]]
    parts = [f"{x:.4f}" for x in view]
    if len(xs) > limit:
        # Indicate we skipped middle elements
        parts = parts[:-1] + ["...", parts[-1]]
    return "[" + ", ".join(parts) + "]"


def _index_of_first(seq: List[str], target: str) -> int:
    try:
        return seq.index(target)
    except ValueError:
        return -1


console = Console()


# Default assumed prediction key used across the tool (ensemble-style). Assumes weighted@4.
ASSUMED_PRED_KEY = "pred_weighted@4"
BENCHMARK = "math"


def _split_steps(text: str) -> List[str]:
    # Split on double newlines which demarcate steps in generation
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def _score_style(value: float) -> str:
    try:
        return "green" if float(value) >= 0 else "red"
    except Exception:
        return "white"


def print_report(
    run_id: str,
    sample: Dict[str, Any],
) -> None:
    problem: str = sample["problem"]
    solution: str = sample["solution"]
    answer: str = sample["answer"]
    unique_id: str = sample["unique_id"]
    subject: str = sample["subject"]
    level: int = sample["level"]
    # The raw chosen completion text from the search method (e.g. best-of-n/beam)
    pred_text: str = sample["pred"]
    pred = sample[ASSUMED_PRED_KEY]  # just the prediction
    # Optional fields produced by the search + scoring pipeline
    completions: List[str] = sample["completions"]
    # PRM score trajectories per completion (list of floats per completion)
    scores: List[List[float]] = sample["scores"]
    # Final per-completion PRM scores computed during scoring (aggregate "last")
    agg_scores: Optional[List[float]] = sample["agg_scores"]

    SHOW_PRED_TEXT = False
    SHOW_SOLUTION_TEXT = False

    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("File", f"output/{run_id}/inference_output.jsonl")
    console.print(Panel(header, title="Inference Sample", box=box.ROUNDED))

    # Assuming I am actually wondering about ASSUMED_PRED_KEY accuracy
    answer_extracted = extract_answer(_wrap_in_boxed(answer), BENCHMARK)
    pred_raw = find_box(pred_text)
    pred_extracted = extract_answer(pred, BENCHMARK)

    assumed_correct = math_equal(answer_extracted, pred_extracted)

    meta_bits = [
        f"id={unique_id}",
        f"subject={subject}",
        f"level={level}",
    ]
    meta_bits = [b for b in meta_bits if b]
    if meta_bits:
        console.print(Text("Meta: " + ", ".join(meta_bits), style="dim"))

    console.print(Panel(_shorten(problem, 800), title="Problem", box=box.SQUARE))

    console.print(
        Panel(_shorten(answer, 200), title="Ground truth answer", style="green")
    )

    extracted_table = Table(
        title="Answer extraction & correctness", box=box.SIMPLE_HEAVY
    )
    extracted_table.add_column("Field", style="bold")
    extracted_table.add_column("Value")
    extracted_table.add_row("Assumed pred key", ASSUMED_PRED_KEY)
    extracted_table.add_row("Prediction (Raw)", pred_raw)
    extracted_table.add_row("Prediction (Extracted)", pred_extracted)
    extracted_table.add_row(
        "Correct?",
        Text(str(assumed_correct), style=("green" if assumed_correct else "red")),
    )
    console.print(Panel(extracted_table, box=box.ROUNDED))

    # Uncertainty metrics - the 3 core metrics from metric_overlap.py
    uncertainty_metrics = _compute_uncertainty_metrics(sample)
    metrics_table = Table(title="Core uncertainty metrics", box=box.SIMPLE_HEAVY)
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_column("Description", style="dim")

    metrics_table.add_row(
        "agreement_ratio",
        f"{uncertainty_metrics['agreement_ratio']:.3f}",
        "Fraction agreeing on most common answer",
    )
    metrics_table.add_row(
        "entropy_freq",
        f"{uncertainty_metrics['entropy_freq']:.3f}",
        "Normalized entropy of answer frequencies",
    )
    metrics_table.add_row(
        "consensus_support",
        f"{uncertainty_metrics['consensus_support']:.3f}",
        "PRM score fraction for top answer group",
    )
    console.print(Panel(metrics_table, box=box.ROUNDED))

    # Per-completion summary: extracted final answer, final PRM score, and correctness
    # - Final PRM score: prefer pre-computed `agg_scores` (added during scoring),
    #   otherwise fall back to the last value of each score path.
    if completions:
        completions_table = Table(title="Per-completion summary", box=box.SIMPLE_HEAVY)
        completions_table.add_column("#", style="bold cyan", justify="right")
        completions_table.add_column("Chosen", justify="center")
        completions_table.add_column("Final answer (extracted)")
        completions_table.add_column("PRM final", justify="right")
        completions_table.add_column("Correct", justify="center")

        # Compute per-completion extracted answers and correctness
        extracted_answers: List[str] = [
            extract_answer(c or "", BENCHMARK) for c in completions
        ]

        # Resolve final PRM scores per completion
        final_scores: List[float] = []
        if isinstance(agg_scores, list) and len(agg_scores) == len(completions):
            try:
                final_scores = [float(s) for s in agg_scores]
            except Exception:
                final_scores = []
        if not final_scores:
            # Fallback to last score from each score trajectory
            try:
                final_scores = [
                    (float(s[-1]) if isinstance(s, list) and len(s) > 0 else 0.0)
                    for s in scores
                ]
            except Exception:
                final_scores = [0.0 for _ in completions]

        # Mark the search-selected completion if we can find it
        chosen_idx = _index_of_first(completions, pred_text)

        for i, (ans, fscore) in enumerate(zip(extracted_answers, final_scores)):
            is_correct = math_equal(answer_extracted, ans)
            is_chosen = i == chosen_idx
            completions_table.add_row(
                str(i + 1),
                Text("✓" if is_chosen else "", style=("yellow" if is_chosen else "")),
                ans,
                Text(f"{fscore:+.4f}", style=_score_style(fscore)),
                Text(
                    "✓" if is_correct else "✗", style=("green" if is_correct else "red")
                ),
            )

        console.print(completions_table)

    if SHOW_PRED_TEXT:
        console.print(Panel(pred_text, title="Chosen prediction (raw)", style="yellow"))

    # chosen_idx = sample.get("chosen_idx")
    # num_beams = sample.get("num_beams")
    # csp = sample.get("chosen_score_path") or []
    # ctok = sample.get("chosen_tokens")

    # beams_table = Table(show_header=False, box=box.SIMPLE)
    # beams_table.add_column("key", style="bold")
    # beams_table.add_column("value")
    # beams_table.add_row("Beams", str(num_beams))
    # beams_table.add_row("Chosen index", str(chosen_idx))
    # beams_table.add_row(
    #     "Score trajectory", Text(_format_float_list(list(csp)), style="magenta")
    # )
    # if ctok is not None:
    #     beams_table.add_row("Completion tokens", str(ctok))
    # console.print(Panel(beams_table, title="Beam details", box=box.ROUNDED))

    # Visualise solution steps with scores
    steps = _split_steps(pred_text)
    step_scores: List[float] = []
    try:
        step_scores = [float(x) for x in (sample.get("chosen_score_path") or [])]
    except Exception:
        step_scores = []
    n_rows = min(len(steps), len(step_scores)) if step_scores else len(steps)
    if steps:
        steps_table = Table(box=box.MINIMAL_HEAVY_HEAD, show_lines=True)
        steps_table.add_column("#", style="bold cyan", justify="right")
        steps_table.add_column("Step", overflow="fold")
        steps_table.add_column("Score", justify="right")
        steps_table.add_column("Mean", justify="right")
        steps_table.add_column("Min", justify="right")
        steps_table.add_column("Last", justify="right")
        steps_table.add_column("Prod", justify="right")

        running_sum: float = 0.0
        running_min: Optional[float] = None
        running_prod: Optional[float] = None
        for i in range(n_rows):
            s_text = steps[i]
            if step_scores:
                s_val = float(step_scores[i])
                running_sum += s_val
                running_min = s_val if running_min is None else min(running_min, s_val)
                running_prod = s_val if running_prod is None else running_prod * s_val
                mean_val = running_sum / (i + 1)
                last_val = s_val
                steps_table.add_row(
                    str(i + 1),
                    s_text,
                    Text(f"{s_val:+.4f}", style=_score_style(s_val)),
                    Text(f"{mean_val:+.4f}", style=_score_style(mean_val)),
                    Text(
                        f"{(running_min if running_min is not None else 0.0):+.4f}",
                        style=_score_style(
                            running_min if running_min is not None else 0.0
                        ),
                    ),
                    Text(f"{last_val:+.4f}", style=_score_style(last_val)),
                    Text(
                        f"{(running_prod if running_prod is not None else 0.0):+.4f}",
                        style=_score_style(
                            running_prod if running_prod is not None else 0.0
                        ),
                    ),
                )
            else:
                steps_table.add_row(str(i + 1), s_text, "-", "-", "-", "-", "-")

        # If needed, we can display a subtitle explaining length mismatch between steps and scores
        # note: Optional[str] = None
        # if step_scores and len(steps) != len(step_scores):
        #     note = f"Note: steps ($${len(steps)}$$) and scores ($${len(step_scores)}$$) lengths differ; showing first $${n_rows}$$."
        # Uncomment to print solution steps
        # console.print(
        #     Panel(steps_table, title="Solution steps", subtitle=note, box=box.ROUNDED)
        # )

    # rank = sample.get("rank_by_last_score") or []
    # if rank:
    #     top = rank[: min(5, len(rank))]
    #     rank_table = Table(title="Top beams by last score", box=box.MINIMAL_HEAVY_HEAD)
    #     rank_table.add_column("#", style="bold cyan", justify="right")
    #     rank_table.add_column("score", justify="right")
    #     for i, s in top:
    #         rank_table.add_row(str(i), f"{s:.4f}")
    #     console.print(rank_table)

    # agg_preds = sample.get("agg_preds") or []
    # agg_correct = {k: v for k, v in (sample.get("agg_correct") or [])}
    # if agg_preds:
    #     agg_table = Table(title="Aggregated predictions", box=box.SIMPLE_HEAVY)
    #     agg_table.add_column("key", style="bold")
    #     agg_table.add_column("inner")
    #     agg_table.add_column("correct", justify="center")
    #     for k, v in agg_preds:
    #         inner = find_box(v) if isinstance(v, str) else ""
    #         ok = agg_correct.get(k, False)
    #         agg_table.add_row(
    #             k, inner, Text("✓" if ok else "✗", style=("green" if ok else "red"))
    #         )
    #     console.print(agg_table)

    if SHOW_SOLUTION_TEXT:
        console.print(
            Panel(
                solution,
                title="Reference solution",
                box=box.SQUARE,
            )
        )


@click.group(help="Inference visualiser utilities")
def cli() -> None:
    pass


@cli.command(name="detail", help="Inspect a single inference sample for debugging")
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
@click.option("--index", required=True, type=int, help="Index of the sample to inspect")
def cmd_detail(run_id: str, index: int) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"

    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        sys.exit(1)
    if index < 0 or index >= len(records):
        console.print(
            Text(
                f"Index out of range: {index} (num records = {len(records)})",
                style="red",
            )
        )
        sys.exit(1)

    sample = records[index]
    print_report(run_id, sample)


def _wrap_in_boxed(s: str) -> str:
    return r"\boxed{" + s + "}"


def _get_question_answer_from_record(rec: Dict[str, Any]) -> QuestionAnswer:
    level = rec["level"]
    answer = rec["answer"]
    unique_id: str = rec["unique_id"]
    pred = rec[ASSUMED_PRED_KEY]

    answer_extracted = extract_answer(_wrap_in_boxed(answer), BENCHMARK)
    pred_extracted = extract_answer(pred, BENCHMARK)

    is_correct = math_equal(answer_extracted, pred_extracted)

    return QuestionAnswer(
        unique_id, answer_extracted, pred_extracted, is_correct, level
    )


def _shorted_unique_name_math(name: str) -> str:
    _, subject, idx_with_json = name.split("/")
    idx = idx_with_json.split(".")[0]
    return f"{subject}/{idx}"


@cli.command(
    name="overview", help="Overview of a run by difficulty with correctness and indices"
)
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
def cmd_overview(run_id: str) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        sys.exit(1)

    level_to_total: Dict[str, int] = defaultdict(int)
    level_to_correct: Dict[str, int] = defaultdict(int)
    level_to_qa: Dict[str, List[QuestionAnswer]] = defaultdict(list)

    for rec in records:
        qa = _get_question_answer_from_record(rec)
        level_to_total[qa.level] += 1
        if qa.is_correct:
            level_to_correct[qa.level] += 1
        level_to_qa[qa.level].append(qa)

    console.print(
        Panel(Text("Run overview", style="bold"), subtitle=f"{run_id}", box=box.ROUNDED)
    )
    summary = Table(box=box.MINIMAL_HEAVY_HEAD)
    summary.add_column("Level", style="bold cyan")
    summary.add_column("Total", justify="right")
    summary.add_column("Correct", justify="right")
    summary.add_column("Incorrect", justify="right")
    summary.add_column("Acc %", justify="right")
    for level in sorted(level_to_total.keys()):
        total = level_to_total[level]
        correct = level_to_correct[level]
        incorrect = total - correct
        acc = (100.0 * correct / total) if total > 0 else 0.0
        summary.add_row(
            str(level),
            str(total),
            Text(str(correct), style="green"),
            Text(str(incorrect), style="red"),
            f"{acc:.1f}",
        )
    console.print(summary)

    benchmark_mapping = BenchmarkMapping(Benchmarks.MATH500.value)

    for level in sorted(level_to_total.keys()):
        total = level_to_total[level]
        correct = level_to_correct[level]
        incorrect = total - correct
        title = f"Level: {level}  |  Total: {total}  Correct: {correct}  Incorrect: {incorrect}"

        correct_idxs = ", ".join(
            [
                benchmark_mapping.get_index(qa.unique_id)
                for qa in level_to_qa[level]
                if qa.is_correct
            ]
        )
        incorrect_idxs = ", ".join(
            [
                benchmark_mapping.get_index(qa.unique_id)
                for qa in level_to_qa[level]
                if not qa.is_correct
            ]
        )

        idx_table = Table.grid(padding=(0, 1))
        idx_table.add_column(style="bold green")
        idx_table.add_column()
        idx_table.add_row(
            "Correct indices",
            Text(correct_idxs if correct_idxs else "[]", style="green"),
        )
        idx_table.add_row(
            "Incorrect indices",
            Text(incorrect_idxs if incorrect_idxs else "[]", style="red"),
        )
        console.print(Panel(idx_table, title=title, box=box.SQUARE))


@cli.command(name="question-answer")
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
@click.option(
    "--show-correct",
    is_flag=True,
    help="Show correct answers as well as incorrect ones",
)
def question_answer(run_id: str, show_correct: bool) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)

    # Organise incorrect answers by level
    level_to_incorrect: Dict[str, List[QuestionAnswer]] = defaultdict(list)

    for rec in records:
        qa = _get_question_answer_from_record(rec)
        level_to_incorrect[qa.level].append(qa)

    # Print organised by level
    for level in sorted(level_to_incorrect.keys()):
        console.print(f"\n[bold cyan]Level {level}:[/bold cyan]")
        for qa in sorted(level_to_incorrect[level], key=lambda qa: qa.unique_id):
            if qa.is_correct and not show_correct:
                continue
            color = "green" if qa.is_correct else "red"
            equality_symbol = "==" if qa.is_correct else "!="
            console.print(
                Text.assemble(
                    (
                        f"{qa.answer_extracted} {equality_symbol} {qa.pred_extracted}",
                        color,
                    ),
                    (f" {qa.unique_id}", "dim"),
                )
            )


@cli.command(name="extract-incorrect")
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
@click.option(
    "--benchmark",
    "benchmark_key",
    default=Benchmarks.MATH500.value.key,
    type=click.Choice([b.value.key for b in Benchmarks]),
    help="Benchmark to use for answer extraction",
)
@click.option("--name", required=True, type=str, help="Name for the subset file")
def extract_incorrect(run_id: str, benchmark_key: str, name: str) -> None:
    benchmark: Benchmark = Benchmarks.from_key(benchmark_key)
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)

    # Collect unique_ids of incorrectly answered questions
    incorrect_unique_ids = []

    for rec in records:
        qa = _get_question_answer_from_record(rec)
        if not qa.is_correct:
            incorrect_unique_ids.append(qa.unique_id)

    # Sort the unique_ids for consistency
    incorrect_unique_ids.sort()

    # Save to JSON file with metadata to support later mapping and reproducibility
    output_dir = BENCHMARK_SUBSETS_ROOT / benchmark.hf_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{name}.json"

    subset_payload = {
        "version": 1,
        "type": "incorrect_subset",
        "benchmark_key": benchmark.key,
        "hf_name": benchmark.hf_name,
        "run_id": run_id,
        "unique_ids": sorted(incorrect_unique_ids),
    }

    with open(output_file, "w") as f:
        json.dump(subset_payload, f, indent=2)

    console.print(
        Text.assemble(
            f"Saved {len(incorrect_unique_ids)} incorrect unique_ids to ",
            (str(output_file), "yellow"),
        )
    )


def _metric_direction_low_is_uncertain(values: List[float], labels: List[bool]) -> bool:
    """Infer whether low values mean uncertainty by comparing means by class."""
    corr_vals = [v for v, y in zip(values, labels) if y]
    inc_vals = [v for v, y in zip(values, labels) if not y]
    mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
    mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
    return mean_corr > mean_inc


def _select_uncertain_indices(
    records: List[Dict[str, Any]], coverage_pct: float, metric_name: str
) -> List[int]:
    metrics_list: List[Dict[str, Any]] = []
    labels: List[bool] = []
    for rec in records:
        qa = _get_question_answer_from_record(rec)
        labels.append(bool(qa.is_correct))
        metrics_list.append(_compute_uncertainty_metrics(rec))

    vals = [m.get(metric_name, 0.0) for m in metrics_list]
    low_is_uncertain = _metric_direction_low_is_uncertain(vals, labels)

    idxs = list(range(len(records)))
    # sort by increasing (low uncertain) or decreasing (high uncertain)
    idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)
    k = max(1, int(round(len(records) * (coverage_pct / 100.0))))
    return idxs[:k]


@cli.command(
    name="export-uncertain",
    help="Export unique_ids for the top coverage%% most uncertain questions",
)
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
@click.option(
    "--coverage",
    required=True,
    type=float,
    help="Percentage coverage to export (e.g. 20 for top 20%%)",
)
@click.option(
    "--metric",
    default="agreement_ratio",
    type=click.Choice(
        [
            "agreement_ratio",
            "entropy_freq",
            "entropy_weighted",
            "prm_margin",
            "prm_top_frac",
            "consensus_support",
            "prm_std",
            "prm_mean",
        ]
    ),
    help="Uncertainty metric to rank by",
)
@click.option(
    "--benchmark",
    "benchmark_key",
    default=Benchmarks.MATH500.value.key,
    type=click.Choice([b.value.key for b in Benchmarks]),
    help="Benchmark to use for answer extraction",
)
@click.option(
    "--name",
    required=False,
    default=None,
    type=str,
    help="Optional custom name; default path is <hf_name>/<run_id>/<coverage>.json",
)
def export_uncertain(
    run_id: str, coverage: float, metric: str, benchmark_key: str, name: Optional[str]
) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        sys.exit(1)

    selected = _select_uncertain_indices(records, coverage, metric)
    unique_ids = [records[i]["unique_id"] for i in selected]

    benchmark = Benchmarks.from_key(benchmark_key)
    output_root = BENCHMARK_SUBSETS_ROOT / benchmark.hf_name / run_id / "coverage"
    if name is None:
        coverage_str = (
            str(int(coverage))
            if abs(coverage - round(coverage)) < 1e-9
            else str(coverage)
        )
        output_file = output_root / f"{coverage_str}.json"
    else:
        output_file = output_root / f"{name}.json"
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


@cli.command(
    name="compare-rerun",
    help="Compare correctness transitions (T->T, T->F, F->T, F->F) between two runs",
)
@click.option("--orig-run-id", required=True, type=str, help="Original run id")
@click.option("--new-run-id", required=True, type=str, help="Re-run id")
@click.option(
    "--subset",
    type=str,
    default=None,
    help="Optional path to subset JSON with unique_ids; if omitted, uses intersection",
)
def compare_rerun(orig_run_id: str, new_run_id: str, subset: Optional[str]) -> None:
    orig_file = Path("./output") / orig_run_id / "inference_output.jsonl"
    new_file = Path("./output") / new_run_id / "inference_output.jsonl"
    orig_records = {rec["unique_id"]: rec for rec in load_jsonl(orig_file)}
    new_records = {rec["unique_id"]: rec for rec in load_jsonl(new_file)}

    if subset is not None:
        with open(subset, "r") as f:
            ids = set(json.load(f))
        candidate_ids = [
            uid for uid in ids if uid in orig_records and uid in new_records
        ]
    else:
        candidate_ids = list(set(orig_records.keys()) & set(new_records.keys()))

    if not candidate_ids:
        console.print(Text("No overlapping unique_ids to compare", style="red"))
        sys.exit(1)

    # Compute correctness per unique_id on both runs
    def is_correct(rec: Dict[str, Any]) -> bool:
        qa = _get_question_answer_from_record(rec)
        return bool(qa.is_correct)

    t_t: List[str] = []
    t_f: List[str] = []
    f_t: List[str] = []
    f_f: List[str] = []

    for uid in sorted(candidate_ids):
        a = is_correct(orig_records[uid])
        b = is_correct(new_records[uid])
        if a and b:
            t_t.append(uid)
        elif a and not b:
            t_f.append(uid)
        elif (not a) and b:
            f_t.append(uid)
        else:
            f_f.append(uid)

    total = len(candidate_ids)
    table = Table(title="Correctness transitions", box=box.SIMPLE_HEAVY)
    table.add_column("Transition", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("% of subset", justify="right")
    for name, ids in [
        ("T -> T", t_t),
        ("T -> F (bad)", t_f),
        ("F -> T (awesome)", f_t),
        ("F -> F", f_f),
    ]:
        pct = (100.0 * len(ids) / total) if total > 0 else 0.0
        table.add_row(name, str(len(ids)), f"{pct:.1f}")
    console.print(table)

    console.print(
        Text.assemble(
            "Subset size ",
            (str(total), "bold"),
            "; ",
            "Net gain ",
            (str(len(f_t) - len(t_f)), "bold"),
            " (F->T minus T->F)",
        )
    )

    # Accuracies: original full, rerun subset, and fused final
    def _acc(records_map: Dict[str, Dict[str, Any]], ids: List[str]) -> float:
        return (
            100.0 * sum(1 for uid in ids if is_correct(records_map[uid])) / len(ids)
            if ids
            else 0.0
        )

    orig_all_ids = list(orig_records.keys())
    acc_orig_full = _acc(orig_records, orig_all_ids)
    acc_new_subset = _acc(new_records, candidate_ids)

    fused_correct = 0
    for uid in orig_all_ids:
        rec = new_records.get(uid) or orig_records[uid]
        fused_correct += 1 if is_correct(rec) else 0
    acc_fused = 100.0 * fused_correct / len(orig_all_ids) if orig_all_ids else 0.0

    acc_table = Table(title="Accuracies", box=box.SIMPLE_HEAVY)
    acc_table.add_column("Metric", style="bold")
    acc_table.add_column("Value", justify="right")
    acc_table.add_row("Original (full dataset)", f"{acc_orig_full:.1f}%")
    acc_table.add_row("Re-run (subset only)", f"{acc_new_subset:.1f}%")
    acc_table.add_row("Final fused (new on subset, else original)", f"{acc_fused:.1f}%")
    console.print(acc_table)


def _safe_entropy(probs: List[float]) -> float:
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    max_h = math.log(len(probs)) if len(probs) > 1 else 1.0
    return float(h / max_h) if max_h > 0 else 0.0


def _compute_uncertainty_metrics(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Extract answers per completion and align with PRM aggregate scores
    completions: List[str] = sample.get("completions", [])
    answers: List[str] = [extract_answer(c or "", BENCHMARK) for c in completions]
    agg_scores: List[float] = sample.get("agg_scores") or []
    if not agg_scores and sample.get("scores"):
        try:
            agg_scores = [
                (float(s[-1]) if isinstance(s, list) and len(s) > 0 else 0.0)
                for s in sample["scores"]
            ]
        except Exception:
            agg_scores = [0.0 for _ in completions]

    n = len(completions)
    if n == 0:
        return {
            "n": 0,
            "agreement_ratio": 0.0,
            "unique_answers": 0,
            "entropy_freq": 0.0,
            "entropy_weighted": 0.0,
            "prm_max": 0.0,
            "prm_mean": 0.0,
            "prm_std": 0.0,
            "prm_margin": 0.0,
            "prm_top_frac": 0.0,
            "consensus_support": 0.0,
        }

    # Group by answer: counts and score sums
    count_by_ans: Dict[str, int] = defaultdict(int)
    score_by_ans: Dict[str, float] = defaultdict(float)
    for ans, s in zip(answers, agg_scores or [0.0] * n):
        count_by_ans[ans] += 1
        try:
            score_by_ans[ans] += float(s)
        except Exception:
            score_by_ans[ans] += 0.0

    counts = list(count_by_ans.values())
    scores_grouped = list(score_by_ans.values())
    sum_scores = float(sum(agg_scores)) if agg_scores else 0.0

    # Agreement and entropies
    agreement_ratio = (max(counts) / n) if counts else 0.0
    unique_answers = len(counts)
    freq_probs = [c / n for c in counts]
    entropy_freq = _safe_entropy(freq_probs)
    if sum_scores > 0 and len(scores_grouped) > 0:
        weighted_probs = [max(0.0, s) / sum_scores for s in scores_grouped]
        entropy_weighted = _safe_entropy(weighted_probs)
        consensus_support = max(weighted_probs)
    else:
        entropy_weighted = 0.0
        consensus_support = 0.0

    # PRM statistics at completion level
    try:
        prm_max = max(float(x) for x in (agg_scores or [0.0]))
        prm_mean = (sum(float(x) for x in (agg_scores or [0.0])) / n) if n > 0 else 0.0
        prm_std = (
            math.sqrt(
                sum((float(x) - prm_mean) ** 2 for x in (agg_scores or [0.0])) / n
            )
            if n > 0
            else 0.0
        )
        sorted_scores = sorted([float(x) for x in (agg_scores or [0.0])], reverse=True)
        prm_margin = (
            (sorted_scores[0] - sorted_scores[1])
            if len(sorted_scores) >= 2
            else sorted_scores[0]
        )
        prm_top_frac = (sorted_scores[0] / sum_scores) if sum_scores > 0 else 0.0
    except Exception:
        prm_max = prm_mean = prm_std = prm_margin = prm_top_frac = 0.0

    return {
        "n": n,
        "agreement_ratio": float(agreement_ratio),
        "unique_answers": int(unique_answers),
        "entropy_freq": float(entropy_freq),
        "entropy_weighted": float(entropy_weighted),
        "prm_max": float(prm_max),
        "prm_mean": float(prm_mean),
        "prm_std": float(prm_std),
        "prm_margin": float(prm_margin),
        "prm_top_frac": float(prm_top_frac),
        "consensus_support": float(consensus_support),
    }


def _summarise_uncertainty(metrics: List[Dict[str, Any]], labels: List[bool]) -> None:
    # Build summary table: means by class and coverage/recall tradeoffs
    fields = [
        "agreement_ratio",
        "entropy_freq",
        "entropy_weighted",
        "prm_margin",
        "prm_top_frac",
        "consensus_support",
        "prm_std",
        "prm_mean",
    ]

    # Means by class
    means_table = Table(title="Means by correctness", box=box.SIMPLE_HEAVY)
    means_table.add_column("Metric", style="bold")
    means_table.add_column("mean(correct)", justify="right")
    means_table.add_column("mean(incorrect)", justify="right")
    means_table.add_column("direction", justify="center")
    for f in fields:
        vals = [m.get(f, 0.0) for m in metrics]
        corr_vals = [v for v, y in zip(vals, labels) if y]
        inc_vals = [v for v, y in zip(vals, labels) if not y]
        mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
        mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
        direction = "↑" if mean_corr > mean_inc else "↓"
        means_table.add_row(
            f,
            f"{mean_corr:.3f}",
            f"{mean_inc:.3f}",
            direction,
        )
    console.print(means_table)

    # Coverage vs recall-of-incorrect for each metric
    cov_table = Table(title="Coverage vs recall of incorrect", box=box.SIMPLE_HEAVY)
    cov_table.add_column("Metric", style="bold")
    coverage_pcts = [10, 20, 30, 40, 50]
    for p in coverage_pcts:
        cov_table.add_column(f"{p}% cov", justify="right")

    total_incorrect = sum(1 for y in labels if not y)
    total = len(labels)
    if total == 0:
        console.print(Text("No samples to summarise", style="red"))
        return

    for f in fields:
        vals = [m.get(f, 0.0) for m in metrics]
        corr_vals = [v for v, y in zip(vals, labels) if y]
        inc_vals = [v for v, y in zip(vals, labels) if not y]
        mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
        mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
        low_is_uncertain = (
            mean_corr > mean_inc
        )  # if higher on correct, lower suggests uncertainty

        rows: List[str] = []
        idxs = list(range(total))
        idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)
        for p in coverage_pcts:
            k = max(1, int(round(total * (p / 100.0))))
            flagged = set(idxs[:k])
            incorrect_flagged = sum(1 for i in flagged if not labels[i])
            recall_incorrect = (
                100.0 * incorrect_flagged / total_incorrect
                if total_incorrect > 0
                else 0.0
            )
            rows.append(f"{recall_incorrect:.1f}")
        cov_table.add_row(f, *rows)
    console.print(cov_table)

    # F:T ratio table with coverage percentages as columns
    ft_table = Table(title="F:T ratio within coverage", box=box.SIMPLE_HEAVY)
    ft_table.add_column("Metric", style="bold")
    for p in coverage_pcts:
        ft_table.add_column(f"{p}% cov", justify="center")

    for f in fields:
        vals = [m.get(f, 0.0) for m in metrics]
        corr_vals = [v for v, y in zip(vals, labels) if y]
        inc_vals = [v for v, y in zip(vals, labels) if not y]
        mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
        mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
        low_is_uncertain = mean_corr > mean_inc

        idxs = list(range(total))
        idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)

        row_data = []
        for p in coverage_pcts:
            k = max(1, int(round(total * (p / 100.0))))
            flagged = idxs[:k]

            f_cnt = sum(1 for i in flagged if not labels[i])
            t_cnt = sum(1 for i in flagged if labels[i])
            denom = f_cnt + t_cnt
            pct_f = 100.0 * f_cnt / denom if denom > 0 else 0.0

            row_data.append(f"{f_cnt}:{t_cnt} ({pct_f:.1f}%)")

        ft_table.add_row(f, *row_data)

    console.print(ft_table)


def _get_confidence(sample: Dict[str, Any], metric: str) -> float:
    """Return a scalar confidence for a sample using existing metrics.

    Falls back to 0.0 if unavailable.
    """
    try:
        m = _compute_uncertainty_metrics(sample)
        v = m.get(metric, 0.0)
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


def _is_correct_record(rec: Dict[str, Any]) -> bool:
    qa = _get_question_answer_from_record(rec)
    return bool(qa.is_correct)


def _load_subset_ids(path: str) -> List[str]:
    """Load a subset JSON that may be a list of ids or a dict with unique_ids."""
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        ids = obj.get("unique_ids")
        if isinstance(ids, list):
            return [str(x) for x in ids]
    return []


def _order_indices_by_metric(
    metrics: List[Dict[str, Any]], labels: List[bool], field: str
) -> List[int]:
    vals = [m.get(field, 0.0) for m in metrics]
    low_is_uncertain = _metric_direction_low_is_uncertain(vals, labels)
    idxs = list(range(len(metrics)))
    idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)
    return idxs


def _order_indices_by_hybrid_rank(
    metrics: List[Dict[str, Any]], labels: List[bool], fields: List[str]
) -> List[int]:
    # Build oriented ranks per field (uncertain-first)
    per_field_rank: List[Dict[int, int]] = []
    n = len(metrics)
    for f in fields:
        order = _order_indices_by_metric(metrics, labels, f)
        rank_map = {idx: r for r, idx in enumerate(order)}
        per_field_rank.append(rank_map)

    # Average rank across fields
    avg_rank = [0.0] * n
    for i in range(n):
        avg_rank[i] = sum(r[i] for r in per_field_rank) / max(1, len(per_field_rank))

    idxs = list(range(n))
    idxs.sort(key=lambda i: avg_rank[i])
    return idxs


def _recall_of_incorrect(
    order: List[int], labels: List[bool], coverage_pct: int
) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    total_incorrect = sum(1 for y in labels if not y)
    if total_incorrect == 0:
        return 0.0
    k = max(1, int(round(total * (coverage_pct / 100.0))))
    flagged = set(order[:k])
    incorrect_flagged = sum(1 for i in flagged if not labels[i])
    return 100.0 * incorrect_flagged / total_incorrect


@cli.command(
    name="compare-selection",
    help=(
        "Compare agreement_ratio against PRM-based metrics and hybrids across coverage. "
        "Reports recall of incorrect within top-K most uncertain."
    ),
)
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
@click.option(
    "--metrics",
    default="agreement_ratio,consensus_support,prm_top_frac,prm_margin,prm_mean,hybrid(agreement_ratio+consensus_support),hybrid(agreement_ratio+prm_margin)",
    type=str,
    help=(
        "Comma-separated list of metrics. Use 'hybrid(a+b+...)' to average uncertainty ranks."
    ),
)
@click.option(
    "--coverages",
    default="10,20,30,40,50",
    type=str,
    help="Comma-separated coverage percentages to evaluate",
)
def compare_selection(run_id: str, metrics: str, coverages: str) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        sys.exit(1)

    metrics_list: List[Dict[str, Any]] = []
    labels: List[bool] = []
    for rec in records:
        qa = _get_question_answer_from_record(rec)
        labels.append(bool(qa.is_correct))
        metrics_list.append(_compute_uncertainty_metrics(rec))

    metric_specs = [m.strip() for m in metrics.split(",") if m.strip()]
    coverage_pcts = [int(x) for x in coverages.split(",") if x.strip()]

    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("File", f"output/{run_id}/inference_output.jsonl")
    header.add_row("Samples", str(len(labels)))
    console.print(Panel(header, title="Selection metric comparison", box=box.ROUNDED))

    table = Table(title="Recall of incorrect within coverage", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="bold")
    for p in coverage_pcts:
        table.add_column(f"{p}%", justify="right")

    for spec in metric_specs:
        if spec.startswith("hybrid(") and spec.endswith(")"):
            inner = spec[len("hybrid(") : -1]
            fields = [s.strip() for s in inner.split("+") if s.strip()]
            order = _order_indices_by_hybrid_rank(metrics_list, labels, fields)
            name = f"hybrid({'+'.join(fields)})"
        else:
            order = _order_indices_by_metric(metrics_list, labels, spec)
            name = spec

        row_vals = [
            f"{_recall_of_incorrect(order, labels, p):.1f}" for p in coverage_pcts
        ]
        table.add_row(name, *row_vals)

    console.print(table)


@cli.command(
    name="fuse-confidence",
    help=(
        "Offline confidence-based fusion between two runs on the intersection (or subset). "
        "For each sample, pick the run with higher confidence (metric) and report fused accuracy."
    ),
)
@click.option("--run-a-id", required=True, type=str, help="Baseline/original run id")
@click.option("--run-b-id", required=True, type=str, help="Alternative run id")
@click.option(
    "--metric",
    default="consensus_support",
    type=click.Choice(
        [
            "agreement_ratio",
            "entropy_freq",
            "entropy_weighted",
            "prm_margin",
            "prm_top_frac",
            "consensus_support",
            "prm_std",
            "prm_mean",
        ]
    ),
    help="Confidence metric used to select per-sample",
)
@click.option(
    "--delta",
    default=0.0,
    type=float,
    help=(
        "Optional minimum margin required for switching from run A to run B (B_conf > A_conf + delta)."
    ),
)
@click.option(
    "--min-b-conf",
    default=None,
    type=float,
    help=(
        "Optional absolute confidence threshold for run B; switch only if B_conf >= min_b_conf."
    ),
)
@click.option(
    "--max-a-conf",
    default=None,
    type=float,
    help=(
        "Optional absolute confidence cap for run A; switch only if A_conf <= max_a_conf."
    ),
)
@click.option(
    "--subset",
    type=str,
    default=None,
    help=(
        "Optional path to subset JSON; may be a list of unique_ids or an object with unique_ids."
    ),
)
def fuse_confidence(
    run_a_id: str,
    run_b_id: str,
    metric: str,
    delta: float,
    min_b_conf: Optional[float],
    max_a_conf: Optional[float],
    subset: Optional[str],
) -> None:
    file_a = Path("./output") / run_a_id / "inference_output.jsonl"
    file_b = Path("./output") / run_b_id / "inference_output.jsonl"
    recs_a = {rec["unique_id"]: rec for rec in load_jsonl(file_a)}
    recs_b = {rec["unique_id"]: rec for rec in load_jsonl(file_b)}

    if subset is not None:
        ids = set(_load_subset_ids(subset))
        candidate_ids = [uid for uid in ids if uid in recs_a and uid in recs_b]
    else:
        candidate_ids = list(set(recs_a.keys()) & set(recs_b.keys()))

    if not candidate_ids:
        console.print(Text("No overlapping unique_ids to fuse", style="red"))
        sys.exit(1)

    # Acc helper
    def _acc(records_map: Dict[str, Dict[str, Any]], ids: List[str]) -> float:
        return (
            100.0
            * sum(1 for uid in ids if _is_correct_record(records_map[uid]))
            / len(ids)
            if ids
            else 0.0
        )

    # Baselines on intersection
    acc_a = _acc(recs_a, candidate_ids)
    acc_b = _acc(recs_b, candidate_ids)

    # Build fused selection
    fused_correct = 0
    chosen_b = 0
    flips_pos = 0  # F->T relative to A
    flips_neg = 0  # T->F relative to A

    for uid in candidate_ids:
        ra = recs_a[uid]
        rb = recs_b[uid]
        ca = _get_confidence(ra, metric)
        cb = _get_confidence(rb, metric)

        choose_b = cb > (ca + delta)
        if choose_b and (min_b_conf is not None) and not (cb >= float(min_b_conf)):
            choose_b = False
        if choose_b and (max_a_conf is not None) and not (ca <= float(max_a_conf)):
            choose_b = False

        rec = rb if choose_b else ra
        if choose_b:
            chosen_b += 1

        a_ok = _is_correct_record(ra)
        fused_ok = _is_correct_record(rec)
        fused_correct += 1 if fused_ok else 0

        if (not a_ok) and fused_ok:
            flips_pos += 1
        elif a_ok and (not fused_ok):
            flips_neg += 1

    acc_fused = 100.0 * fused_correct / len(candidate_ids)

    # Report
    table = Table(title="Confidence-based fusion (intersection)", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Run A", run_a_id)
    table.add_row("Run B", run_b_id)
    table.add_row("Confidence", metric)
    table.add_row("Delta", f"{delta:.4f}")
    if min_b_conf is not None:
        table.add_row("min B conf", f"{float(min_b_conf):.4f}")
    if max_a_conf is not None:
        table.add_row("max A conf", f"{float(max_a_conf):.4f}")
    table.add_row("Samples", str(len(candidate_ids)))
    table.add_row("Chosen from B", str(chosen_b))
    table.add_row("Accuracy A", f"{acc_a:.1f}%")
    table.add_row("Accuracy B", f"{acc_b:.1f}%")
    table.add_row("Accuracy fused", f"{acc_fused:.1f}%")
    table.add_row("F->T vs A", str(flips_pos))
    table.add_row("T->F vs A", str(flips_neg))
    table.add_row("Net gain (F->T - T->F)", str(flips_pos - flips_neg))
    console.print(table)


@cli.command(name="uncertainty", help="Evaluate uncertainty heuristics over a run")
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
def cmd_uncertainty(run_id: str) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        sys.exit(1)

    metrics_list: List[Dict[str, Any]] = []
    labels: List[bool] = []

    for rec in records:
        qa = _get_question_answer_from_record(rec)
        labels.append(bool(qa.is_correct))
        metrics_list.append(_compute_uncertainty_metrics(rec))

    total = len(labels)
    acc = (100.0 * sum(1 for y in labels if y) / total) if total > 0 else 0.0
    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("File", f"output/{run_id}/inference_output.jsonl")
    header.add_row("Samples", str(total))
    header.add_row("Accuracy (assumed pred)", f"{acc:.1f}%")
    console.print(Panel(header, title="Uncertainty analysis", box=box.ROUNDED))

    _summarise_uncertainty(metrics_list, labels)


@cli.command(
    name="export-compact", help="Export all question details in compact JSON format"
)
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
@click.option(
    "--output-file",
    type=str,
    default=None,
    help="Output JSON file path (default: output/{run_id}/compact_export.json)",
)
def cmd_export_compact(run_id: str, output_file: Optional[str]) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        sys.exit(1)

    # Determine output file path
    output_path = Path("./data/question_insight") / run_id / "compact_export.json"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create compact schema-separated format
    export_data = {
        "meta": {
            "run_id": run_id,
            "source": str(out_file),
            "count": len(records),
            "timestamp": datetime.now().isoformat(),
        },
        "schema": {
            "question_fields": [
                "idx",
                "uid",
                "subject",
                "level",
                "problem",
                "gt_answer",
                "gt_extracted",
                "pred_extracted",
                "is_correct",
                "num_completions",
                "chosen_idx",
            ],
            "metrics_fields": [
                "agreement_ratio",
                "entropy_freq",
                "consensus_support",
                "unique_answers",
                "prm_mean",
                "prm_std",
                "prm_margin",
            ],
            "completion_fields": [
                "question_idx",
                "completion_idx",
                "extracted_answer",
                "prm_score",
                "is_correct",
            ],
        },
        "questions": [],
        "completions": [],  # Separate array for all completions across questions
    }

    completion_global_idx = 0

    for i, sample in enumerate(records):
        # Basic question info
        unique_id = sample["unique_id"]
        problem = sample["problem"]
        answer = sample["answer"]
        subject = sample["subject"]
        level = sample["level"]

        # Predictions and correctness
        pred_text = sample["pred"]
        pred = sample.get(ASSUMED_PRED_KEY, "")
        answer_extracted = extract_answer(_wrap_in_boxed(answer), BENCHMARK)

        pred_extracted = extract_answer(pred, BENCHMARK)
        is_correct = math_equal(answer_extracted, pred_extracted)

        # Completions info
        completions = sample.get("completions", [])
        agg_scores = sample.get("agg_scores", [])

        # If no agg_scores, fall back to last scores from trajectories
        if not agg_scores and sample.get("scores"):
            try:
                agg_scores = [
                    (float(s[-1]) if isinstance(s, list) and len(s) > 0 else 0.0)
                    for s in sample["scores"]
                ]
            except Exception:
                agg_scores = [0.0 for _ in completions]

        # Find chosen completion index
        chosen_idx = _index_of_first(completions, pred_text) if completions else -1

        # Uncertainty metrics
        uncertainty_metrics = _compute_uncertainty_metrics(sample)

        # Compact question data (array format following schema)
        question_row = [
            i,  # idx
            unique_id,  # uid
            subject,  # subject
            level,  # level
            problem,  # problem
            answer,  # gt_answer
            answer_extracted,  # gt_extracted
            pred_extracted,  # pred_extracted
            is_correct,  # is_correct
            len(completions),  # num_completions
            chosen_idx,  # chosen_idx
        ]

        # Compact metrics data (array format following schema)
        metrics_row = [
            round(uncertainty_metrics["agreement_ratio"], 4),
            round(uncertainty_metrics["entropy_freq"], 4),
            round(uncertainty_metrics["consensus_support"], 4),
            uncertainty_metrics["unique_answers"],
            round(uncertainty_metrics["prm_mean"], 4),
            round(uncertainty_metrics["prm_std"], 4),
            round(uncertainty_metrics["prm_margin"], 4),
        ]

        export_data["questions"].append([question_row, metrics_row])

        # Process completions separately
        if completions:
            extracted_answers = [
                extract_answer(c or "", BENCHMARK) for c in completions
            ]

            for j, (ans, score) in enumerate(zip(extracted_answers, agg_scores or [])):
                completion_row = [
                    i,  # question_idx (reference back to question)
                    j,  # completion_idx within question
                    ans,  # extracted_answer
                    round(float(score) if score is not None else 0.0, 4),  # prm_score
                    math_equal(answer_extracted, ans),  # is_correct
                ]
                export_data["completions"].append(completion_row)
                completion_global_idx += 1

    # Write to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    # Summary
    total_correct = sum(
        1 for q in export_data["questions"] if q["prediction"]["is_correct"]
    )
    accuracy = 100.0 * total_correct / len(records) if records else 0.0

    summary_table = Table(title="Export summary", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Run ID", run_id)
    summary_table.add_row("Total questions", str(len(records)))
    summary_table.add_row("Correct answers", str(total_correct))
    summary_table.add_row("Accuracy", f"{accuracy:.1f}%")
    summary_table.add_row("Output file", str(output_path))

    summary_table.add_row("File size", f"{output_path.stat().st_size / 1024:.1f} KB")

    console.print(Panel(summary_table, box=box.ROUNDED))
    console.print(Text(f"✓ Exported compact data to {output_path}", style="green"))


if __name__ == "__main__":
    cli()
