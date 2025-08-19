#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sal.evaluation.grader import math_equal
from sal.evaluation.parser import extract_answer, find_box
from sal.utils.constants import BENCHMARK_SUBSETS_ROOT, DATASETS, Benchmark
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
    pred_text: str = sample["pred"]  # prediction text
    pred = sample[ASSUMED_PRED_KEY]  # just the prediction
    # completions: List[str] = sample["completions"]
    # scores: List[List[float]] = sample.get("scores", [])
    # completion_tokens: Any = sample["completion_tokens"]

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

    print(sample[ASSUMED_PRED_KEY])

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

        note: Optional[str] = None
        if step_scores and len(steps) != len(step_scores):
            note = f"Note: steps ($${len(steps)}$$) and scores ($${len(step_scores)}$$) lengths differ; showing first $${n_rows}$$."
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

    benchmark_mapping = BenchmarkMapping("math500")

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
    default=Benchmark.MATH500.value,
    type=click.Choice([b.value for b in Benchmark]),
    help="Benchmark to use for answer extraction",
)
@click.option("--name", required=True, type=str, help="Name for the subset file")
def extract_incorrect(run_id: str, benchmark: str, name: str) -> None:
    benchmark_enum = Benchmark(benchmark)
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

    # Save to JSON file
    output_dir = BENCHMARK_SUBSETS_ROOT / DATASETS[benchmark_enum.value]["hf_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{name}.json"

    with open(output_file, "w") as f:
        json.dump(incorrect_unique_ids, f, indent=2)

    console.print(
        Text.assemble(
            f"Saved {len(incorrect_unique_ids)} incorrect unique_ids to ",
            (str(output_file), "yellow"),
        )
    )


if __name__ == "__main__":
    cli()
