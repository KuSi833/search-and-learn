#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
from datasets import Dataset
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sal.evaluation.evaluate import evaluate_single_dataset
from sal.evaluation.grader import math_equal
from sal.evaluation.parser import extract_answer, find_box
from sal.utils.logging import setup_logging

setup_logging()


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


def analyse_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Extract key fields (throw if not found)
    problem: str = sample["problem"]
    solution: str = sample["solution"]
    answer: str = sample["answer"]
    unique_id: str = sample["unique_id"]
    subject: str = sample["subject"]
    level: int = sample["level"]

    completions: List[str] = sample["completions"]
    pred: str = sample["pred"]
    # scores: List[List[float]] = sample.get("scores", [])
    completion_tokens: Any = sample["completion_tokens"]

    # Assuming I am actually wondering about ASSUMED_PRED_KEY accuracy
    pred = sample[ASSUMED_PRED_KEY]

    answer_extracted = extract_answer(answer, BENCHMARK)
    pred_extracted = extract_answer(pred, BENCHMARK)

    assumed_correct = math_equal(answer_extracted, pred_extracted)

    return {
        "unique_id": unique_id,
        "subject": subject,
        "level": level,
        "problem": problem,
        "solution": solution,
        "answer": answer,
        "num_beams": len(completions),
        "pred": pred,
        "pred_extracted": pred_extracted,
        "assumed_correct": assumed_correct,
        "completion_tokens": completion_tokens,
    }


def print_report(
    run_id: str,
    analysis: Dict[str, Any],
    index: Optional[int] = None,
) -> None:
    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("File", f"output/{run_id}/inference_output.jsonl")
    if index is not None:
        header.add_row("Index", str(index))
    console.print(Panel(header, title="Inference Sample", box=box.ROUNDED))

    uid = analysis.get("unique_id")
    subj = analysis.get("subject")
    level = analysis.get("level")
    meta_bits = [
        f"id={uid}" if uid is not None else None,
        f"subject={subj}" if subj is not None else None,
        f"level={level}" if level is not None else None,
    ]
    meta_bits = [b for b in meta_bits if b]
    if meta_bits:
        console.print(Text("Meta: " + ", ".join(meta_bits), style="dim"))

    problem_text = _shorten(str(analysis.get("problem") or ""), 800)
    console.print(Panel(problem_text, title="Problem", box=box.SQUARE))

    ans = str(analysis.get("answer") or "")
    console.print(Panel(_shorten(ans, 200), title="Ground truth answer", style="green"))

    pred_raw = _shorten(str(analysis.get("pred") or ""), 800)
    console.print(Panel(pred_raw, title="Chosen prediction (raw)", style="yellow"))

    extracted = _shorten(str(analysis.get("pred_extracted") or ""), 200)
    raw_ok = bool(analysis.get("raw_correct"))

    assumed_key = str(analysis.get("assumed_pred_key") or ASSUMED_PRED_KEY)
    assumed_text = str(analysis.get("assumed_pred_text") or "")
    assumed_extracted = _shorten(str(analysis.get("assumed_pred_extracted") or ""), 200)
    assumed_ok = bool(analysis.get("assumed_correct"))

    extracted_table = Table(
        title="Answer extraction & correctness", box=box.SIMPLE_HEAVY
    )
    extracted_table.add_column("Field", style="bold")
    extracted_table.add_column("Value")
    extracted_table.add_row("Raw extracted", extracted)
    extracted_table.add_row(
        "Raw correct", Text(str(raw_ok), style=("green" if raw_ok else "red"))
    )
    if assumed_text:
        extracted_table.add_row("Assumed key", assumed_key)
        extracted_table.add_row("Assumed extracted", assumed_extracted)
        extracted_table.add_row(
            "Assumed correct",
            Text(str(assumed_ok), style=("green" if assumed_ok else "red")),
        )
    console.print(Panel(extracted_table, box=box.ROUNDED))

    chosen_idx = analysis.get("chosen_idx")
    num_beams = analysis.get("num_beams")
    csp = analysis.get("chosen_score_path") or []
    ctok = analysis.get("chosen_tokens")

    beams_table = Table(show_header=False, box=box.SIMPLE)
    beams_table.add_column("key", style="bold")
    beams_table.add_column("value")
    beams_table.add_row("Beams", str(num_beams))
    beams_table.add_row("Chosen index", str(chosen_idx))
    beams_table.add_row(
        "Score trajectory", Text(_format_float_list(list(csp)), style="magenta")
    )
    if ctok is not None:
        beams_table.add_row("Completion tokens", str(ctok))
    console.print(Panel(beams_table, title="Beam details", box=box.ROUNDED))

    # Visualise solution steps with scores
    full_pred_text = str(analysis.get("pred") or "")
    steps = _split_steps(full_pred_text)
    step_scores: List[float] = []
    try:
        step_scores = [float(x) for x in (analysis.get("chosen_score_path") or [])]
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
        console.print(
            Panel(steps_table, title="Solution steps", subtitle=note, box=box.ROUNDED)
        )

    rank = analysis.get("rank_by_last_score") or []
    if rank:
        top = rank[: min(5, len(rank))]
        rank_table = Table(title="Top beams by last score", box=box.MINIMAL_HEAVY_HEAD)
        rank_table.add_column("#", style="bold cyan", justify="right")
        rank_table.add_column("score", justify="right")
        for i, s in top:
            rank_table.add_row(str(i), f"{s:.4f}")
        console.print(rank_table)

    agg_preds = analysis.get("agg_preds") or []
    agg_correct = {k: v for k, v in (analysis.get("agg_correct") or [])}
    if agg_preds:
        agg_table = Table(title="Aggregated predictions", box=box.SIMPLE_HEAVY)
        agg_table.add_column("key", style="bold")
        agg_table.add_column("inner")
        agg_table.add_column("correct", justify="center")
        for k, v in agg_preds:
            inner = find_box(v) if isinstance(v, str) else ""
            ok = agg_correct.get(k, False)
            agg_table.add_row(
                k, inner, Text("✓" if ok else "✗", style=("green" if ok else "red"))
            )
        console.print(agg_table)

    sol = analysis.get("solution") or ""
    if isinstance(sol, str) and sol.strip():
        console.print(
            Panel(
                sol,
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
    analysis = analyse_sample(sample)
    print_report(run_id, analysis, index)


def _record_level_and_correct(rec: Dict[str, Any]) -> Tuple[str, bool]:
    level: str = rec["level"]
    answer: str = rec["answer"]
    pred: str = rec[ASSUMED_PRED_KEY]

    answer_extracted = extract_answer(answer, BENCHMARK)
    pred_extracted = extract_answer(pred, BENCHMARK)

    is_correct = math_equal(answer_extracted, pred_extracted)

    return level, is_correct


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
    level_to_correct_indices: Dict[str, List[int]] = defaultdict(list)
    level_to_incorrect_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, rec in enumerate(records):
        level, correct = _record_level_and_correct(rec)
        level_to_total[level] += 1
        if correct:
            level_to_correct[level] += 1
            level_to_correct_indices[level].append(idx)
        else:
            level_to_incorrect_indices[level].append(idx)

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

    for level in sorted(level_to_total.keys()):
        total = level_to_total[level]
        correct = level_to_correct[level]
        incorrect = total - correct
        title = f"Level: {level}  |  Total: {total}  Correct: {correct}  Incorrect: {incorrect}"
        correct_idxs = (
            ", ".join(map(str, level_to_correct_indices[level]))
            if level_to_correct_indices[level]
            else ""
        )
        incorrect_idxs = (
            ", ".join(map(str, level_to_incorrect_indices[level]))
            if level_to_incorrect_indices[level]
            else ""
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
def question_answer(run_id: str) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"

    jsonl_data = list(load_jsonl(out_file))
    dataset = Dataset.from_list(jsonl_data)

    print(dataset)

    # dataset = dataset.map(
    #     parse_gt,
    #     desc="Parsing ground truth",
    #     num_proc=4,
    #     load_from_cache_file=False,
    # )

    # experiment_config = ExperimentConfig()
    # dataset = score(dataset, experiment_config, 4)

    evaluate_single_dataset(
        benchmark="math",
        dataset=dataset,
        dataset_col="pred",
        output_file=Path("./out.res"),
    )

    # for row in dataset:
    #     answer = row.get("answer")
    #     pred = row.get("pred_weighted@4")

    #     # answer_canonical = "\\boxed{" + memoized_canonical_form(answer) + "}"
    #     # pred_canonical = memoized_canonical_form(pred)
    #     # print(answer_canonical)
    #     # print(pred_canonical)

    #     ok = answer_canonical == pred_canonical
    #     if not ok:
    #         # print(f"{answer} : {pred}")
    #         print(f"{answer_canonical} : {pred_canonical}")
    # for idx, rec in enumerate(dataset):
    #     unique_id = rec.get("unique_id")
    #     answer = rec.get("answer")
    #     assumed_text = rec.get("pred_weighted@4")
    #     if isinstance(assumed_text, str):
    #         inner = find_box(assumed_text)
    #         answer_map = extract_answer_map()
    #         # print(inner)
    #         print(f"{unique_id:40} {assumed_text} -> {inner} : {answer}")
    # # extracted = extract_answer(assumed_text, "math")
    # exit()
    #     extracted = inner if inner else


if __name__ == "__main__":
    cli()
