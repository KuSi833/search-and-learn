#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click

from sal.utils.qwen_math_parser import (
    extract_answer,
    find_box,
    math_equal,
)


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


def analyse_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Extract key fields with fallbacks
    problem = sample.get("problem") or sample.get("question") or ""
    solution = sample.get("solution", "")
    answer = sample.get("answer", "")
    unique_id = sample.get("unique_id") or sample.get("id")
    subject = sample.get("subject")
    level = sample.get("level")

    completions: List[str] = sample.get("completions", [])
    pred: str = sample.get("pred", "")
    scores: List[List[float]] = sample.get("scores", [])
    completion_tokens = sample.get("completion_tokens")

    # Locate chosen completion among beams (fallback by highest last-score if not found)
    chosen_idx = _index_of_first(completions, pred)
    if chosen_idx < 0 and scores:
        # Fallback: choose beam with largest last score
        try:
            last_scores = [
                s[-1] if isinstance(s, list) and len(s) > 0 else float("-inf")
                for s in scores
            ]
            chosen_idx = (
                int(max(range(len(last_scores)), key=lambda i: last_scores[i]))
                if last_scores
                else -1
            )
        except Exception:
            chosen_idx = -1

    chosen_score_path: List[float] = (
        scores[chosen_idx] if 0 <= chosen_idx < len(scores) else []
    )
    chosen_tokens: Any = None
    if isinstance(completion_tokens, list) and 0 <= chosen_idx < len(completion_tokens):
        chosen_tokens = completion_tokens[chosen_idx]
    elif isinstance(completion_tokens, int):
        chosen_tokens = completion_tokens

    # Extract boxed answers for ensemble-style preds if present
    agg_preds: List[Tuple[str, str]] = []  # (key, value)
    for k, v in sample.items():
        if isinstance(k, str) and isinstance(v, str):
            if (
                k.startswith("pred_weighted@")
                or k.startswith("pred_maj@")
                or k.startswith("pred_naive@")
            ):
                agg_preds.append((k, v))
    agg_preds.sort(key=lambda kv: kv[0])

    # Compute correctness signals
    # For the raw pred: attempt to extract an answer from the text
    extracted_pred = extract_answer(pred, "math") if isinstance(pred, str) else ""
    # Boxed ensemble preds → extract inside box
    agg_correct: List[Tuple[str, bool]] = []
    for k, v in agg_preds:
        boxed_inner = find_box(v) if isinstance(v, str) else ""
        try:
            ok = math_equal(boxed_inner, answer)
        except Exception:
            ok = False
        agg_correct.append((k, ok))

    # Correctness for the extracted raw pred
    try:
        raw_correct = math_equal(extracted_pred, answer)
    except Exception:
        raw_correct = False

    # Build ranking by final score if available
    rank: List[Tuple[int, float]] = []
    if scores and all(isinstance(s, list) for s in scores):
        for i, s in enumerate(scores):
            if s:
                rank.append((i, float(s[-1])))
        rank.sort(key=lambda t: t[1], reverse=True)

    return {
        "unique_id": unique_id,
        "subject": subject,
        "level": level,
        "problem": problem,
        "solution": solution,
        "answer": answer,
        "num_beams": len(completions),
        "chosen_idx": chosen_idx,
        "pred": pred,
        "pred_extracted": extracted_pred,
        "raw_correct": raw_correct,
        "chosen_score_path": chosen_score_path,
        "chosen_tokens": chosen_tokens,
        "rank_by_last_score": rank,
        "agg_preds": agg_preds,
        "agg_correct": agg_correct,
        "completion_tokens": completion_tokens,
    }


def print_report(
    run_id: str,
    sample: Dict[str, Any],
    analysis: Dict[str, Any],
    index: Optional[int] = None,
) -> None:
    print(f"Run: {run_id}")
    print(f"File: output/{run_id}/inference_output.jsonl")
    if index is not None:
        print(f"Index: {index}")

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
        print("Meta: " + ", ".join(meta_bits))

    print("-")
    print("Problem:")
    print(_shorten(str(analysis.get("problem") or ""), 800))

    ans = str(analysis.get("answer") or "")
    print("-")
    print("Ground truth answer:")
    print(_shorten(ans, 200))

    print("-")
    print("Chosen prediction (raw):")
    print(_shorten(str(analysis.get("pred") or ""), 800))
    print(
        f"Extracted answer: {_shorten(str(analysis.get('pred_extracted') or ''), 200)}"
    )
    print(f"Correct (raw extracted vs answer): {bool(analysis.get('raw_correct'))}")

    chosen_idx = analysis.get("chosen_idx")
    num_beams = analysis.get("num_beams")
    print("-")
    print(f"Beams: {num_beams}, chosen_index: {chosen_idx}")
    csp = analysis.get("chosen_score_path") or []
    print(f"Score trajectory (chosen): {_format_float_list(list(csp))}")
    ctok = analysis.get("chosen_tokens")
    if ctok is not None:
        print(f"Completion tokens (chosen): {ctok}")

    rank = analysis.get("rank_by_last_score") or []
    if rank:
        top = rank[: min(5, len(rank))]
        pretty = ", ".join([f"#{i}: {s:.4f}" for i, s in top])
        print(f"Top beams by last score: {pretty}")

    agg_preds = analysis.get("agg_preds") or []
    agg_correct = {k: v for k, v in (analysis.get("agg_correct") or [])}
    if agg_preds:
        print("-")
        print("Aggregated predictions (correctness):")
        for k, v in agg_preds:
            inner = find_box(v) if isinstance(v, str) else ""
            ok = agg_correct.get(k, False)
            print(f"  {k}: {inner}  -> correct={ok}")

    # Optionally show beginning of the solution for context
    sol = analysis.get("solution") or ""
    if isinstance(sol, str) and sol.strip():
        print("-")
        print("Reference solution (truncated):")
        print(_shorten(sol, 600))


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
        print(f"No records found in {out_file}")
        sys.exit(1)
    if index < 0 or index >= len(records):
        print(f"Index out of range: {index} (num records = {len(records)})")
        sys.exit(1)

    sample = records[index]
    analysis = analyse_sample(sample)
    print_report(run_id, sample, analysis, index=index)


def _record_level_and_correct(rec: Dict[str, Any]) -> Tuple[str, bool]:
    level = str(rec.get("level", "unknown"))
    answer = str(rec.get("answer", ""))
    pred = str(rec.get("pred", ""))
    extracted_pred = extract_answer(pred, "math") if isinstance(pred, str) else ""
    try:
        is_correct = math_equal(extracted_pred, answer)
    except Exception:
        is_correct = False
    return level, bool(is_correct)


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
        print(f"No records found in {out_file}")
        sys.exit(1)

    from collections import defaultdict

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

    print(f"Run: {run_id}")
    print(f"File: output/{run_id}/inference_output.jsonl")
    print("-")
    for level in sorted(level_to_total.keys()):
        total = level_to_total[level]
        correct = level_to_correct[level]
        incorrect = total - correct
        print(f"Level: {level}")
        print(f"  Total: {total} | Correct: {correct} | Incorrect: {incorrect}")
        if level_to_correct_indices[level]:
            idxs = ", ".join(map(str, level_to_correct_indices[level]))
            print(f"  Correct indices: [{idxs}]")
        else:
            print("  Correct indices: []")
        if level_to_incorrect_indices[level]:
            idxs = ", ".join(map(str, level_to_incorrect_indices[level]))
            print(f"  Incorrect indices: [{idxs}]")
        else:
            print("  Incorrect indices: []")


if __name__ == "__main__":
    cli()
