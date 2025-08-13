#!/usr/bin/env python3
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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
                # Skip malformed lines to be robust
                continue
    return records


def _split_steps(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def _choose_beam_index(sample: Dict[str, Any]) -> int:
    completions = sample.get("completions") or []
    pred = sample.get("pred")
    try:
        return completions.index(pred)
    except Exception:
        return -1


def _chosen_score_path(sample: Dict[str, Any], chosen_idx: int) -> List[float]:
    scores = sample.get("scores") or []
    if not isinstance(scores, list) or chosen_idx < 0:
        return []
    try:
        # scores shape: [num_beams][step_scores]
        return scores[chosen_idx] if 0 <= chosen_idx < len(scores) else []
    except Exception:
        return []


def _build_prefix_from_pred(pred_text: str, fail_step: int) -> str:
    # Keep steps up to fail_step-1
    steps = _split_steps(pred_text)
    if fail_step <= 1:
        return ""
    kept = steps[: max(0, fail_step - 1)]
    if not kept:
        return ""
    # Join with double newline to match step boundary and end with boundary
    joined = "\n\n".join(kept)
    return joined + "\n\n"


@click.group(help="Create and manage probe data records from inference outputs")
def cli() -> None:
    pass


@cli.command(name="add", help="Create a single probe datum from a run and sample index")
@click.option("--run-id", required=True, type=str, help="Run id under ./output/")
@click.option(
    "--index", required=True, type=int, help="Sample index in inference_output.jsonl"
)
@click.option(
    "--fail-step",
    required=True,
    type=int,
    help="First incorrect step (prefix keeps steps 1..fail_step-1)",
)
@click.option(
    "--prefix-mode",
    type=click.Choice(["model"], case_sensitive=False),
    default="model",
    show_default=True,
    help="How to construct prefix. Currently supports only model (from pred)",
)
def cmd_add(run_id: str, index: int, fail_step: int, prefix_mode: str) -> None:
    out_dir = Path("./output") / run_id
    src_file = out_dir / "inference_output.jsonl"
    records = _load_jsonl(src_file)
    if index < 0 or index >= len(records):
        click.secho(
            f"Index out of range: {index} (num records = {len(records)})",
            fg="red",
            err=True,
        )
        sys.exit(1)

    sample = records[index]

    problem = sample.get("problem")
    solution = sample.get("solution")
    answer = sample.get("answer")
    subject = sample.get("subject")
    level = sample.get("level")
    unique_id = sample.get("unique_id")
    pred_text = sample.get("pred", "")

    if prefix_mode.lower() == "model":
        prefix_text = _build_prefix_from_pred(pred_text, fail_step)
    else:
        prefix_text = ""

    chosen_idx = _choose_beam_index(sample)
    chosen_scores = _chosen_score_path(sample, chosen_idx)

    datum: Dict[str, Any] = {
        "run_id": run_id,
        "index": index,
        "fail_step": fail_step,
        "prefix_mode": prefix_mode,
        "step_delimiter": "\n\n",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        # Minimal context
        "problem": problem,
        "answer": answer,
        "subject": subject,
        "level": level,
        "unique_id": unique_id,
        # Prefix + reference material from model
        "prefix_text": prefix_text,
        "pred_full": pred_text,
        "chosen_beam_idx": chosen_idx,
        "chosen_score_path": chosen_scores,
        # Optional: retain the full solution text for later heuristics
        "solution": solution,
    }

    dst_dir = Path("./data/probe_data") / run_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / f"{index}.jsonl"

    with dst_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(datum, ensure_ascii=False))
        f.write("\n")

    click.secho(f"✔ Wrote probe datum to {dst_file}", fg="green")


if __name__ == "__main__":
    cli()
