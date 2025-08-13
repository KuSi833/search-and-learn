#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_add_src_to_path()

# After sys.path adjustment, import project helpers
from sal.utils.qwen_math_parser import (  # type: ignore  # noqa: E402
    extract_answer,
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
                continue
    return records


def analyse_record(record: Dict[str, Any]) -> Tuple[str, bool]:
    level = str(record.get("level", "unknown"))
    answer = str(record.get("answer", ""))
    pred = str(record.get("pred", ""))
    extracted_pred = extract_answer(pred, "math") if isinstance(pred, str) else ""
    try:
        is_correct = math_equal(extracted_pred, answer)
    except Exception:
        is_correct = False
    return level, bool(is_correct)


@click.command(
    help="Overview of a run: counts per difficulty and correctness with indices"
)
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd(),
    show_default=True,
    help="Project root",
)
def main(run_id: str, base_dir: Path) -> None:
    out_file = base_dir.resolve() / "output" / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        print(f"No records found in {out_file}")
        sys.exit(1)

    # Aggregations
    level_to_total: Dict[str, int] = defaultdict(int)
    level_to_correct: Dict[str, int] = defaultdict(int)
    level_to_correct_indices: Dict[str, List[int]] = defaultdict(list)
    level_to_incorrect_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, rec in enumerate(records):
        level, correct = analyse_record(rec)
        level_to_total[level] += 1
        if correct:
            level_to_correct[level] += 1
            level_to_correct_indices[level].append(idx)
        else:
            level_to_incorrect_indices[level].append(idx)

    # Print overview
    print(f"Run: {run_id}")
    print(f"File: output/{run_id}/inference_output.jsonl")
    print("-")
    levels_sorted = sorted(level_to_total.keys())
    for level in levels_sorted:
        total = level_to_total[level]
        correct = level_to_correct[level]
        incorrect = total - correct
        print(f"Level: {level}")
        print(f"  Total: {total}")
        print(f"  Correct: {correct}")
        print(f"  Incorrect: {incorrect}")
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
    main()
