#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sal.evaluation.grader import math_equal
from sal.evaluation.parser import extract_answer
from sal.utils.constants import BENCHMARK_SUBSETS_ROOT, Benchmarks

console = Console()


ASSUMED_PRED_KEY = "pred_weighted@4"
BENCHMARK = "math"


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


def _wrap_in_boxed(s: str) -> str:
    return r"\boxed{" + s + "}"


def _is_correct_record(rec: Dict[str, Any]) -> bool:
    answer = rec["answer"]
    pred = rec.get(ASSUMED_PRED_KEY, rec.get("pred", ""))
    answer_extracted = extract_answer(_wrap_in_boxed(answer), BENCHMARK)
    pred_extracted = extract_answer(pred or "", BENCHMARK)
    return bool(math_equal(answer_extracted, pred_extracted))


def _safe_entropy(probs: List[float]) -> float:
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    max_h = math.log(len(probs)) if len(probs) > 1 else 1.0
    return float(h / max_h) if max_h > 0 else 0.0


def _compute_core_metrics(sample: Dict[str, Any]) -> Dict[str, Any]:
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
            "agreement_ratio": 0.0,
            "entropy_freq": 0.0,
            "consensus_support": 0.0,
        }

    count_by_ans: Dict[str, int] = {}
    score_by_ans: Dict[str, float] = {}
    for ans, s in zip(answers, agg_scores or [0.0] * n):
        count_by_ans[ans] = count_by_ans.get(ans, 0) + 1
        try:
            score_by_ans[ans] = score_by_ans.get(ans, 0.0) + float(s)
        except Exception:
            score_by_ans[ans] = score_by_ans.get(ans, 0.0) + 0.0

    counts = list(count_by_ans.values())
    freq_probs = [c / n for c in counts] if n > 0 else []
    entropy_freq = _safe_entropy(freq_probs)
    agreement_ratio = (max(counts) / n) if counts else 0.0

    scores_grouped = list(score_by_ans.values())
    sum_scores = float(sum(agg_scores)) if agg_scores else 0.0
    if sum_scores > 0 and len(scores_grouped) > 0:
        weighted_probs = [max(0.0, s) / sum_scores for s in scores_grouped]
        consensus_support = max(weighted_probs)
    else:
        consensus_support = 0.0

    return {
        "agreement_ratio": float(agreement_ratio),
        "entropy_freq": float(entropy_freq),
        "consensus_support": float(consensus_support),
    }


def _metric_direction_low_is_uncertain(values: List[float], labels: List[bool]) -> bool:
    corr_vals = [v for v, y in zip(values, labels) if y]
    inc_vals = [v for v, y in zip(values, labels) if not y]
    mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
    mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
    return mean_corr > mean_inc


def _order_indices_by_metric(
    metrics: List[Dict[str, Any]], labels: List[bool], field: str
) -> List[int]:
    vals = [m.get(field, 0.0) for m in metrics]
    low_is_uncertain = _metric_direction_low_is_uncertain(vals, labels)
    idxs = list(range(len(metrics)))
    idxs.sort(key=lambda i: vals[i], reverse=not low_is_uncertain)
    return idxs


def _export_subset(
    run_id: str,
    benchmark_key: str,
    name: str,
    unique_ids: List[str],
) -> Path:
    benchmark = Benchmarks.from_key(benchmark_key)
    output_dir = BENCHMARK_SUBSETS_ROOT / benchmark.hf_name / run_id / "coverage"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{name}.json"
    payload = {
        "version": 1,
        "type": "uncertain_subset",
        "benchmark_key": benchmark.key,
        "hf_name": benchmark.hf_name,
        "run_id": run_id,
        "name": name,
        "unique_ids": sorted(unique_ids),
    }
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)
    return output_file


@click.command(
    help=(
        "Export subsets selected by single metrics or boolean OR/AND combos of"
        " agreement_ratio, entropy_freq, consensus_support at a given coverage."
    )
)
@click.option("--run-id", required=True, type=str, help="Run id (./output/<run>/...)")
@click.option(
    "--benchmark",
    "benchmark_key",
    default=Benchmarks.MATH500.value.key,
    type=click.Choice([b.value.key for b in Benchmarks]),
    help="Benchmark key (e.g., math500)",
)
@click.option(
    "--coverage",
    required=True,
    type=int,
    help="Coverage percentage to target for each single metric (K=ceil(N*p/100)).",
)
@click.option(
    "--export-or",
    is_flag=True,
    help="Also export OR (union) sets for pairs and all three metrics",
)
@click.option(
    "--export-and",
    is_flag=True,
    help="Also export AND (intersection) sets for pairs and all three metrics",
)
def main(
    run_id: str, benchmark_key: str, coverage: int, export_or: bool, export_and: bool
) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        sys.exit(1)

    labels: List[bool] = []
    metrics_list: List[Dict[str, Any]] = []
    for rec in records:
        labels.append(_is_correct_record(rec))
        metrics_list.append(_compute_core_metrics(rec))

    total = len(labels)
    k = max(1, int(round(total * (coverage / 100.0))))

    # Rankings per metric (uncertain-first)
    order_agree = _order_indices_by_metric(metrics_list, labels, "agreement_ratio")
    order_entr = _order_indices_by_metric(metrics_list, labels, "entropy_freq")
    order_group = _order_indices_by_metric(metrics_list, labels, "consensus_support")

    idx_to_id = [rec["unique_id"] for rec in records]

    # Single metric subsets
    for metric_name, order in [
        ("agreement_ratio", order_agree),
        ("entropy_freq", order_entr),
        ("consensus_support", order_group),
    ]:
        ids = [idx_to_id[i] for i in order[:k]]
        name = f"single__{metric_name}__{coverage}"
        path = _export_subset(run_id, benchmark_key, name, ids)
        console.print(Text.assemble("Exported ", (name, "yellow"), " -> ", str(path)))

    # Boolean combos
    pairs = [
        ("agree_entropy", (order_agree, order_entr)),
        ("agree_group", (order_agree, order_group)),
        ("entropy_group", (order_entr, order_group)),
    ]
    triples = [("agree_entropy_group", (order_agree, order_entr, order_group))]

    def ids_from_indices(indices: Set[int]) -> List[str]:
        return [idx_to_id[i] for i in sorted(indices)]

    if export_or:
        for name, orders in pairs + triples:
            sets = [set(o[:k]) for o in orders]
            union = set().union(*sets)
            ids = ids_from_indices(union)
            out_name = f"or__{name}__{coverage}"
            path = _export_subset(run_id, benchmark_key, out_name, ids)
            console.print(
                Text.assemble("Exported ", (out_name, "yellow"), " -> ", str(path))
            )

    if export_and:
        for name, orders in pairs + triples:
            sets = [set(o[:k]) for o in orders]
            inter = set.intersection(*sets) if sets else set()
            ids = ids_from_indices(inter)
            out_name = f"and__{name}__{coverage}"
            path = _export_subset(run_id, benchmark_key, out_name, ids)
            console.print(
                Text.assemble("Exported ", (out_name, "yellow"), " -> ", str(path))
            )

    # Summary panel
    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("Benchmark", benchmark_key)
    header.add_row("Coverage", f"{coverage}%")
    header.add_row(
        "Output root",
        str(
            BENCHMARK_SUBSETS_ROOT
            / Benchmarks.from_key(benchmark_key).hf_name
            / run_id
            / "coverage"
        ),
    )
    console.print(Panel(header, title="Exported subsets", border_style="green"))


if __name__ == "__main__":
    main()
