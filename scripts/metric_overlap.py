#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sal.evaluation.grader import math_equal
from sal.evaluation.parser import extract_answer
from sal.utils.runs import fusion_base_runs_best

console = Console()


# Assumptions matching the rest of the tooling
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
    # Extract per-completion final answers and aggregate PRM scores
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
            "entropy_freq": 0.0,
            "consensus_support": 0.0,
        }

    # Group by extracted answer
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
        "n": n,
        "agreement_ratio": float(agreement_ratio),
        "entropy_freq": float(entropy_freq),
        "consensus_support": float(consensus_support),
    }


def _metric_direction_low_is_uncertain(values: List[float], labels: List[bool]) -> bool:
    # If the metric is on average higher for correct answers, low suggests uncertainty
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


def _recall_of_incorrect(order: List[int], labels: List[bool], k: int) -> float:
    total_incorrect = sum(1 for y in labels if not y)
    if total_incorrect == 0:
        return 0.0
    flagged = set(order[:k])
    incorrect_flagged = sum(1 for i in flagged if not labels[i])
    return 100.0 * incorrect_flagged / total_incorrect


def _jaccard(a: Set[int], b: Set[int]) -> float:
    u = len(a | b)
    return (len(a & b) / u) if u > 0 else 0.0


@click.group(help="Metric overlap and combo analysis")
def cli() -> None:
    pass


@cli.command(
    name="overlap", help="Overlap of top-K uncertain sets across three metrics"
)
@click.option("--run-id", required=True, type=str, help="Run id (./output/<run>/...)")
@click.option(
    "--coverages",
    default="10,20,30,40,50",
    type=str,
    help="Comma-separated coverage percentages (e.g., '10,20,50')",
)
def overlap(run_id: str, coverages: str) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        sys.exit(1)

    # Compute labels and metrics
    labels: List[bool] = []
    mlist: List[Dict[str, Any]] = []
    for rec in records:
        labels.append(_is_correct_record(rec))
        mlist.append(_compute_core_metrics(rec))

    total = len(labels)
    acc = (100.0 * sum(1 for y in labels if y) / total) if total > 0 else 0.0

    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("File", f"output/{run_id}/inference_output.jsonl")
    header.add_row("Samples", str(total))
    header.add_row("Accuracy (assumed pred)", f"{acc:.1f}%")
    console.print(Panel(header, title="Metric overlap analysis", box=box.ROUNDED))

    # Build oriented rankings
    order_agree = _order_indices_by_metric(mlist, labels, "agreement_ratio")
    order_entr = _order_indices_by_metric(mlist, labels, "entropy_freq")
    order_group = _order_indices_by_metric(mlist, labels, "consensus_support")

    coverage_pcts = [int(x) for x in coverages.split(",") if x.strip()]

    # Overlap summary table
    table = Table(title="Overlap of top-K uncertain sets", box=box.SIMPLE_HEAVY)
    table.add_column("Pair", style="bold")
    for p in coverage_pcts:
        table.add_column(f"{p}% (cnt;Jacc%)", justify="center")

    for pair_name, a_order, b_order in [
        ("agree ∩ entropy", order_agree, order_entr),
        ("agree ∩ group", order_agree, order_group),
        ("entropy ∩ group", order_entr, order_group),
    ]:
        row_vals: List[str] = []
        for p in coverage_pcts:
            k = max(1, int(round(total * (p / 100.0))))
            a_set, b_set = set(a_order[:k]), set(b_order[:k])
            cnt = len(a_set & b_set)
            j = 100.0 * _jaccard(a_set, b_set)
            row_vals.append(f"{cnt};{j:.1f}")
        table.add_row(pair_name, *row_vals)

    # Triple overlap
    triple = Table(title="Triple overlap (all three)", box=box.SIMPLE_HEAVY)
    triple.add_column("Coverage", style="bold")
    triple.add_column("Count", justify="right")
    triple.add_column("% of k", justify="right")
    triple.add_column("3-set Jacc%", justify="right")
    for p in coverage_pcts:
        k = max(1, int(round(total * (p / 100.0))))
        a_set, b_set, c_set = (
            set(order_agree[:k]),
            set(order_entr[:k]),
            set(order_group[:k]),
        )
        inter = a_set & b_set & c_set
        union3 = a_set | b_set | c_set
        cnt = len(inter)
        pct_of_k = 100.0 * cnt / k if k > 0 else 0.0
        j3 = 100.0 * (len(inter) / len(union3)) if len(union3) > 0 else 0.0
        triple.add_row(str(p), str(cnt), f"{pct_of_k:.1f}", f"{j3:.1f}")

    console.print(table)
    console.print(triple)


def _precision_recall_f1(flags: Set[int], labels: List[bool]) -> tuple:
    if not flags:
        return (0.0, 0.0, 0.0)
    tp = sum(1 for i in flags if not labels[i])  # correctly flagged incorrect
    fp = sum(1 for i in flags if labels[i])  # flagged but actually correct
    fn = sum(1 for i, y in enumerate(labels) if (not y) and (i not in flags))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision * 2 + recall * 2))
        if (precision + recall) > 0
        else 0.0
    )
    return (precision, recall, f1)


@cli.command(
    name="boolean-combos",
    help=(
        "Evaluate OR/AND combinations of agreement_ratio, entropy_freq, consensus_support. "
        "At each coverage p, each metric flags its top-K most uncertain; OR=union, AND=intersection."
    ),
)
@click.option("--run-id", required=True, type=str, help="Run id (./output/<run>/...)")
@click.option(
    "--coverages",
    default="10,20,30,40,50",
    type=str,
    help="Comma-separated coverage percentages (e.g., '10,20,50')",
)
def boolean_combos(run_id: str, coverages: str) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)
    if not records:
        console.print(Text(f"No records found in {out_file}", style="red"))
        sys.exit(1)

    labels: List[bool] = []
    mlist: List[Dict[str, Any]] = []
    for rec in records:
        labels.append(_is_correct_record(rec))
        mlist.append(_compute_core_metrics(rec))

    total = len(labels)
    coverage_pcts = [int(x) for x in coverages.split(",") if x.strip()]

    # Rankings per metric
    order_agree = _order_indices_by_metric(mlist, labels, "agreement_ratio")
    order_entr = _order_indices_by_metric(mlist, labels, "entropy_freq")
    order_group = _order_indices_by_metric(mlist, labels, "consensus_support")

    combos = [
        ("agree+entropy", (order_agree, order_entr)),
        ("agree+group", (order_agree, order_group)),
        ("entropy+group", (order_entr, order_group)),
        ("agree+entropy+group", (order_agree, order_entr, order_group)),
    ]

    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("File", f"output/{run_id}/inference_output.jsonl")
    header.add_row("Samples", str(total))
    console.print(Panel(header, title="Boolean combos (OR / AND)", box=box.ROUNDED))

    # Single-metric baselines (same format as above)
    single_table = Table(title="Single metric baselines", box=box.SIMPLE_HEAVY)
    single_table.add_column("Metric", style="bold")
    for p in coverage_pcts:
        single_table.add_column(f"{p}% (cnt;T:F;prec;rec;F1)", justify="center")

    for metric_name, order in [
        ("agreement_ratio", order_agree),
        ("entropy_freq", order_entr),
        ("consensus_support", order_group),
    ]:
        cells: List[str] = []
        for p in coverage_pcts:
            k = max(1, int(round(total * (p / 100.0))))
            s = set(order[:k])
            prec, rec, f1 = _precision_recall_f1(s, labels)
            tp = sum(1 for i in s if not labels[i])
            fp = sum(1 for i in s if labels[i])
            cells.append(
                f"{len(s)};{tp}:{fp};{prec * 100:.1f};{rec * 100:.1f};{f1 * 100:.1f}"
            )
        single_table.add_row(metric_name, *cells)

    # OR results table (precision/recall/F1;count)
    or_table = Table(title="OR = any metric flags false", box=box.SIMPLE_HEAVY)
    or_table.add_column("Combo", style="bold")
    for p in coverage_pcts:
        or_table.add_column(f"{p}% (cnt;T:F;prec;rec;F1)", justify="center")

    # AND results table
    and_table = Table(title="AND = all metrics agree false", box=box.SIMPLE_HEAVY)
    and_table.add_column("Combo", style="bold")
    for p in coverage_pcts:
        and_table.add_column(f"{p}% (cnt;T:F;prec;rec;F1)", justify="center")

    for name, orders in combos:
        or_cells: List[str] = []
        and_cells: List[str] = []
        for p in coverage_pcts:
            k = max(1, int(round(total * (p / 100.0))))
            sets = [set(o[:k]) for o in orders]
            or_set: Set[int] = set().union(*sets)
            and_set: Set[int] = set.intersection(*sets) if sets else set()

            prec_or, rec_or, f1_or = _precision_recall_f1(or_set, labels)
            prec_and, rec_and, f1_and = _precision_recall_f1(and_set, labels)

            tp_or = sum(1 for i in or_set if not labels[i])
            fp_or = sum(1 for i in or_set if labels[i])
            tp_and = sum(1 for i in and_set if not labels[i])
            fp_and = sum(1 for i in and_set if labels[i])

            or_cells.append(
                f"{len(or_set)};{tp_or}:{fp_or};{prec_or * 100:.1f};{rec_or * 100:.1f};{f1_or * 100:.1f}"
            )
            and_cells.append(
                f"{len(and_set)};{tp_and}:{fp_and};{prec_and * 100:.1f};{rec_and * 100:.1f};{f1_and * 100:.1f}"
            )

        or_table.add_row(name, *or_cells)
        and_table.add_row(name, *and_cells)

    console.print(single_table)
    console.print(or_table)
    console.print(and_table)


def boolean_combos_experiment() -> None:
    run_ids = fusion_base_runs_best()
    coverages = [10, 20, 30, 40, 50]

    if not run_ids:
        console.print(Text("No run ids provided", style="red"))
        return

    coverage_pcts = coverages

    # Aggregators for averages across runs
    metric_names = ["agreement_ratio", "entropy_freq", "consensus_support"]
    combo_defs = [
        ("agree+entropy", ("agreement_ratio", "entropy_freq")),
        ("agree+group", ("agreement_ratio", "consensus_support")),
        ("entropy+group", ("entropy_freq", "consensus_support")),
        (
            "agree+entropy+group",
            ("agreement_ratio", "entropy_freq", "consensus_support"),
        ),
    ]

    def _blank_cell():
        return {"cnt": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}

    agg_single: Dict[str, Dict[int, Dict[str, float]]] = {
        m: {p: _blank_cell() for p in coverage_pcts} for m in metric_names
    }
    agg_or: Dict[str, Dict[int, Dict[str, float]]] = {
        name: {p: _blank_cell() for p in coverage_pcts} for name, _ in combo_defs
    }
    agg_and: Dict[str, Dict[int, Dict[str, float]]] = {
        name: {p: _blank_cell() for p in coverage_pcts} for name, _ in combo_defs
    }

    successful_runs = []

    for run_id in run_ids:
        out_file = Path("./output") / run_id / "inference_output.jsonl"
        try:
            records = load_jsonl(out_file)
        except FileNotFoundError:
            console.print(
                Text(f"Missing file for run {run_id}: {out_file}", style="yellow")
            )
            continue

        if not records:
            console.print(
                Text(f"No records in {out_file}; skipping run {run_id}", style="yellow")
            )
            continue

        labels: List[bool] = []
        mlist: List[Dict[str, Any]] = []
        for rec in records:
            labels.append(_is_correct_record(rec))
            mlist.append(_compute_core_metrics(rec))

        total = len(labels)
        if total == 0:
            console.print(
                Text(f"Zero samples for run {run_id}; skipping", style="yellow")
            )
            continue

        # Build orders per metric for this run
        orders_by_metric: Dict[str, List[int]] = {
            "agreement_ratio": _order_indices_by_metric(
                mlist, labels, "agreement_ratio"
            ),
            "entropy_freq": _order_indices_by_metric(mlist, labels, "entropy_freq"),
            "consensus_support": _order_indices_by_metric(
                mlist, labels, "consensus_support"
            ),
        }

        # Accumulate single metric baselines
        for p in coverage_pcts:
            k = max(1, int(round(total * (p / 100.0))))
            for metric_name in metric_names:
                order = orders_by_metric[metric_name]
                s = set(order[:k])
                prec, rec, f1 = _precision_recall_f1(s, labels)
                cell = agg_single[metric_name][p]
                cell["cnt"] += float(len(s))
                cell["prec"] += prec
                cell["rec"] += rec
                cell["f1"] += f1

        # Accumulate combos
        for p in coverage_pcts:
            k = max(1, int(round(total * (p / 100.0))))
            for combo_name, combo_metrics in combo_defs:
                sets = [set(orders_by_metric[m][:k]) for m in combo_metrics]
                or_set: Set[int] = set().union(*sets)
                and_set: Set[int] = set.intersection(*sets) if sets else set()

                prec_or, rec_or, f1_or = _precision_recall_f1(or_set, labels)
                prec_and, rec_and, f1_and = _precision_recall_f1(and_set, labels)

                cell_or = agg_or[combo_name][p]
                cell_or["cnt"] += float(len(or_set))
                cell_or["prec"] += prec_or
                cell_or["rec"] += rec_or
                cell_or["f1"] += f1_or

                cell_and = agg_and[combo_name][p]
                cell_and["cnt"] += float(len(and_set))
                cell_and["prec"] += prec_and
                cell_and["rec"] += rec_and
                cell_and["f1"] += f1_and

        successful_runs.append(run_id)

    num_runs = len(successful_runs)
    if num_runs == 0:
        console.print(Text("No valid runs to aggregate", style="red"))
        return

    # Header
    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Runs", ", ".join(successful_runs))
    header.add_row("#Runs", str(num_runs))
    header.add_row("Coverages", ", ".join(str(p) for p in coverage_pcts))
    console.print(
        Panel(header, title="Boolean combos (averaged across runs)", box=box.ROUNDED)
    )

    # Single metric baselines table (averaged)
    single_table = Table(title="Single metric baselines (avg)", box=box.SIMPLE_HEAVY)
    single_table.add_column("Metric", style="bold")
    for p in coverage_pcts:
        single_table.add_column(f"{p}% (cnt;prec;rec;F1)", justify="center")

    for metric_name in metric_names:
        cells: List[str] = []
        for p in coverage_pcts:
            cell = agg_single[metric_name][p]
            cnt_avg = cell["cnt"] / num_runs
            prec_avg = cell["prec"] / num_runs
            rec_avg = cell["rec"] / num_runs
            f1_avg = cell["f1"] / num_runs
            cells.append(
                f"{cnt_avg:.1f};{prec_avg * 100:.1f};{rec_avg * 100:.1f};{f1_avg * 100:.1f}"
            )
        single_table.add_row(metric_name, *cells)

    # OR table (averaged)
    or_table = Table(title="OR = any metric flags false (avg)", box=box.SIMPLE_HEAVY)
    or_table.add_column("Combo", style="bold")
    for p in coverage_pcts:
        or_table.add_column(f"{p}% (cnt;prec;rec;F1)", justify="center")

    for combo_name, _ in combo_defs:
        cells: List[str] = []
        for p in coverage_pcts:
            cell = agg_or[combo_name][p]
            cnt_avg = cell["cnt"] / num_runs
            prec_avg = cell["prec"] / num_runs
            rec_avg = cell["rec"] / num_runs
            f1_avg = cell["f1"] / num_runs
            cells.append(
                f"{cnt_avg:.1f};{prec_avg * 100:.1f};{rec_avg * 100:.1f};{f1_avg * 100:.1f}"
            )
        or_table.add_row(combo_name, *cells)

    # AND table (averaged)
    and_table = Table(title="AND = all metrics agree false (avg)", box=box.SIMPLE_HEAVY)
    and_table.add_column("Combo", style="bold")
    for p in coverage_pcts:
        and_table.add_column(f"{p}% (cnt;prec;rec;F1)", justify="center")

    for combo_name, _ in combo_defs:
        cells: List[str] = []
        for p in coverage_pcts:
            cell = agg_and[combo_name][p]
            cnt_avg = cell["cnt"] / num_runs
            prec_avg = cell["prec"] / num_runs
            rec_avg = cell["rec"] / num_runs
            f1_avg = cell["f1"] / num_runs
            cells.append(
                f"{cnt_avg:.1f};{prec_avg * 100:.1f};{rec_avg * 100:.1f};{f1_avg * 100:.1f}"
            )
        and_table.add_row(combo_name, *cells)

    console.print(single_table)
    console.print(or_table)
    console.print(and_table)


if __name__ == "__main__":
    # boolean_combos_experiment()
    cli()
