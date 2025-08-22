#!/usr/bin/env python3
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np

from sal.evaluation.grader import math_equal
from sal.evaluation.parser import extract_answer

# Keep this consistent with `scripts/inference_visualiser.py`
ASSUMED_PRED_KEY = "pred_weighted@4"
BENCHMARK = "math"


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
                continue
    return records


def _wrap_in_boxed(s: str) -> str:
    return r"\boxed{" + s + "}"


def _get_question_answer_from_record(rec: Dict[str, Any]) -> QuestionAnswer:
    level = rec.get("level", "")
    answer = rec["answer"]
    unique_id: str = rec["unique_id"]
    pred = rec.get(ASSUMED_PRED_KEY, rec.get("pred", ""))

    answer_extracted = extract_answer(_wrap_in_boxed(answer), BENCHMARK)
    pred_extracted = extract_answer(pred, BENCHMARK)

    is_correct = math_equal(answer_extracted, pred_extracted)

    return QuestionAnswer(
        unique_id, answer_extracted, pred_extracted, is_correct, level
    )


def _safe_entropy(probs: List[float]) -> float:
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    max_h = math.log(len(probs)) if len(probs) > 1 else 1.0
    return float(h / max_h) if max_h > 0 else 0.0


def _compute_uncertainty_metrics(sample: Dict[str, Any]) -> Dict[str, Any]:
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
            "group_top_frac": 0.0,
        }

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

    agreement_ratio = (max(counts) / n) if counts else 0.0
    unique_answers = len(counts)
    freq_probs = [c / n for c in counts]
    entropy_freq = _safe_entropy(freq_probs)
    if sum_scores > 0 and len(scores_grouped) > 0:
        weighted_probs = [max(0.0, s) / sum_scores for s in scores_grouped]
        entropy_weighted = _safe_entropy(weighted_probs)
        group_top_frac = max(weighted_probs)
    else:
        entropy_weighted = 0.0
        group_top_frac = 0.0

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
        "group_top_frac": float(group_top_frac),
    }


DEFAULT_METRICS: Tuple[str, ...] = (
    "agreement_ratio",
    "entropy_freq",
    "entropy_weighted",
    "prm_margin",
    "prm_top_frac",
    "group_top_frac",
    "prm_std",
    "prm_mean",
)


def _metric_direction_low_is_uncertain(
    values: Sequence[float], labels: Sequence[bool]
) -> bool:
    corr_vals = [v for v, y in zip(values, labels) if y]
    inc_vals = [v for v, y in zip(values, labels) if not y]
    mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
    mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
    return mean_corr > mean_inc


def _compute_means_by_correctness(
    metrics_list: List[Dict[str, Any]], labels: List[bool], metrics: Sequence[str]
) -> Dict[str, Tuple[float, float, str]]:
    out: Dict[str, Tuple[float, float, str]] = {}
    for m in metrics:
        vals = [float(mm.get(m, 0.0)) for mm in metrics_list]
        corr_vals = [v for v, y in zip(vals, labels) if y]
        inc_vals = [v for v, y in zip(vals, labels) if not y]
        mean_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0.0
        mean_inc = sum(inc_vals) / len(inc_vals) if inc_vals else 0.0
        direction = "up" if mean_corr > mean_inc else "down"
        out[m] = (mean_corr, mean_inc, direction)
    return out


def _rank_indices_by_uncertainty(
    values: Sequence[float], labels: Sequence[bool]
) -> List[int]:
    low_is_uncertain = _metric_direction_low_is_uncertain(values, labels)
    idxs = list(range(len(values)))
    idxs.sort(key=lambda i: values[i], reverse=not low_is_uncertain)
    return idxs


def _compute_recall_and_ft(
    values: Sequence[float], labels: Sequence[bool], coverages: Sequence[int]
) -> Tuple[List[float], List[float], List[Tuple[int, int]]]:
    idxs = _rank_indices_by_uncertainty(values, labels)
    total_incorrect = sum(1 for y in labels if not y)
    n = len(labels)
    recall_vals: List[float] = []
    frac_incorrect_in_cov: List[float] = []
    ft_counts: List[Tuple[int, int]] = []  # (F, T)
    for p in coverages:
        k = max(1, int(round(n * (p / 100.0))))
        flagged = idxs[:k]
        f_cnt = sum(1 for i in flagged if not labels[i])
        t_cnt = sum(1 for i in flagged if labels[i])
        denom = max(1, f_cnt + t_cnt)
        recall = (100.0 * f_cnt / total_incorrect) if total_incorrect > 0 else 0.0
        pct_f = 100.0 * f_cnt / denom
        recall_vals.append(recall)
        frac_incorrect_in_cov.append(pct_f)
        ft_counts.append((f_cnt, t_cnt))
    return recall_vals, frac_incorrect_in_cov, ft_counts


def _ensure_outdir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)


def _save_means_plot(
    means: Dict[str, Tuple[float, float, str]], outdir: Path, run_id: str
) -> None:
    metrics = list(means.keys())
    mean_corr = [means[m][0] for m in metrics]
    mean_inc = [means[m][1] for m in metrics]

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 5))

    x = range(len(metrics))
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x],
        mean_corr,
        width=width,
        label="correct",
        color="#2ca02c",
    )
    ax.bar(
        [i + width / 2 for i in x],
        mean_inc,
        width=width,
        label="incorrect",
        color="#d62728",
    )

    ax.set_title(f"Means by correctness — {run_id}")
    ax.set_ylabel("mean value")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(outdir / "means_by_correctness.png", dpi=200)
    plt.close(fig)


def _save_recall_plot(
    per_metric_recall: Dict[str, List[float]],
    coverages: Sequence[int],
    outdir: Path,
    run_id: str,
    per_metric_ft_counts: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    annotate: str = "last",
) -> None:
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    for m, ys in per_metric_recall.items():
        ax.plot(coverages, ys, marker="o", linewidth=2, label=m)
        if per_metric_ft_counts and annotate in {"last", "all"}:
            counts = per_metric_ft_counts.get(m, [])
            for j, (x, y) in enumerate(zip(coverages, ys)):
                if annotate == "last" and j != len(ys) - 1:
                    continue
                if j < len(counts):
                    f_cnt, t_cnt = counts[j]
                    ax.annotate(
                        f"{f_cnt}:{t_cnt}",
                        xy=(x, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )
    ax.set_title(f"Recall of incorrect vs coverage — {run_id}")
    ax.set_xlabel("coverage (%)")
    ax.set_ylabel("recall of incorrect (%)")
    ax.set_xticks(list(coverages))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "recall_vs_coverage.png", dpi=200)
    plt.close(fig)


def _save_frac_incorrect_plot(
    per_metric_frac: Dict[str, List[float]],
    coverages: Sequence[int],
    outdir: Path,
    run_id: str,
    per_metric_ft_counts: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    annotate: str = "last",
) -> None:
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    for m, ys in per_metric_frac.items():
        ax.plot(coverages, ys, marker="s", linewidth=2, label=m)
        if per_metric_ft_counts and annotate in {"last", "all"}:
            counts = per_metric_ft_counts.get(m, [])
            for j, (x, y) in enumerate(zip(coverages, ys)):
                if annotate == "last" and j != len(ys) - 1:
                    continue
                if j < len(counts):
                    f_cnt, t_cnt = counts[j]
                    ax.annotate(
                        f"{f_cnt}:{t_cnt}",
                        xy=(x, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )
    ax.set_title(f"% incorrect within coverage — {run_id}")
    ax.set_xlabel("coverage (%)")
    ax.set_ylabel("% incorrect in selected subset")
    ax.set_xticks(list(coverages))
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "incorrect_fraction_vs_coverage.png", dpi=200)
    plt.close(fig)


def _save_distribution_plots(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: Sequence[str],
    outdir: Path,
    run_id: str,
) -> None:
    if not chosen_metrics:
        return
    plt.style.use("seaborn-v0_8")
    num_metrics = len(chosen_metrics)
    cols = 2 if num_metrics <= 6 else 3
    rows = (num_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
    for idx, m in enumerate(chosen_metrics):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        vals = [float(mm.get(m, 0.0)) for mm in metrics_list]
        corr_vals = [v for v, y in zip(vals, labels) if y]
        inc_vals = [v for v, y in zip(vals, labels) if not y]
        if not corr_vals and not inc_vals:
            ax.axis("off")
            continue
        all_vals = corr_vals + inc_vals
        if len(set(all_vals)) <= 1:
            ax.hist(all_vals, bins=5, color="#7f7f7f", alpha=0.7, density=True)
        else:
            low = min(all_vals)
            high = max(all_vals)
            bins = 30
            ax.hist(
                corr_vals,
                bins=bins,
                range=(low, high),
                alpha=0.5,
                density=True,
                label=f"correct (n={len(corr_vals)})",
                color="#2ca02c",
            )
            ax.hist(
                inc_vals,
                bins=bins,
                range=(low, high),
                alpha=0.5,
                density=True,
                label=f"incorrect (n={len(inc_vals)})",
                color="#d62728",
            )
            if corr_vals:
                ax.axvline(
                    sum(corr_vals) / len(corr_vals),
                    color="#2ca02c",
                    linestyle="--",
                    linewidth=1,
                )
            if inc_vals:
                ax.axvline(
                    sum(inc_vals) / len(inc_vals),
                    color="#d62728",
                    linestyle=":",
                    linewidth=1,
                )
        ax.set_title(m)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(frameon=False, fontsize=9)
    for j in range(num_metrics, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")
    fig.suptitle(f"Metric distributions by correctness — {run_id}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "distributions.png", dpi=200)
    plt.close(fig)


def _dump_summary_json(
    outdir: Path,
    means: Dict[str, Tuple[float, float, str]],
    coverages: Sequence[int],
    per_metric_recall: Dict[str, List[float]],
    per_metric_frac: Dict[str, List[float]],
    per_metric_ft_counts: Dict[str, List[Tuple[int, int]]],
    totals: Dict[str, int],
    selected_k_by_coverage: Dict[int, int],
) -> None:
    payload = {
        "means_by_correctness": {
            m: {
                "mean_correct": means[m][0],
                "mean_incorrect": means[m][1],
                "direction": "↑" if means[m][2] == "up" else "↓",
            }
            for m in means
        },
        "coverage_percentages": list(coverages),
        "selected_k_by_coverage": selected_k_by_coverage,
        "totals": totals,
        "recall_of_incorrect": per_metric_recall,
        "percent_incorrect_in_coverage": per_metric_frac,
        "ft_counts": {
            m: [(f, t) for (f, t) in v] for m, v in per_metric_ft_counts.items()
        },
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_cumulative_stacked(
    metrics_list: List[Dict[str, Any]],
    labels: List[bool],
    chosen_metrics: Sequence[str],
    outdir: Path,
    run_id: str,
    bins: int,
) -> None:
    if not chosen_metrics:
        return
    plt.style.use("seaborn-v0_8")
    num_metrics = len(chosen_metrics)
    cols = 2 if num_metrics <= 6 else 3
    rows = (num_metrics + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.5 * cols, 4.0 * rows), squeeze=False
    )

    n_total = len(labels)
    for idx, m in enumerate(chosen_metrics):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        vals = [float(mm.get(m, 0.0)) for mm in metrics_list]
        corr_vals = [v for v, y in zip(vals, labels) if y]
        inc_vals = [v for v, y in zip(vals, labels) if not y]
        if not corr_vals and not inc_vals:
            ax.axis("off")
            continue
        all_vals = corr_vals + inc_vals
        vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0.0, 1.0)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        counts_inc, edges = np.histogram(inc_vals, bins=bins, range=(vmin, vmax))
        counts_cor, _ = np.histogram(corr_vals, bins=bins, range=(vmin, vmax))
        cum_inc = np.cumsum(counts_inc) / max(1, n_total)
        cum_cor = np.cumsum(counts_cor) / max(1, n_total)

        x = edges[1:]  # right edges as x positions
        ax.fill_between(
            x, 0, cum_inc, step="pre", color="#d62728", alpha=0.5, label="incorrect"
        )
        ax.fill_between(
            x,
            cum_inc,
            cum_inc + cum_cor,
            step="pre",
            color="#2ca02c",
            alpha=0.5,
            label="correct",
        )
        ax.set_ylim(0, 1)
        ax.set_xlim(vmin, vmax)
        ax.set_title(m)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(frameon=False, fontsize=9)

    for j in range(num_metrics, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis("off")

    fig.suptitle("Cumulative stacked (≤ x) by correctness")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "cumulative_stacked.png", dpi=200)
    plt.close(fig)


def _parse_metrics_arg(values: Optional[Iterable[str]]) -> List[str]:
    if not values:
        return list(DEFAULT_METRICS)
    out: List[str] = []
    for v in values:
        if "," in v:
            out.extend([x.strip() for x in v.split(",") if x.strip()])
        else:
            vv = v.strip()
            if vv:
                out.append(vv)
    # Deduplicate preserving order
    seen = set()
    deduped: List[str] = []
    for m in out:
        if m in seen:
            continue
        seen.add(m)
        deduped.append(m)
    # Validate against supported
    supported = set(DEFAULT_METRICS)
    final = [m for m in deduped if m in supported]
    return final or list(DEFAULT_METRICS)


@click.command(help="Generate uncertainty analysis figures under ./figures/<run-id>/")
@click.option(
    "--run-id", required=True, type=str, help="W&B run id (directory under ./output)"
)
@click.option(
    "--metrics",
    multiple=True,
    type=str,
    help="Metrics to include (repeat or comma-separate). Supported: "
    + ", ".join(DEFAULT_METRICS),
)
@click.option(
    "--coverages",
    default="10,20,30,40,50",
    type=str,
    help="Coverage percentages CSV (e.g., '10,20,30,40,50')",
)
@click.option(
    "--annotate",
    type=click.Choice(["none", "last", "all"]),
    default="none",
    help="Annotate coverage plots with F:T counts (which points to label)",
)
@click.option(
    "--skip-distributions",
    is_flag=True,
    default=False,
    help="Skip generating per-metric distribution overlays",
)
@click.option(
    "--bins",
    type=int,
    default=50,
    help="Number of bins for cumulative stacked plots",
)
def main(
    run_id: str,
    metrics: Tuple[str, ...],
    coverages: str,
    annotate: str,
    skip_distributions: bool,
    bins: int,
) -> None:
    out_file = Path("./output") / run_id / "inference_output.jsonl"
    records = load_jsonl(out_file)

    chosen_metrics = _parse_metrics_arg(metrics)
    coverage_pcts: List[int] = [int(x) for x in coverages.split(",") if x.strip()]

    labels: List[bool] = []
    metrics_list: List[Dict[str, Any]] = []
    for rec in records:
        qa = _get_question_answer_from_record(rec)
        labels.append(bool(qa.is_correct))
        metrics_list.append(_compute_uncertainty_metrics(rec))

    # Means by correctness
    means = _compute_means_by_correctness(metrics_list, labels, chosen_metrics)

    # Coverage analyses
    per_metric_recall: Dict[str, List[float]] = {}
    per_metric_frac: Dict[str, List[float]] = {}
    per_metric_ft_counts: Dict[str, List[Tuple[int, int]]] = {}
    for m in chosen_metrics:
        vals = [float(mm.get(m, 0.0)) for mm in metrics_list]
        recall, frac_in_cov, ft_counts = _compute_recall_and_ft(
            vals, labels, coverage_pcts
        )
        per_metric_recall[m] = recall
        per_metric_frac[m] = frac_in_cov
        per_metric_ft_counts[m] = ft_counts

    # Output dir
    outdir = Path("./figures/selection/") / run_id
    _ensure_outdir(outdir)

    _save_means_plot(means, outdir, run_id)
    _save_recall_plot(
        per_metric_recall,
        coverage_pcts,
        outdir,
        run_id,
        per_metric_ft_counts=per_metric_ft_counts,
        annotate=annotate,
    )
    _save_frac_incorrect_plot(
        per_metric_frac,
        coverage_pcts,
        outdir,
        run_id,
        per_metric_ft_counts=per_metric_ft_counts,
        annotate=annotate,
    )
    if not skip_distributions:
        _save_distribution_plots(metrics_list, labels, chosen_metrics, outdir, run_id)
    _save_cumulative_stacked(
        metrics_list,
        labels,
        chosen_metrics,
        outdir,
        run_id,
        bins=bins,
    )
    # Summary extras
    totals = {
        "total_samples": len(labels),
        "total_correct": int(sum(1 for y in labels if y)),
        "total_incorrect": int(sum(1 for y in labels if not y)),
    }
    selected_k_by_coverage = {
        p: max(1, int(round(len(labels) * (p / 100.0)))) for p in coverage_pcts
    }

    _dump_summary_json(
        outdir,
        means,
        coverage_pcts,
        per_metric_recall,
        per_metric_frac,
        per_metric_ft_counts,
        totals,
        selected_k_by_coverage,
    )

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
