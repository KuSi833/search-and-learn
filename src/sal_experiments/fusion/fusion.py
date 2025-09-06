from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich import box
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from sal.evaluation.grader import math_equal
from sal.evaluation.parser import extract_answer

console = Console()


# --------------- Core utilities (offline; reuse logic from visualiser) ---------------


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


def _is_correct_record_math(rec: Dict[str, Any]) -> bool:
    # Assumes benchmark is math
    answer = rec["answer"]
    pred = rec.get("pred_weighted@4") or rec.get("pred") or ""
    ans_ex = extract_answer(_wrap_in_boxed(answer), "math")
    pred_ex = extract_answer(pred, "math")
    return bool(math_equal(ans_ex, pred_ex))


def _safe_entropy(probs: List[float]) -> float:
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    import math

    h = -sum(p * math.log(p) for p in probs)
    max_h = math.log(len(probs)) if len(probs) > 1 else 1.0
    return float(h / max_h) if max_h > 0 else 0.0


def _compute_uncertainty_metrics(sample: Dict[str, Any]) -> Dict[str, Any]:
    completions: List[str] = sample.get("completions", [])
    answers: List[str] = [extract_answer(c or "", "math") for c in completions]
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

    from collections import defaultdict

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
        import math

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


def _get_confidence(sample: Dict[str, Any], metric: str) -> float:
    try:
        m = _compute_uncertainty_metrics(sample)
        v = m.get(metric, 0.0)
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


def _load_subset_ids(path: Path) -> List[str]:
    with path.open("r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        ids = obj.get("unique_ids")
        if isinstance(ids, list):
            return [str(x) for x in ids]
    return []


# --------------- Experiment configuration ---------------


@dataclass
class FusionSetting:
    metric: str
    delta: float = 0.0
    min_rerun_conf: Optional[float] = None
    max_base_conf: Optional[float] = None


@dataclass
class FusionRunConfig:
    base_run_id: str
    rerun_id: str
    subset: Optional[Path] = None
    settings: Sequence[FusionSetting] = ()
    save_dir: Optional[Path] = None


@dataclass
class FusionResult:
    metric: str
    delta: float
    min_rerun_conf: Optional[float]
    max_base_conf: Optional[float]
    total_samples: int
    rerun_samples: int
    overrides_used: int
    acc_base: float
    acc_rerun: float
    acc_fused: float
    flips_pos: int
    flips_neg: int
    # New T/F conversion tracking
    conversions_tt: int = 0  # True -> True
    conversions_ff: int = 0  # False -> False
    conversions_tf: int = 0  # True -> False
    conversions_ft: int = 0  # False -> True


def _get_base_samples(
    base_recs: Dict[str, Dict[str, Any]],
    rerun_recs: Dict[str, Dict[str, Any]],
    subset: Optional[Path],
) -> Tuple[List[str], List[str]]:
    """Returns (all_base_ids, rerun_ids) where rerun_ids is subset of all_base_ids"""
    if subset is None:
        all_base_ids = list(base_recs.keys())
    else:
        ids = set(_load_subset_ids(subset))
        all_base_ids = [uid for uid in ids if uid in base_recs]

    # Rerun IDs are those that exist in both base and rerun
    rerun_ids = [uid for uid in all_base_ids if uid in rerun_recs]

    return all_base_ids, rerun_ids


def _acc(records_map: Dict[str, Dict[str, Any]], ids: List[str]) -> float:
    return (
        100.0
        * sum(1 for uid in ids if _is_correct_record_math(records_map[uid]))
        / len(ids)
        if ids
        else 0.0
    )


def run_fusion_once(
    base_recs: Dict[str, Dict[str, Any]],
    rerun_recs: Dict[str, Dict[str, Any]],
    all_base_ids: List[str],
    rerun_ids: List[str],
    setting: FusionSetting,
) -> FusionResult:
    # Calculate baseline accuracies
    acc_base = _acc(base_recs, all_base_ids)
    acc_rerun = _acc(rerun_recs, rerun_ids) if rerun_ids else 0.0

    fused_correct = 0
    overrides_used = 0
    flips_pos = 0
    flips_neg = 0

    # T/F conversion tracking
    conversions_tt = 0  # True -> True
    conversions_ff = 0  # False -> False
    conversions_tf = 0  # True -> False
    conversions_ft = 0  # False -> True

    for uid in all_base_ids:
        base_rec = base_recs[uid]
        base_ok = _is_correct_record_math(base_rec)

        # Start with base result
        final_ok = base_ok

        # Check if we should override with rerun
        if uid in rerun_recs:
            rerun_rec = rerun_recs[uid]
            base_conf = _get_confidence(base_rec, setting.metric)
            rerun_conf = _get_confidence(rerun_rec, setting.metric)

            # Decide whether to use rerun override
            use_rerun = rerun_conf > (base_conf + setting.delta)

            # Apply confidence filters
            if (
                use_rerun
                and (setting.min_rerun_conf is not None)
                and not (rerun_conf >= float(setting.min_rerun_conf))
            ):
                use_rerun = False
            if (
                use_rerun
                and (setting.max_base_conf is not None)
                and not (base_conf <= float(setting.max_base_conf))
            ):
                use_rerun = False

            if use_rerun:
                rerun_ok = _is_correct_record_math(rerun_rec)
                final_ok = rerun_ok
                overrides_used += 1

                # Track flips only when we actually override
                if (not base_ok) and final_ok:
                    flips_pos += 1
                elif base_ok and (not final_ok):
                    flips_neg += 1

                # Track T/F conversions only for overrides
                if base_ok and rerun_ok:
                    conversions_tt += 1
                elif (not base_ok) and (not rerun_ok):
                    conversions_ff += 1
                elif base_ok and (not rerun_ok):
                    conversions_tf += 1
                elif (not base_ok) and rerun_ok:
                    conversions_ft += 1

        fused_correct += 1 if final_ok else 0

    acc_fused = 100.0 * fused_correct / len(all_base_ids) if all_base_ids else 0.0

    return FusionResult(
        metric=setting.metric,
        delta=setting.delta,
        min_rerun_conf=setting.min_rerun_conf,
        max_base_conf=setting.max_base_conf,
        total_samples=len(all_base_ids),
        rerun_samples=len(rerun_ids),
        overrides_used=overrides_used,
        acc_base=acc_base,
        acc_rerun=acc_rerun,
        acc_fused=acc_fused,
        flips_pos=flips_pos,
        flips_neg=flips_neg,
        conversions_tt=conversions_tt,
        conversions_ff=conversions_ff,
        conversions_tf=conversions_tf,
        conversions_ft=conversions_ft,
    )


def run_sweep(cfg: FusionRunConfig) -> List[FusionResult]:
    base_file = Path("./output") / cfg.base_run_id / "inference_output.jsonl"
    rerun_file = Path("./output") / cfg.rerun_id / "inference_output.jsonl"
    base_recs = {rec["unique_id"]: rec for rec in load_jsonl(base_file)}
    rerun_recs = {rec["unique_id"]: rec for rec in load_jsonl(rerun_file)}

    all_base_ids, rerun_ids = _get_base_samples(base_recs, rerun_recs, cfg.subset)
    if not all_base_ids:
        raise SystemExit("No base samples found")

    results: List[FusionResult] = []
    for setting in tqdm(cfg.settings, desc="Running Sweep"):
        res = run_fusion_once(base_recs, rerun_recs, all_base_ids, rerun_ids, setting)
        results.append(res)

    if cfg.save_dir is not None:
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        out_file = cfg.save_dir / f"{cfg.base_run_id}__{cfg.rerun_id}.json"
        with out_file.open("w") as f:
            json.dump([res.__dict__ for res in results], f, indent=2)

    # Print analysis to command line
    print_results_table(results)
    print_best_delta_analysis(results)

    return results


def print_results_table(results: List[FusionResult]) -> None:
    table = Table(title="Base/Rerun fusion sweep", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="bold")
    table.add_column("delta", justify="right")
    table.add_column("min_rerun", justify="right")
    table.add_column("max_base", justify="right")
    table.add_column("total_samples", justify="right")
    table.add_column("rerun_samples", justify="right")
    table.add_column("overrides", justify="right")
    table.add_column("acc_base", justify="right")
    table.add_column("acc_rerun", justify="right")
    table.add_column("acc_fused", justify="right")
    table.add_column("F->T", justify="right")
    table.add_column("T->F", justify="right")
    table.add_column("net", justify="right")

    # Sort by fused accuracy descending
    for r in sorted(results, key=lambda x: x.acc_fused, reverse=True):
        table.add_row(
            r.metric,
            f"{r.delta:.3f}",
            "-" if r.min_rerun_conf is None else f"{r.min_rerun_conf:.3f}",
            "-" if r.max_base_conf is None else f"{r.max_base_conf:.3f}",
            str(r.total_samples),
            str(r.rerun_samples),
            str(r.overrides_used),
            f"{r.acc_base:.1f}%",
            f"{r.acc_rerun:.1f}%",
            f"{r.acc_fused:.1f}%",
            str(r.flips_pos),
            str(r.flips_neg),
            str(r.flips_pos - r.flips_neg),
        )

    console.print(table)


def print_best_delta_analysis(results: List[FusionResult]) -> None:
    """Print analysis showing the best delta for each metric's peak accuracy."""
    from collections import defaultdict

    # Group results by metric
    by_metric: Dict[str, List[FusionResult]] = defaultdict(list)
    for r in results:
        by_metric[r.metric].append(r)

    console.print("\n[bold blue]Best Delta Analysis by Metric[/bold blue]")
    console.print("=" * 60)

    analysis_table = Table(box=box.SIMPLE)
    analysis_table.add_column("Metric", style="bold")
    analysis_table.add_column("Best Accuracy", justify="right")
    analysis_table.add_column("Best Delta", justify="right")
    analysis_table.add_column("Min Rerun", justify="right")
    analysis_table.add_column("Max Base", justify="right")
    analysis_table.add_column("Overrides", justify="right")
    analysis_table.add_column("Net Flips", justify="right")

    # For each metric, find the setting that achieved highest accuracy
    metric_best = []
    for metric, metric_results in by_metric.items():
        best = max(metric_results, key=lambda x: x.acc_fused)
        metric_best.append((metric, best))

        analysis_table.add_row(
            metric,
            f"{best.acc_fused:.2f}%",
            f"{best.delta:.3f}",
            "-" if best.min_rerun_conf is None else f"{best.min_rerun_conf:.3f}",
            "-" if best.max_base_conf is None else f"{best.max_base_conf:.3f}",
            str(best.overrides_used),
            str(best.flips_pos - best.flips_neg),
        )

    console.print(analysis_table)

    # Print summary statistics
    console.print("\n[bold]Summary:[/bold]")
    overall_best = max(results, key=lambda x: x.acc_fused)
    console.print(
        f"Overall best: {overall_best.metric} with delta={overall_best.delta:.3f} → {overall_best.acc_fused:.2f}%"
    )

    # Delta distribution analysis
    delta_counts = defaultdict(int)
    for _, best in metric_best:
        delta_counts[best.delta] += 1

    console.print("\nDelta preference distribution:")
    for delta in sorted(delta_counts.keys()):
        count = delta_counts[delta]
        console.print(
            f"  δ={delta:.3f}: {count} metrics ({100 * count / len(metric_best):.1f}%)"
        )

    console.print("=" * 60)


def best_accuracy() -> Tuple[str, str]:
    base_run = "5lvoti3i"
    rerun_id = "0oe2xr1b"
    return base_run, rerun_id


def convert_45() -> Tuple[str, str]:
    base_run = "53vig20u"
    rerun_id = "9qup1u07"
    return base_run, rerun_id


# Supported uncertainty metrics (aligned with figures/visualiser)
ALL_METRICS: Sequence[str] = (
    "agreement_ratio",
    "entropy_freq",
    "entropy_weighted",
    "prm_margin",
    "prm_top_frac",
    "group_top_frac",
    "prm_std",
    "prm_mean",
)


def run_ultraminimal_experiment(base_run: str, rerun_id: str) -> None:
    """Run the ultimate minimal experiment: just test which metric works best.

    All other parameters proven useless by analysis. Only 8 experiments total!

    Output path: ./output/fusion_sweeps/ultraminimal/<base>__<rerun>.json
    """
    save_dir = Path("./output/fusion_sweeps/ultraminimal")

    # Test all metrics with the proven optimal strategy
    metrics = list(ALL_METRICS)

    settings = [
        FusionSetting(
            metric=m,
            delta=0.0,  # Proven optimal
            min_rerun_conf=None,  # Proven useless
            max_base_conf=None,  # Proven useless
        )
        for m in metrics
    ]

    cfg = FusionRunConfig(
        base_run_id=base_run,
        rerun_id=rerun_id,
        subset=None,
        settings=settings,
        save_dir=save_dir,
    )

    print(f"Running ultraminimal fusion experiment: {len(settings)} settings")
    print("Strategy: Always pick the answer with higher confidence (no thresholds)")
    print(f"Testing metrics: {metrics}")

    run_sweep(cfg)


def run_minimal_experiment(base_run: str, rerun_id: str) -> None:
    """Run a compact sweep for a single metric (group_top_frac) and save JSON only.

    Output path: ./output/fusion_sweeps/minimal/<base>__<rerun>.json
    """
    save_dir = Path("./figures/fusion_sweeps/minimal")
    metrics = ["group_top_frac"]
    deltas = [0.00, 0.02, 0.05]
    min_rerun_list: List[Optional[float]] = [None, 0.50, 0.60]
    max_base_list: List[Optional[float]] = [None, 0.80]

    settings = [
        FusionSetting(metric=m, delta=d, min_rerun_conf=mr, max_base_conf=mb)
        for (m, d, mr, mb) in itertools.product(
            metrics, deltas, min_rerun_list, max_base_list
        )
    ]

    cfg = FusionRunConfig(
        base_run_id=base_run,
        rerun_id=rerun_id,
        subset=None,
        settings=settings,
        save_dir=save_dir,
    )

    run_sweep(cfg)


def run_simple_experiment(
    base_run: str,
    rerun_id: str,
    subset: Optional[Path] = None,
) -> None:
    """Run a simple sweep with delta=0 and key questions about thresholds.

    Output path: ./output/fusion_sweeps/simple/<base>__<rerun>.json
    """
    save_dir = Path("./output/fusion_sweeps/simple")

    # Focus on top metrics from previous analysis
    key_metrics = ["group_top_frac", "agreement_ratio", "prm_mean", "prm_top_frac"]

    # Delta is always 0 (proven useless)
    delta = 0.0

    # Test key threshold combinations
    min_rerun_list: List[Optional[float]] = [None, 0.50, 0.70]  # Reduced
    max_base_list: List[Optional[float]] = [None, 0.80]  # Reduced

    settings = [
        FusionSetting(metric=m, delta=delta, min_rerun_conf=mr, max_base_conf=mb)
        for (m, mr, mb) in itertools.product(key_metrics, min_rerun_list, max_base_list)
    ]

    cfg = FusionRunConfig(
        base_run_id=base_run,
        rerun_id=rerun_id,
        subset=subset,
        settings=settings,
        save_dir=save_dir,
    )

    print(f"Running simple fusion experiment: {len(settings)} settings")
    print(f"Metrics: {key_metrics}")
    print(f"Delta: {delta} (always)")
    print(f"Min rerun thresholds: {min_rerun_list}")
    print(f"Max base thresholds: {max_base_list}")

    run_sweep(cfg)


def run_extensive_experiment(
    base_run: str,
    rerun_id: str,
    subset: Optional[Path] = None,
    more_deltas: Optional[Sequence[float]] = None,
) -> None:
    """Run a broader sweep across all metrics and a richer parameter grid.


    Output path: ./output/fusion_sweeps/extensive/<base>__<rerun>.json
    """
    save_dir = Path("./output/fusion_sweeps/extensive")
    metrics = list(ALL_METRICS)
    deltas = list(more_deltas) if more_deltas is not None else [0.00, 0.01, 0.02, 0.05]
    min_rerun_list: List[Optional[float]] = [None, 0.50, 0.60, 0.70]
    max_base_list: List[Optional[float]] = [None, 0.80, 0.70]

    settings = [
        FusionSetting(metric=m, delta=d, min_rerun_conf=mr, max_base_conf=mb)
        for (m, d, mr, mb) in itertools.product(
            metrics, deltas, min_rerun_list, max_base_list
        )
    ]

    cfg = FusionRunConfig(
        base_run_id=base_run,
        rerun_id=rerun_id,
        subset=subset,
        settings=settings,
        save_dir=save_dir,
    )

    run_sweep(cfg)


if __name__ == "__main__":
    # Define your sweep here.
    # BASE_RUN, RERUN_ID = best_accuracy()
    BASE_RUN, RERUN_ID = convert_45()

    # BASE_RUN, RERUN_ID = "gfw8x07r", "8yyge5wj"
    # BASE_RUN, RERUN_ID = "77pyab58", "8ff83v7m"
    # BASE_RUN, RERUN_ID = "77pyab58", "0hermenf"

    # Ultra-minimal experiment: 8 metrics × 1 strategy = 8 settings total!
    run_minimal_experiment(BASE_RUN, RERUN_ID)

    # Uncomment for other experiments:
    # run_simple_experiment(BASE_RUN, RERUN_ID)  # 24 settings (with thresholds)
    # run_minimal_experiment(BASE_RUN, RERUN_ID)  # Single metric with parameters
    # run_extensive_experiment(BASE_RUN, RERUN_ID)  # All 384 combinations
