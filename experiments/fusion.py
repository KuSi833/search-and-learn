from __future__ import annotations

import copy
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

    for uid in all_base_ids:
        base_rec = base_recs[uid]
        base_ok = _is_correct_record_math(base_rec)

        # Start with base result
        final_rec = base_rec
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
                final_rec = rerun_rec
                final_ok = _is_correct_record_math(rerun_rec)
                overrides_used += 1

                # Track flips only when we actually override
                if (not base_ok) and final_ok:
                    flips_pos += 1
                elif base_ok and (not final_ok):
                    flips_neg += 1

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


def best_accuracy():
    base_run = "5lvoti3i"
    rerun_id = "0oe2xr1b"
    return base_run, rerun_id


def convert_45():
    base_run = "53vig20u"
    rerun_id = "9qup1u07"
    return base_run, rerun_id


if __name__ == "__main__":
    # Define your sweep here (edit and commit as an experiment definition)
    # BASE_RUN = "5lvoti3i"  # The base run with all samples
    # BASE_RUN = "qi7dtlzw"  # weaker run
    # BASE_RUN = "51bl0yxj"  # weaker bon
    # BASE_RUN = "53vig20u"
    # RERUN_ID = "0oe2xr1b"  # The rerun with subset of samples to potentially override
    # BASE_RUN, RERUN_ID = best_accuracy()
    BASE_RUN, RERUN_ID = convert_45()
    SUBSET: Optional[Path] = None

    metrics = [
        "group_top_frac",
        # "prm_margin",
    ]
    deltas = [0.00, 0.02, 0.05]
    min_rerun_list: List[Optional[float]] = [None, 0.50, 0.60]
    max_base_list: List[Optional[float]] = [None, 0.80]

    settings = [
        FusionSetting(metric=m, delta=d, min_rerun_conf=mr, max_base_conf=mb)
        for (m, d, mr, mb) in itertools.product(
            metrics, deltas, min_rerun_list, max_base_list
        )
    ]

    save_dir = Path("./output/fusion_sweeps")
    cfg = FusionRunConfig(
        base_run_id=BASE_RUN,
        rerun_id=RERUN_ID,
        subset=SUBSET,
        settings=settings,
        save_dir=save_dir,
    )

    for rerun_id in [
        # "stlcwjg2",
        # "9qup1u07",
        RERUN_ID
    ]:
        cfg_var = copy.deepcopy(cfg)
        cfg_var.rerun_id = rerun_id
        results = run_sweep(cfg_var)
        print_results_table(results)
