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
    min_b_conf: Optional[float] = None
    max_a_conf: Optional[float] = None


@dataclass
class FusionRunConfig:
    run_a_id: str
    run_b_id: str
    subset: Optional[Path] = None
    settings: Sequence[FusionSetting] = ()
    save_dir: Optional[Path] = None


@dataclass
class FusionResult:
    metric: str
    delta: float
    min_b_conf: Optional[float]
    max_a_conf: Optional[float]
    samples: int
    chosen_from_b: int
    acc_a: float
    acc_b: float
    acc_fused: float
    flips_pos: int
    flips_neg: int


def _intersection_candidates(
    recs_a: Dict[str, Dict[str, Any]],
    recs_b: Dict[str, Dict[str, Any]],
    subset: Optional[Path],
) -> List[str]:
    if subset is None:
        return list(set(recs_a.keys()) & set(recs_b.keys()))
    ids = set(_load_subset_ids(subset))
    return [uid for uid in ids if uid in recs_a and uid in recs_b]


def _acc(records_map: Dict[str, Dict[str, Any]], ids: List[str]) -> float:
    return (
        100.0
        * sum(1 for uid in ids if _is_correct_record_math(records_map[uid]))
        / len(ids)
        if ids
        else 0.0
    )


def run_fusion_once(
    recs_a: Dict[str, Dict[str, Any]],
    recs_b: Dict[str, Dict[str, Any]],
    uids: List[str],
    setting: FusionSetting,
) -> FusionResult:
    acc_a = _acc(recs_a, uids)
    acc_b = _acc(recs_b, uids)

    fused_correct = 0
    chosen_b = 0
    flips_pos = 0
    flips_neg = 0

    for uid in uids:
        ra = recs_a[uid]
        rb = recs_b[uid]
        ca = _get_confidence(ra, setting.metric)
        cb = _get_confidence(rb, setting.metric)

        choose_b = cb > (ca + setting.delta)
        if (
            choose_b
            and (setting.min_b_conf is not None)
            and not (cb >= float(setting.min_b_conf))
        ):
            choose_b = False
        if (
            choose_b
            and (setting.max_a_conf is not None)
            and not (ca <= float(setting.max_a_conf))
        ):
            choose_b = False

        rec = rb if choose_b else ra
        if choose_b:
            chosen_b += 1

        a_ok = _is_correct_record_math(ra)
        fused_ok = _is_correct_record_math(rec)
        fused_correct += 1 if fused_ok else 0

        if (not a_ok) and fused_ok:
            flips_pos += 1
        elif a_ok and (not fused_ok):
            flips_neg += 1

    acc_fused = 100.0 * fused_correct / len(uids) if uids else 0.0

    return FusionResult(
        metric=setting.metric,
        delta=setting.delta,
        min_b_conf=setting.min_b_conf,
        max_a_conf=setting.max_a_conf,
        samples=len(uids),
        chosen_from_b=chosen_b,
        acc_a=acc_a,
        acc_b=acc_b,
        acc_fused=acc_fused,
        flips_pos=flips_pos,
        flips_neg=flips_neg,
    )


def run_sweep(cfg: FusionRunConfig) -> List[FusionResult]:
    file_a = Path("./output") / cfg.run_a_id / "inference_output.jsonl"
    file_b = Path("./output") / cfg.run_b_id / "inference_output.jsonl"
    recs_a = {rec["unique_id"]: rec for rec in load_jsonl(file_a)}
    recs_b = {rec["unique_id"]: rec for rec in load_jsonl(file_b)}

    uids = _intersection_candidates(recs_a, recs_b, cfg.subset)
    if not uids:
        raise SystemExit("No overlapping unique_ids to fuse")

    results: List[FusionResult] = []
    for setting in tqdm(cfg.settings, desc="Running Sweep"):
        res = run_fusion_once(recs_a, recs_b, uids, setting)
        results.append(res)

    if cfg.save_dir is not None:
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        out_file = cfg.save_dir / f"{cfg.run_a_id}__{cfg.run_b_id}.json"
        with out_file.open("w") as f:
            json.dump([res.__dict__ for res in results], f, indent=2)

    return results


def print_results_table(results: List[FusionResult]) -> None:
    table = Table(title="Confidence fusion sweep", box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="bold")
    table.add_column("delta", justify="right")
    table.add_column("min_b", justify="right")
    table.add_column("max_a", justify="right")
    table.add_column("samples", justify="right")
    table.add_column("chosen_B", justify="right")
    table.add_column("acc_A", justify="right")
    table.add_column("acc_B", justify="right")
    table.add_column("acc_fused", justify="right")
    table.add_column("F->T", justify="right")
    table.add_column("T->F", justify="right")
    table.add_column("net", justify="right")

    # Sort by fused accuracy descending
    for r in sorted(results, key=lambda x: x.acc_fused, reverse=True):
        table.add_row(
            r.metric,
            f"{r.delta:.3f}",
            "-" if r.min_b_conf is None else f"{r.min_b_conf:.3f}",
            "-" if r.max_a_conf is None else f"{r.max_a_conf:.3f}",
            str(r.samples),
            str(r.chosen_from_b),
            f"{r.acc_a:.1f}%",
            f"{r.acc_b:.1f}%",
            f"{r.acc_fused:.1f}%",
            str(r.flips_pos),
            str(r.flips_neg),
            str(r.flips_pos - r.flips_neg),
        )

    console.print(table)


if __name__ == "__main__":
    # Define your sweep here (edit and commit as an experiment definition)
    RUN_A = "5lvoti3i"
    RUN_B = "0oe2xr1b"
    SUBSET: Optional[Path] = None

    metrics = [
        "group_top_frac",
        "prm_margin",
    ]
    deltas = [0.00, 0.02, 0.05]
    min_b_list: List[Optional[float]] = [None, 0.50, 0.60]
    max_a_list: List[Optional[float]] = [None, 0.80]

    settings = [
        FusionSetting(metric=m, delta=d, min_b_conf=mb, max_a_conf=ma)
        for (m, d, mb, ma) in itertools.product(metrics, deltas, min_b_list, max_a_list)
    ]

    save_dir = Path("./output/fusion_sweeps")
    cfg = FusionRunConfig(
        run_a_id=RUN_A,
        run_b_id=RUN_B,
        subset=SUBSET,
        settings=settings,
        save_dir=save_dir,
    )

    results = run_sweep(cfg)
    print_results_table(results)
