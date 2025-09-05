#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class Record:
    problem_id: int
    level: Optional[int]
    iteration: int
    beam_index: int
    # Signals
    draft_best_prm: float
    draft_variance: float
    draft_margin: float
    draft_slope: Optional[float]
    cross_best_delta: float  # target_best - draft_best
    overlap_jaccard: float
    best_text_match: int


@dataclass
class StepPerf:
    problem_id: int
    level: Optional[int]
    iteration: int
    generation_ms_draft: float
    generation_ms_target: float
    total_tokens_draft: int
    total_tokens_target: int
    ms_per_token_draft: Optional[float]
    ms_per_token_target: Optional[float]


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_records(
    results_path: Path,
) -> Tuple[List[Record], Dict[str, float], List[StepPerf]]:
    records: List[Record] = []
    latency_accum = {
        "templating_draft": [],
        "templating_target": [],
        "generation_draft": [],
        "generation_target": [],
        "prm_scoring_draft": [],
        "prm_scoring_target": [],
    }
    perfs: List[StepPerf] = []
    for problem in load_jsonl(results_path):
        level = problem.get("level")
        pid = int(problem.get("problem_id", -1))
        for step in problem.get("steps", []):
            it = int(step.get("iteration", -1))
            lat = step.get("latency_ms", {})
            for k in latency_accum:
                v = lat.get(k)
                if isinstance(v, (int, float)):
                    latency_accum[k].append(float(v))
            # Aggregate tokens across beams for this step
            total_tokens_draft = 0
            total_tokens_target = 0
            for state in step.get("beam_states", []):
                draft_summary = state.get("draft_summary", {})
                cross_model = state.get("cross_model", {})
                # Sum tokens across candidates (output tokens) for throughput normalisation
                dc = state.get("draft_candidates", []) or []
                tc = state.get("target_candidates", []) or []
                total_tokens_draft += sum(int(c.get("tokens", 0) or 0) for c in dc)
                total_tokens_target += sum(int(c.get("tokens", 0) or 0) for c in tc)
                rec = Record(
                    problem_id=pid,
                    level=level if isinstance(level, int) else None,
                    iteration=it,
                    beam_index=int(state.get("beam_index", -1)),
                    draft_best_prm=float(draft_summary.get("best_prm_agg", 0.0)),
                    draft_variance=float(draft_summary.get("variance", 0.0)),
                    draft_margin=float(draft_summary.get("top2_margin", 0.0)),
                    draft_slope=(
                        float(draft_summary["slope_since_prev"])
                        if draft_summary.get("slope_since_prev") is not None
                        else None
                    ),
                    cross_best_delta=float(cross_model.get("best_prm_delta", 0.0)),
                    overlap_jaccard=float(cross_model.get("overlap_jaccard", 0.0)),
                    best_text_match=int(cross_model.get("best_text_match", 0)),
                )
                records.append(rec)
            # Build per-step performance sample once per step
            gen_draft_ms = float(lat.get("generation_draft", 0.0) or 0.0)
            gen_target_ms = float(lat.get("generation_target", 0.0) or 0.0)
            mspt_draft = (
                gen_draft_ms / total_tokens_draft if total_tokens_draft > 0 else None
            )
            mspt_target = (
                gen_target_ms / total_tokens_target if total_tokens_target > 0 else None
            )
            perfs.append(
                StepPerf(
                    problem_id=pid,
                    level=level if isinstance(level, int) else None,
                    iteration=it,
                    generation_ms_draft=gen_draft_ms,
                    generation_ms_target=gen_target_ms,
                    total_tokens_draft=int(total_tokens_draft),
                    total_tokens_target=int(total_tokens_target),
                    ms_per_token_draft=mspt_draft,
                    ms_per_token_target=mspt_target,
                )
            )
    latency_means = {
        k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in latency_accum.items()
    }
    return records, latency_means, perfs


def compute_upgrade_stats(
    records: List[Record], delta_threshold: float = 0.0
) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "n_records": 0,
            "upgrade_rate": 0.0,
            "by_level": {},
            "by_iteration": {},
            "uplift_mean": 0.0,
            "uplift_median": 0.0,
        }
    uplift = np.array([r.cross_best_delta for r in records], dtype=float)
    upgrade = uplift > float(delta_threshold)
    levels = np.array(
        [r.level if r.level is not None else -1 for r in records], dtype=int
    )
    iters = np.array([r.iteration for r in records], dtype=int)

    by_level: Dict[str, float] = {}
    for L in sorted(set(levels.tolist())):
        mask = levels == L
        if mask.any():
            by_level[str(int(L))] = float(np.mean(upgrade[mask]))
    by_iter: Dict[str, float] = {}
    for iter_idx in sorted(set(iters.tolist())):
        mask = iters == iter_idx
        if mask.any():
            by_iter[str(int(iter_idx))] = float(np.mean(upgrade[mask]))
    return {
        "n_records": int(n),
        "upgrade_rate": float(np.mean(upgrade)),
        "by_level": by_level,
        "by_iteration": by_iter,
        "uplift_mean": float(np.mean(uplift)),
        "uplift_median": float(np.median(uplift)),
    }


def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    # Mann–Whitney U relation to AUC
    # AUC = (rank_sum_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    mask = ~np.isnan(scores)
    y = y_true[mask]
    s = scores[mask]
    if y.size == 0 or (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    rank_sum_pos = float(ranks[y == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def indicator_auc(
    records: List[Record], delta_threshold: float = 0.0
) -> Dict[str, float]:
    y = np.array(
        [1 if r.cross_best_delta > float(delta_threshold) else 0 for r in records],
        dtype=int,
    )
    indicators: Dict[str, np.ndarray] = {
        "neg_margin": -np.array([r.draft_margin for r in records], dtype=float),
        "variance": np.array([r.draft_variance for r in records], dtype=float),
        "neg_best_prm": -np.array([r.draft_best_prm for r in records], dtype=float),
        "neg_slope": -np.array(
            [r.draft_slope if r.draft_slope is not None else np.nan for r in records],
            dtype=float,
        ),
        "neg_overlap": -np.array([r.overlap_jaccard for r in records], dtype=float),
    }
    aucs: Dict[str, float] = {}
    for name, s in indicators.items():
        aucs[name] = auc_from_scores(y, s)
    return aucs


def plot_upgrade_rate_by_iteration(by_iter: Dict[str, float], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    xs = [int(k) for k in by_iter.keys()]
    xs.sort()
    ys = [by_iter[str(k)] for k in xs]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Upgrade rate (uplift > 0)")
    plt.title("Upgrade frequency by iteration")
    plt.grid(True, alpha=0.3)
    (out_dir / "upgrade_rate_by_iteration.png").parent.mkdir(
        parents=True, exist_ok=True
    )
    plt.tight_layout()
    plt.savefig(out_dir / "upgrade_rate_by_iteration.png", dpi=150)
    plt.close()


def plot_uplift_box_by_level(records: List[Record], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    uplift_by_level: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        L = str(r.level if r.level is not None else -1)
        uplift_by_level[L].append(r.cross_best_delta)
    labels = sorted(uplift_by_level.keys(), key=lambda x: int(x))
    data = [uplift_by_level[k] for k in labels]
    plt.figure(figsize=(8, 4))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xlabel("Level")
    plt.ylabel("Target − Draft PRM (best)")
    plt.title("Uplift distribution by difficulty level")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "uplift_box_by_level.png", dpi=150)
    plt.close()


def plot_indicator_auc_bar(aucs: Dict[str, float], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(aucs.keys())
    vals = [aucs[k] for k in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, vals)
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUC (predicting target win)")
    plt.title("Indicator separability (higher is better)")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "indicator_auc.png", dpi=150)
    plt.close()


def plot_uplift_histogram_by_iteration(
    records: List[Record], delta_threshold: float, out_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    iters = [
        r.iteration for r in records if r.cross_best_delta > float(delta_threshold)
    ]
    if len(iters) == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(iters, bins=20)
    plt.xlabel("Iteration (where uplift > threshold)")
    plt.ylabel("Count")
    plt.title("Distribution of upgrade-worthy iterations")
    plt.tight_layout()
    plt.savefig(out_dir / "uplift_histogram_iteration.png", dpi=150)
    plt.close()


def plot_latency_summary(lat_means: Dict[str, float], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    keys = [
        "templating_draft",
        "templating_target",
        "generation_draft",
        "generation_target",
        "prm_scoring_draft",
        "prm_scoring_target",
    ]
    vals = [lat_means.get(k, 0.0) for k in keys]
    plt.figure(figsize=(9, 4))
    plt.bar(keys, vals)
    plt.ylabel("Mean latency per iteration (ms)")
    plt.title("Latency components (mean across problems and iterations)")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_components.png", dpi=150)
    plt.close()


def plot_ms_per_token_hist(perfs: List[StepPerf], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    d = [p.ms_per_token_draft for p in perfs if p.ms_per_token_draft is not None]
    t = [p.ms_per_token_target for p in perfs if p.ms_per_token_target is not None]
    if len(d) == 0 and len(t) == 0:
        return
    plt.figure(figsize=(8, 4))
    if len(d) > 0:
        plt.hist(d, bins=30, alpha=0.6, label="draft")
    if len(t) > 0:
        plt.hist(t, bins=30, alpha=0.6, label="target")
    plt.xlabel("ms per output token")
    plt.ylabel("Count")
    plt.title("Throughput distribution (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ms_per_token_hist.png", dpi=150)
    plt.close()


def plot_latency_vs_tokens_scatter(perfs: List[StepPerf], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    xd = [p.total_tokens_draft for p in perfs]
    yd = [p.generation_ms_draft for p in perfs]
    xt = [p.total_tokens_target for p in perfs]
    yt = [p.generation_ms_target for p in perfs]
    plt.figure(figsize=(8, 4))
    plt.scatter(xd, yd, s=12, alpha=0.5, label="draft")
    plt.scatter(xt, yt, s=12, alpha=0.5, label="target")
    plt.xlabel("Output tokens per step (sum across beams)")
    plt.ylabel("Generation latency per step (ms)")
    plt.title("Step latency vs tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "latency_vs_tokens_scatter.png", dpi=150)
    plt.close()


def save_summary_json(summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def compute_token_stats(perfs: List[StepPerf]) -> Dict[str, Any]:
    def safe_np(arr: List[Optional[float]]) -> np.ndarray:
        vals = [float(x) for x in arr if x is not None]
        return (
            np.array(vals, dtype=float) if len(vals) > 0 else np.array([], dtype=float)
        )

    toks_d = np.array([p.total_tokens_draft for p in perfs], dtype=float)
    toks_t = np.array([p.total_tokens_target for p in perfs], dtype=float)
    mspt_d = safe_np([p.ms_per_token_draft for p in perfs])
    mspt_t = safe_np([p.ms_per_token_target for p in perfs])

    def stats(a: np.ndarray) -> Dict[str, float]:
        if a.size == 0:
            return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
        }

    return {
        "steps": int(len(perfs)),
        "total_tokens": {
            "draft_sum": float(np.sum(toks_d)) if toks_d.size > 0 else 0.0,
            "target_sum": float(np.sum(toks_t)) if toks_t.size > 0 else 0.0,
            "draft_stats_per_step": stats(toks_d),
            "target_stats_per_step": stats(toks_t),
        },
        "ms_per_token": {
            "draft_stats": stats(mspt_d),
            "target_stats": stats(mspt_t),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse diagnostic TTC results.")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run identifier (directory under output)",
    )
    parser.add_argument(
        "--base-dir", type=str, default="output", help="Base output directory"
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.0,
        help="Threshold for target uplift (PRM units)",
    )
    args = parser.parse_args()

    diag_dir = Path(args.base_dir) / args.run_id / "diagnostics"
    results_path = diag_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"results.jsonl not found at {results_path}")

    records, lat_means, perfs = collect_records(results_path)
    stats = compute_upgrade_stats(records, delta_threshold=args.delta_threshold)
    aucs = indicator_auc(records, delta_threshold=args.delta_threshold)
    token_stats = compute_token_stats(perfs)

    analysis_dir = diag_dir / "analysis"
    save_summary_json(
        {
            **stats,
            "indicator_auc": aucs,
            "latency_means": lat_means,
            "token_stats": token_stats,
        },
        analysis_dir,
    )
    plot_upgrade_rate_by_iteration(stats["by_iteration"], analysis_dir)
    plot_uplift_box_by_level(records, analysis_dir)
    plot_indicator_auc_bar(aucs, analysis_dir)
    plot_uplift_histogram_by_iteration(records, args.delta_threshold, analysis_dir)
    plot_latency_summary(lat_means, analysis_dir)
    plot_ms_per_token_hist(perfs, analysis_dir)
    plot_latency_vs_tokens_scatter(perfs, analysis_dir)

    # Print concise console summary
    print(
        json.dumps(
            {
                k: stats[k]
                for k in ["n_records", "upgrade_rate", "uplift_mean", "uplift_median"]
            },
            indent=2,
        )
    )
    print(
        "Indicator AUC (top 3):",
        sorted(
            aucs.items(), key=lambda x: (x[1] if x[1] == x[1] else -1), reverse=True
        )[:3],
    )


if __name__ == "__main__":
    main()
