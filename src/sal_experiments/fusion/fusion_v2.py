import math as _m
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy import stats
from tqdm import tqdm

from .fusion import (
    ALL_METRICS,
    _acc,
    _get_confidence,
    _is_correct_record_math,
    load_jsonl,
)


@dataclass
class FusionRerun:
    base_run_id: str
    rerun_run_id: str
    coverage: int


# List of FusionRerun objects based on your data
fusion_reruns = [
    FusionRerun(base_run_id="77pyab58", rerun_run_id="0hermenf", coverage=20),
    FusionRerun(base_run_id="gfw8x07r", rerun_run_id="58xqqffr", coverage=20),
    FusionRerun(base_run_id="tqfyvf5w", rerun_run_id="6nuelycf", coverage=20),
    FusionRerun(base_run_id="77pyab58", rerun_run_id="defwqu3v", coverage=20),
    FusionRerun(base_run_id="tqfyvf5w", rerun_run_id="dtjaqwyg", coverage=20),
    FusionRerun(base_run_id="tqfyvf5w", rerun_run_id="whldvunb", coverage=20),
    FusionRerun(base_run_id="tqfyvf5w", rerun_run_id="22z91qa7", coverage=10),
    FusionRerun(base_run_id="tqfyvf5w", rerun_run_id="2jcdgx7c", coverage=10),
    FusionRerun(base_run_id="77pyab58", rerun_run_id="8ff83v7m", coverage=10),
    FusionRerun(base_run_id="gfw8x07r", rerun_run_id="8yyge5wj", coverage=10),
    FusionRerun(base_run_id="tqfyvf5w", rerun_run_id="ch6s6c0c", coverage=10),
    FusionRerun(base_run_id="77pyab58", rerun_run_id="ylsdosex", coverage=10),
]


# --- Helper utilities ---

OUTPUT_DIR = Path("./output")
EXPECTED_FILENAME = "inference_output.jsonl"

console = Console()


def _expected_output_path(run_id: str) -> Path:
    return OUTPUT_DIR / run_id / EXPECTED_FILENAME


def _unique_run_ids(pairs: Iterable[FusionRerun]) -> Set[str]:
    ids: Set[str] = set()
    for pr in pairs:
        ids.add(pr.base_run_id)
        ids.add(pr.rerun_run_id)
    return ids


def check_all_runs_exist(pairs: Iterable[FusionRerun]) -> None:
    """Verify required output files exist for all run IDs.

    Prints any missing run IDs with the expected file path. Exits with code 1 if any
    files are missing. Does nothing (and prints a short confirmation) if all exist.
    """
    unique_ids = _unique_run_ids(pairs)

    missing: Dict[str, Path] = {}
    for run_id in unique_ids:
        path = _expected_output_path(run_id)
        if not path.exists():
            missing[run_id] = path

    # Pretty status table for all runs
    table = Table(title="Required run outputs", box=box.SIMPLE_HEAVY)
    table.add_column("run_id", style="bold")
    table.add_column("expected file", overflow="fold")
    table.add_column("status", justify="center")

    for run_id in sorted(unique_ids):
        path = _expected_output_path(run_id)
        exists = path.exists()
        status = "[green]OK[/green]" if exists else "[red]MISSING[/red]"
        table.add_row(run_id, str(path), status)

    if missing:
        console.print(Panel.fit("Missing run outputs detected", border_style="red"))
        console.print(table)
        sys.exit(1)

    console.print(
        Panel.fit(
            f"All runs present ({len(unique_ids)} unique run IDs)", border_style="green"
        )
    )


def analyze_pair(pair: FusionRerun) -> Dict[str, Any]:
    base_file = OUTPUT_DIR / pair.base_run_id / EXPECTED_FILENAME
    rerun_file = OUTPUT_DIR / pair.rerun_run_id / EXPECTED_FILENAME
    base_recs = {rec["unique_id"]: rec for rec in load_jsonl(base_file)}
    rerun_recs = {rec["unique_id"]: rec for rec in load_jsonl(rerun_file)}

    all_base_ids: List[str] = list(base_recs.keys())
    overlap_ids: List[str] = [uid for uid in all_base_ids if uid in rerun_recs]

    acc_base = _acc(base_recs, all_base_ids)

    acc_rerun = _acc(rerun_recs, overlap_ids) if overlap_ids else 0.0

    conversions: Dict[str, int] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
    for uid in overlap_ids:
        b_ok = _is_correct_record_math(base_recs[uid])
        r_ok = _is_correct_record_math(rerun_recs[uid])
        if b_ok and r_ok:
            conversions["TT"] += 1
        elif (not b_ok) and (not r_ok):
            conversions["FF"] += 1
        elif b_ok and (not r_ok):
            conversions["TF"] += 1
        elif (not b_ok) and r_ok:
            conversions["FT"] += 1

    oracle_net_gain = conversions["FT"]
    # Always override baseline (override wherever rerun exists)
    always_net = conversions["FT"] - conversions["TF"]
    acc_fused_always = acc_base + (
        always_net * 100.0 / len(all_base_ids) if all_base_ids else 0.0
    )
    always_gain_pp = acc_fused_always - acc_base

    metrics = (
        "agreement_ratio",
        "entropy_freq",
        "entropy_weighted",
        "prm_margin",
        "prm_top_frac",
        "consensus_support",
        "prm_std",
        "prm_mean",
    )

    results_rows: List[Dict[str, Any]] = []
    for metric in tqdm(metrics, desc="Scanning metrics", leave=False):
        per: List[tuple] = []
        for uid in overlap_ids:
            b = base_recs[uid]
            r = rerun_recs[uid]
            b_ok = _is_correct_record_math(b)
            r_ok = _is_correct_record_math(r)
            b_conf = _get_confidence(b, metric)
            r_conf = _get_confidence(r, metric)
            delta = r_conf - b_conf
            if (not b_ok) and r_ok:
                label = 1
                conv = "FT"
            elif b_ok and (not r_ok):
                label = -1
                conv = "TF"
            elif b_ok and r_ok:
                label = 0
                conv = "TT"
            else:
                label = 0
                conv = "FF"
            per.append((delta, label, conv))

        per.sort(key=lambda x: x[0], reverse=True)
        best_net = 0
        best_k = 0
        ft = tf = tt = ff = 0
        cur_net = 0
        cur_ft = cur_tf = cur_tt = cur_ff = 0
        for k, (_, label, conv) in enumerate(per, start=1):
            cur_net += label
            if conv == "FT":
                cur_ft += 1
            elif conv == "TF":
                cur_tf += 1
            elif conv == "TT":
                cur_tt += 1
            else:
                cur_ff += 1
            if cur_net > best_net:
                best_net = cur_net
                best_k = k
                ft, tf, tt, ff = cur_ft, cur_tf, cur_tt, cur_ff

        acc_fused_best = acc_base + (
            best_net * 100.0 / len(all_base_ids) if all_base_ids else 0.0
        )
        gain_pp = acc_fused_best - acc_base
        results_rows.append(
            {
                "metric": metric,
                "best_net": int(best_net),
                "overrides": int(best_k),
                "FT": int(ft),
                "TF": int(tf),
                "TT": int(tt),
                "FF": int(ff),
                "acc_base": float(acc_base),
                "acc_fused_best": float(acc_fused_best),
                "gain_pp": float(gain_pp),
            }
        )

    # Choose overall best metric by net gain
    best_row = max(results_rows, key=lambda x: x["best_net"]) if results_rows else None

    return {
        "pair": pair,
        "all_base_ids": all_base_ids,
        "overlap_ids": overlap_ids,
        "acc_base": acc_base,
        "acc_rerun": acc_rerun,
        "conversions": conversions,
        "oracle_net_gain": oracle_net_gain,
        "results_rows": results_rows,
        "best_row": best_row,
        "always_net": int(always_net),
        "acc_fused_always": float(acc_fused_always),
        "always_gain_pp": float(always_gain_pp),
    }


def render_pair_analysis(result: Dict[str, Any]) -> None:
    pair: FusionRerun = result["pair"]
    all_base_ids: List[str] = result["all_base_ids"]
    overlap_ids: List[str] = result["overlap_ids"]
    acc_base: float = result["acc_base"]
    acc_rerun: float = result["acc_rerun"]
    conversions: Dict[str, int] = result["conversions"]
    oracle_net_gain: int = result["oracle_net_gain"]
    results_rows: List[Dict[str, Any]] = result["results_rows"]
    always_net: int = result.get("always_net", 0)
    acc_fused_always: float = result.get("acc_fused_always", acc_base)
    always_gain_pp: float = result.get("always_gain_pp", 0.0)

    console.print(
        Panel.fit(
            f"Analyzing pair base={pair.base_run_id} rerun={pair.rerun_run_id}",
            border_style="cyan",
        )
    )

    summary = Table(title="Pair overview", box=box.SIMPLE_HEAVY)
    summary.add_column("field", style="bold")
    summary.add_column("value", justify="right")
    summary.add_row("total base samples", str(len(all_base_ids)))
    summary.add_row("rerun overlap", str(len(overlap_ids)))
    summary.add_row("base acc", f"{acc_base:.2f}%")
    summary.add_row("rerun acc (overlap)", f"{acc_rerun:.2f}%")
    summary.add_row("oracle max net gain", str(oracle_net_gain))
    ag = always_gain_pp
    ag_str = (
        f"[green]+{ag:.2f} pp[/green]"
        if ag > 0
        else (f"[red]{ag:.2f} pp[/red]" if ag < 0 else f"{ag:.2f} pp")
    )
    summary.add_row("always override net", str(always_net))
    summary.add_row("always override fused acc", f"{acc_fused_always:.2f}%")
    summary.add_row("always override gain", ag_str)
    console.print(summary)

    conv_table = Table(
        title="Conversions if always overriding where possible", box=box.SIMPLE_HEAVY
    )
    for col in ("TT", "FF", "TF", "FT"):
        conv_table.add_column(col, justify="right")
    conv_table.add_row(
        str(conversions["TT"]),
        str(conversions["FF"]),
        str(conversions["TF"]),
        str(conversions["FT"]),
    )
    console.print(conv_table)

    res_table = Table(
        title="Best separator per metric (delta = rerun - base)", box=box.SIMPLE_HEAVY
    )
    for col, j in [
        ("metric", None),
        ("best_net", "right"),
        ("overrides", "right"),
        ("FT", "right"),
        ("TF", "right"),
        ("TT", "right"),
        ("FF", "right"),
        ("acc_base", "right"),
        ("acc_fused_best", "right"),
        ("gain_pp", "right"),
    ]:
        res_table.add_column(col, justify=(j or "left"))

    for row in sorted(results_rows, key=lambda x: x["best_net"], reverse=True):
        gain = row["gain_pp"]
        gain_str = (
            f"[green]+{gain:.2f}[/green]"
            if gain > 0
            else (f"[red]{gain:.2f}[/red]" if gain < 0 else f"{gain:.2f}")
        )
        res_table.add_row(
            row["metric"],
            str(row["best_net"]),
            str(row["overrides"]),
            str(row["FT"]),
            str(row["TF"]),
            str(row["TT"]),
            str(row["FF"]),
            f"{row['acc_base']:.2f}%",
            f"{row['acc_fused_best']:.2f}%",
            gain_str,
        )
    console.print(res_table)

    # Highlight overall best fused accuracy and gain
    best_row = result.get("best_row")
    if best_row:
        fused = best_row["acc_fused_best"]
        gain = best_row["gain_pp"]
        console.print(
            Panel.fit(
                f"Best metric: [bold]{best_row['metric']}[/bold]  Fused acc: [bold]{fused:.2f}%[/bold]  Gain vs base: "
                + (
                    f"[green]+{gain:.2f} pp[/green]"
                    if gain > 0
                    else (f"[red]{gain:.2f} pp[/red]" if gain < 0 else f"{gain:.2f} pp")
                ),
                border_style="magenta",
            )
        )


def run_single_pair_analysis(pair_index: int = 0) -> None:
    pair = fusion_reruns[pair_index]
    result = analyze_pair(pair)
    render_pair_analysis(result)


# ----------------- Optional: Linear classifier-based separator (analysis only) -----------------


def _build_features(
    base_rec: Dict[str, Any], rerun_rec: Dict[str, Any], metrics: Iterable[str]
) -> List[float]:
    feats: List[float] = []
    for m in metrics:
        b = _get_confidence(base_rec, m)
        r = _get_confidence(rerun_rec, m)
        feats.extend([b, r, r - b])
    return feats


def analyze_pair_with_linear_classifier(pair: FusionRerun) -> Dict[str, Any]:
    if np is None:
        console.print(
            Panel.fit(
                "NumPy not available; skipping classifier analysis",
                border_style="yellow",
            )
        )
        return {}

    base_file = OUTPUT_DIR / pair.base_run_id / EXPECTED_FILENAME
    rerun_file = OUTPUT_DIR / pair.rerun_run_id / EXPECTED_FILENAME
    base_recs = {rec["unique_id"]: rec for rec in load_jsonl(base_file)}
    rerun_recs = {rec["unique_id"]: rec for rec in load_jsonl(rerun_file)}

    all_base_ids: List[str] = list(base_recs.keys())
    overlap_ids: List[str] = [uid for uid in all_base_ids if uid in rerun_recs]

    # Build dataset: only examples where base and rerun disagree (FT or TF)
    X_rows: List[List[float]] = []
    y_vals: List[int] = []
    metrics = list(ALL_METRICS)
    for uid in overlap_ids:
        b = base_recs[uid]
        r = rerun_recs[uid]
        b_ok = _is_correct_record_math(b)
        r_ok = _is_correct_record_math(r)
        if b_ok == r_ok:
            continue  # skip TT/FF for training
        y_vals.append(1 if ((not b_ok) and r_ok) else -1)  # +1 for FT, -1 for TF
        X_rows.append(_build_features(b, r, metrics))

    if len(set(y_vals)) < 2 or not X_rows:
        console.print(
            Panel.fit(
                "Not enough disagreement cases for classifier training",
                border_style="yellow",
            )
        )
        return {}

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)

    # Ridge regression solution (acts like linear classifier for ranking)
    reg = 1e-3
    XtX = X.T @ X
    w = np.linalg.solve(XtX + reg * np.eye(X.shape[1]), X.T @ y)

    # Score all overlap items (including TT/FF) and compute best prefix net gain
    scored: List[tuple] = []
    for uid in overlap_ids:
        b = base_recs[uid]
        r = rerun_recs[uid]
        feats = np.array(_build_features(b, r, metrics), dtype=float)
        score = float(feats @ w)
        b_ok = _is_correct_record_math(b)
        r_ok = _is_correct_record_math(r)
        if (not b_ok) and r_ok:
            label = 1
            conv = "FT"
        elif b_ok and (not r_ok):
            label = -1
            conv = "TF"
        elif b_ok and r_ok:
            label = 0
            conv = "TT"
        else:
            label = 0
            conv = "FF"
        scored.append((score, label, conv))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_net = 0
    best_k = 0
    ft = tf = tt = ff = 0
    cur_net = 0
    cur_ft = cur_tf = cur_tt = cur_ff = 0
    for k, (_, label, conv) in enumerate(scored, start=1):
        cur_net += label
        if conv == "FT":
            cur_ft += 1
        elif conv == "TF":
            cur_tf += 1
        elif conv == "TT":
            cur_tt += 1
        else:
            cur_ff += 1
        if cur_net > best_net:
            best_net = cur_net
            best_k = k
            ft, tf, tt, ff = cur_ft, cur_tf, cur_tt, cur_ff

    acc_base = _acc(base_recs, all_base_ids)
    acc_fused_best = acc_base + (
        best_net * 100.0 / len(all_base_ids) if all_base_ids else 0.0
    )

    # Prepare readable weights
    weight_rows: List[Dict[str, Any]] = []
    names: List[str] = []
    for m in metrics:
        names.extend([f"base_{m}", f"rerun_{m}", f"delta_{m}"])
    for name, weight in sorted(
        zip(names, w.tolist()), key=lambda x: abs(x[1]), reverse=True
    ):
        weight_rows.append({"feature": name, "weight": float(weight)})

    return {
        "pair": pair,
        "acc_base": float(acc_base),
        "acc_fused_best": float(acc_fused_best),
        "best_net": int(best_net),
        "overrides": int(best_k),
        "FT": int(ft),
        "TF": int(tf),
        "TT": int(tt),
        "FF": int(ff),
        "weights": weight_rows,
    }


def render_classifier_analysis(result: Dict[str, Any]) -> None:
    if not result:
        return
    pair: FusionRerun = result["pair"]
    acc_base: float = result["acc_base"]
    acc_fused: float = result["acc_fused_best"]
    gain = acc_fused - acc_base

    console.print(
        Panel.fit(
            f"Linear classifier analysis for base={pair.base_run_id} rerun={pair.rerun_run_id}",
            border_style="cyan",
        )
    )

    summary = Table(title="Classifier summary", box=box.SIMPLE_HEAVY)
    summary.add_column("field", style="bold")
    summary.add_column("value", justify="right")
    summary.add_row("best net gain", str(result["best_net"]))
    summary.add_row("overrides used", str(result["overrides"]))
    summary.add_row("FT", str(result["FT"]))
    summary.add_row("TF", str(result["TF"]))
    summary.add_row("TT", str(result["TT"]))
    summary.add_row("FF", str(result["FF"]))
    summary.add_row("base acc", f"{acc_base:.2f}%")
    summary.add_row("fused acc", f"{acc_fused:.2f}%")
    gain_str = (
        f"[green]+{gain:.2f} pp[/green]"
        if gain > 0
        else (f"[red]{gain:.2f} pp[/red]" if gain < 0 else f"{gain:.2f} pp")
    )
    summary.add_row("gain vs base", gain_str)
    console.print(summary)

    wt = Table(title="Top feature weights (|weight| desc)", box=box.SIMPLE_HEAVY)
    wt.add_column("feature")
    wt.add_column("weight", justify="right")
    for row in result["weights"][:20]:
        wv = row["weight"]
        wv_str = f"[green]{wv:.4f}[/green]" if wv > 0 else f"[red]{wv:.4f}[/red]"
        wt.add_row(row["feature"], wv_str)
    console.print(wt)


def run_classifier_pair_analysis(pair_index: int = 0) -> None:
    pair = fusion_reruns[pair_index]
    res = analyze_pair_with_linear_classifier(pair)
    render_classifier_analysis(res)


# ----------------- Greedy small-feature linear separator with CV (explainable) -----------------


def analyze_pair_with_greedy_delta_cv(
    pair: FusionRerun, max_features: int = 3, kfolds: int = 5, reg: float = 1e-3
) -> Dict[str, Any]:
    if np is None:
        console.print(
            Panel.fit(
                "NumPy not available; skipping greedy CV analysis",
                border_style="yellow",
            )
        )
        return {}

    base_file = OUTPUT_DIR / pair.base_run_id / EXPECTED_FILENAME
    rerun_file = OUTPUT_DIR / pair.rerun_run_id / EXPECTED_FILENAME
    base_recs = {rec["unique_id"]: rec for rec in load_jsonl(base_file)}
    rerun_recs = {rec["unique_id"]: rec for rec in load_jsonl(rerun_file)}

    all_base_ids: List[str] = list(base_recs.keys())
    overlap_ids: List[str] = [uid for uid in all_base_ids if uid in rerun_recs]

    metrics = list(ALL_METRICS)

    # Build per-id labels (FT=+1, TF=-1, else 0) and delta feature map per metric
    labels: List[int] = []
    per_metric_delta: Dict[str, List[float]] = {m: [] for m in metrics}
    for uid in overlap_ids:
        b = base_recs[uid]
        r = rerun_recs[uid]
        b_ok = _is_correct_record_math(b)
        r_ok = _is_correct_record_math(r)
        if (not b_ok) and r_ok:
            y = 1
        elif b_ok and (not r_ok):
            y = -1
        else:
            y = 0
        labels.append(y)
        for m in metrics:
            per_metric_delta[m].append(_get_confidence(r, m) - _get_confidence(b, m))

    y_all = np.array(labels, dtype=float)
    idx_disagree = np.where(np.abs(y_all) == 1)[0]
    if idx_disagree.size < 4:
        console.print(
            Panel.fit(
                "Not enough FT/TF cases for CV analysis",
                border_style="yellow",
            )
        )
        return {}

    # Helper: evaluate feature set by K-fold CV on disagreements
    seed = 42
    rng = np.random.default_rng(seed)

    def cv_score(feature_set: List[str]) -> Dict[str, Any]:
        if not feature_set:
            return {"mean": 0.0, "std": 0.0}
        X_all = np.stack([per_metric_delta[m] for m in feature_set], axis=1)
        # Use only disagreements for training/validation
        Xd = X_all[idx_disagree]
        yd = y_all[idx_disagree]

        # Build folds
        idx = np.arange(Xd.shape[0])
        rng.shuffle(idx)
        folds = np.array_split(idx, kfolds)
        nets: List[int] = []
        for f in folds:
            if f.size == 0:
                continue
            mask = np.ones_like(idx, dtype=bool)
            mask[np.isin(idx, f)] = False
            tr = idx[mask]
            va = f
            Xtr = Xd[tr]
            ytr = yd[tr]
            Xva = Xd[va]
            yva = yd[va]
            # Closed-form ridge
            XtX = Xtr.T @ Xtr
            w = np.linalg.solve(XtX + reg * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
            # Evaluate on validation via best-prefix net gain
            scores = Xva @ w
            order = np.argsort(-scores)
            cur = 0
            best = 0
            for j in order:
                cur += int(yva[j])
                if cur > best:
                    best = cur
            nets.append(int(best))
        if not nets:
            return {"mean": 0.0, "std": 0.0}
        return {"mean": float(np.mean(nets)), "std": float(np.std(nets))}

    # Greedy forward selection on delta features
    selected: List[str] = []
    history: List[Dict[str, Any]] = []
    remaining = set(metrics)
    for _ in range(max_features):
        best_m = None
        best_stats = {"mean": -1e9, "std": 0.0}
        for m in sorted(remaining):
            stats = cv_score(selected + [m])
            if stats["mean"] > best_stats["mean"]:
                best_stats = stats
                best_m = m
        if best_m is None:
            break
        selected.append(best_m)
        remaining.remove(best_m)
        history.append({"features": list(selected), **best_stats})

    # Train final model on all disagreements with selected features
    if not selected:
        return {}
    X_all = np.stack([per_metric_delta[m] for m in selected], axis=1)
    Xd = X_all[idx_disagree]
    yd = y_all[idx_disagree]
    XtX = Xd.T @ Xd
    w = np.linalg.solve(XtX + reg * np.eye(Xd.shape[1]), Xd.T @ yd)

    # Score full overlap and compute best prefix net and fused acc
    scores_full = X_all @ w
    order_full = np.argsort(-scores_full)
    labels_full = y_all
    cur = 0
    best = 0
    best_k = 0
    ft = tf = tt = ff = 0
    cur_ft = cur_tf = cur_tt = cur_ff = 0
    for k, j in enumerate(order_full, start=1):
        cur += int(labels_full[j])
        # conv tallies for interpretability
        if labels_full[j] == 1:
            cur_ft += 1
        elif labels_full[j] == -1:
            cur_tf += 1
        else:
            # we don't know TT vs FF here directly without re-checking; recompute
            b = base_recs[overlap_ids[j]]
            r = rerun_recs[overlap_ids[j]]
            if _is_correct_record_math(b) and _is_correct_record_math(r):
                cur_tt += 1
            elif (not _is_correct_record_math(b)) and (not _is_correct_record_math(r)):
                cur_ff += 1
        if cur > best:
            best = cur
            best_k = k
            ft, tf, tt, ff = cur_ft, cur_tf, cur_tt, cur_ff

    acc_base = _acc(base_recs, all_base_ids)
    acc_fused_best = acc_base + (
        best * 100.0 / len(all_base_ids) if all_base_ids else 0.0
    )

    # Compute reproducible threshold t such that selecting score >= t yields exactly 'best_k' overrides
    sorted_scores = scores_full[order_full]
    if best_k == 0:
        threshold = float("inf")
    else:
        s_k = float(sorted_scores[best_k - 1])
        s_next = (
            float(sorted_scores[best_k]) if best_k < len(sorted_scores) else (s_k - 1.0)
        )
        threshold = (s_k + s_next) / 2.0

    # Package weights mapped to feature names
    weight_rows: List[Dict[str, Any]] = []
    for name, weight in zip(selected, w.tolist()):
        weight_rows.append({"feature": f"delta_{name}", "weight": float(weight)})

    return {
        "pair": pair,
        "selected": selected,
        "cv_history": history,
        "weights": weight_rows,
        "threshold": float(threshold),
        "kfolds": int(kfolds),
        "reg": float(reg),
        "seed": int(seed),
        "best_net": int(best),
        "overrides": int(best_k),
        "FT": int(ft),
        "TF": int(tf),
        "TT": int(tt),
        "FF": int(ff),
        "acc_base": float(acc_base),
        "acc_fused_best": float(acc_fused_best),
    }


def render_greedy_cv_analysis(result: Dict[str, Any]) -> None:
    if not result:
        return
    pair: FusionRerun = result["pair"]
    console.print(
        Panel.fit(
            f"Greedy delta-CV analysis for base={pair.base_run_id} rerun={pair.rerun_run_id}",
            border_style="cyan",
        )
    )

    hist = result["cv_history"]
    wt = result["weights"]
    acc_base = result["acc_base"]
    acc_fused = result["acc_fused_best"]
    gain = acc_fused - acc_base

    s = Table(title="Selected features (greedy CV)", box=box.SIMPLE_HEAVY)
    s.add_column("step", justify="right")
    s.add_column("features")
    s.add_column("cv_best_net_mean", justify="right")
    s.add_column("cv_best_net_std", justify="right")
    for i, h in enumerate(hist, start=1):
        s.add_row(
            str(i), ", ".join(h["features"]), f"{h['mean']:.2f}", f"{h['std']:.2f}"
        )
    console.print(s)

    wtab = Table(title="Final weights (delta features)", box=box.SIMPLE_HEAVY)
    wtab.add_column("feature")
    wtab.add_column("weight", justify="right")
    for row in wt:
        wv = row["weight"]
        wv_str = f"[green]{wv:.4f}[/green]" if wv > 0 else f"[red]{wv:.4f}[/red]"
        wtab.add_row(row["feature"], wv_str)
    console.print(wtab)

    # Decision rule and training params
    threshold = result.get("threshold", 0.0)
    params = Table(title="Decision rule and params", box=box.SIMPLE_HEAVY)
    params.add_column("field", style="bold")
    params.add_column("value")
    params.add_row(
        "decision rule", "score = sum_i w_i * delta(metric_i); select if score >= t"
    )
    params.add_row("threshold t", f"{threshold:.6f}")
    params.add_row("kfolds", str(result.get("kfolds", "-")))
    params.add_row("reg", f"{float(result.get('reg', 0.0)):.6g}")
    params.add_row("seed", str(result.get("seed", "-")))
    console.print(params)

    summary = Table(title="Greedy CV summary", box=box.SIMPLE_HEAVY)
    summary.add_column("field", style="bold")
    summary.add_column("value", justify="right")
    summary.add_row("best net gain", str(result["best_net"]))
    summary.add_row("overrides used", str(result["overrides"]))
    summary.add_row("FT", str(result["FT"]))
    summary.add_row("TF", str(result["TF"]))
    summary.add_row("TT", str(result["TT"]))
    summary.add_row("FF", str(result["FF"]))
    summary.add_row("base acc", f"{acc_base:.2f}%")
    summary.add_row("fused acc", f"{acc_fused:.2f}%")
    gain_str = (
        f"[green]+{gain:.2f} pp[/green]"
        if gain > 0
        else (f"[red]{gain:.2f} pp[/red]" if gain < 0 else f"{gain:.2f} pp")
    )
    summary.add_row("gain vs base", gain_str)
    console.print(summary)


def run_greedy_cv_pair_analysis(pair_index: int = 0, max_features: int = 3) -> None:
    pair = fusion_reruns[pair_index]
    res = analyze_pair_with_greedy_delta_cv(pair, max_features=max_features)
    render_greedy_cv_analysis(res)


# ----------------- Multiprocessing: run greedy CV on all pairs -----------------


def _greedy_cv_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    pair_index: int = int(args["pair_index"])  # type: ignore
    max_features: int = int(args.get("max_features", 3))  # type: ignore
    pair = fusion_reruns[pair_index]
    try:
        res = analyze_pair_with_greedy_delta_cv(pair, max_features=max_features)
        if not res:
            return {
                "pair_index": pair_index,
                "base": pair.base_run_id,
                "rerun": pair.rerun_run_id,
                "ok": False,
                "error": "insufficient data",
            }
        return {
            "pair_index": pair_index,
            "base": pair.base_run_id,
            "rerun": pair.rerun_run_id,
            "ok": True,
            "selected": res.get("selected", []),
            "weights": res.get("weights", []),
            "threshold": res.get("threshold", 0.0),
            "best_net": res.get("best_net", 0),
            "overrides": res.get("overrides", 0),
            "acc_base": res.get("acc_base", 0.0),
            "acc_fused": res.get("acc_fused_best", 0.0),
        }
    except Exception as e:
        return {
            "pair_index": pair_index,
            "base": pair.base_run_id,
            "rerun": pair.rerun_run_id,
            "ok": False,
            "error": str(e),
        }


def run_greedy_cv_all_pairs(
    max_features: int = 3, processes: int | None = None
) -> None:
    tasks = [
        {"pair_index": i, "max_features": max_features}
        for i in range(len(fusion_reruns))
    ]
    results: List[Dict[str, Any]] = []

    console.print(
        Panel.fit(f"Running greedy CV on {len(tasks)} pairs", border_style="cyan")
    )
    with ProcessPoolExecutor(max_workers=processes) as ex:
        futures = [ex.submit(_greedy_cv_worker, t) for t in tasks]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Pairs", leave=False
        ):
            results.append(fut.result())

    # Aggregate
    ok_results = [r for r in results if r.get("ok")]
    if not ok_results:
        console.print(Panel.fit("No successful results", border_style="red"))
        return

    # Per-pair table
    per = Table(title="Greedy CV per pair", box=box.SIMPLE_HEAVY)
    per.add_column("#", justify="right")
    per.add_column("base→rerun")
    per.add_column("selected")
    per.add_column("rule")
    per.add_column("best_net", justify="right")
    per.add_column("overrides", justify="right")
    per.add_column("acc_base", justify="right")
    per.add_column("acc_fused", justify="right")
    per.add_column("gain_pp", justify="right")

    feature_counts: Dict[str, int] = {}
    gains: List[float] = []
    for r in sorted(
        ok_results,
        key=lambda x: (x.get("acc_fused", 0.0) - x.get("acc_base", 0.0)),
        reverse=True,
    ):
        sels: List[str] = r.get("selected", [])  # type: ignore
        for m in sels:
            feature_counts[m] = feature_counts.get(m, 0) + 1
        gain = float(r.get("acc_fused", 0.0) - r.get("acc_base", 0.0))
        gains.append(gain)
        gain_str = (
            f"[green]+{gain:.2f}[/green]"
            if gain > 0
            else (f"[red]{gain:.2f}[/red]" if gain < 0 else f"{gain:.2f}")
        )
        # Build compact decision rule text from weights and threshold
        weights_list: List[Dict[str, Any]] = r.get("weights", [])  # type: ignore
        thr = float(r.get("threshold", 0.0))
        if weights_list:
            terms = []
            for wrow in weights_list[:3]:
                name = str(wrow.get("feature", "delta_m")).replace("delta_", "Δ")
                wv = float(wrow.get("weight", 0.0))
                terms.append(f"{wv:.3f}*{name}")
            more = " + …" if len(weights_list) > 3 else ""
            rule_text = f"{' + '.join(terms)}{more} ≥ {thr:.6f}"
        else:
            rule_text = f"linear(Δ{len(sels)}) ≥ {thr:.6f}"

        per.add_row(
            str(r.get("pair_index")),
            f"{r.get('base')}→{r.get('rerun')}",
            ", ".join(sels),
            rule_text,
            str(r.get("best_net")),
            str(r.get("overrides")),
            f"{float(r.get('acc_base', 0.0)):.2f}%",
            f"{float(r.get('acc_fused', 0.0)):.2f}%",
            gain_str,
        )
    console.print(per)

    # Feature frequency table
    freq = Table(
        title="Selected delta-metric frequency across pairs", box=box.SIMPLE_HEAVY
    )
    freq.add_column("metric")
    freq.add_column("count", justify="right")
    for m, c in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        freq.add_row(m, str(c))
    console.print(freq)

    # Summary
    import math as _m

    mean_gain = sum(gains) / len(gains)
    std_gain = (
        _m.sqrt(sum((g - mean_gain) ** 2 for g in gains) / len(gains)) if gains else 0.0
    )
    summ = Table(title="Across-pair summary", box=box.SIMPLE_HEAVY)
    summ.add_column("field", style="bold")
    summ.add_column("value", justify="right")
    summ.add_row("pairs succeeded", f"{len(ok_results)}/{len(results)}")
    summ.add_row("mean gain (pp)", f"{mean_gain:.2f}")
    summ.add_row("std gain (pp)", f"{std_gain:.2f}")
    console.print(summ)


# ----------------- Cross-pair evaluation: train once, test across pairs -----------------


def _load_pair_overlap(pair: FusionRerun) -> Dict[str, Any]:
    base_file = OUTPUT_DIR / pair.base_run_id / EXPECTED_FILENAME
    rerun_file = OUTPUT_DIR / pair.rerun_run_id / EXPECTED_FILENAME
    base_recs = {rec["unique_id"]: rec for rec in load_jsonl(base_file)}
    rerun_recs = {rec["unique_id"]: rec for rec in load_jsonl(rerun_file)}
    all_base_ids: List[str] = list(base_recs.keys())
    overlap_ids: List[str] = [uid for uid in all_base_ids if uid in rerun_recs]
    acc_base = _acc(base_recs, all_base_ids)
    # Build labels and metric deltas per item
    items: List[Dict[str, Any]] = []
    for uid in overlap_ids:
        b = base_recs[uid]
        r = rerun_recs[uid]
        b_ok = _is_correct_record_math(b)
        r_ok = _is_correct_record_math(r)
        if (not b_ok) and r_ok:
            label = 1
        elif b_ok and (not r_ok):
            label = -1
        else:
            label = 0
        items.append({"uid": uid, "b": b, "r": r, "label": label})
    return {
        "pair": pair,
        "base_recs": base_recs,
        "rerun_recs": rerun_recs,
        "all_base_ids": all_base_ids,
        "overlap_ids": overlap_ids,
        "acc_base": acc_base,
        "items": items,
    }


def _metric_scores(items: List[Dict[str, Any]], metric: str) -> List[tuple]:
    out: List[tuple] = []
    for it in items:
        b = it["b"]
        r = it["r"]
        score = _get_confidence(r, metric) - _get_confidence(b, metric)
        out.append((float(score), int(it["label"])))
    return out


def _choose_threshold(scores_labels: List[tuple]) -> float:
    # Return threshold t such that selecting score >= t maximizes summed label
    if not scores_labels:
        return float("inf")
    sl = sorted(scores_labels, key=lambda x: x[0], reverse=True)
    cur = 0
    best = 0
    best_k = 0
    for k, (_, lab) in enumerate(sl, start=1):
        cur += int(lab)
        if cur > best:
            best = cur
            best_k = k
    if best_k == 0:
        return float("inf")  # select none
    # place threshold between kth and (k+1)th score
    s_k = sl[best_k - 1][0]
    s_next = sl[best_k][0] if best_k < len(sl) else (s_k - 1.0)
    return (s_k + s_next) / 2.0


def _metric_best_per_pair_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker for per-pair best separator to allow pickling under spawn.

    Expected args keys: 'metric' (str), 'ds' (dataset dict), 'pair_index' (int)
    """
    metric: str = args["metric"]
    ds: Dict[str, Any] = args["ds"]
    idx: int = int(args.get("pair_index", -1))

    items = ds["items"]
    # Build (score, label, conv)
    triplets: List[tuple] = []
    for it in items:
        b = it["b"]
        r = it["r"]
        score = _get_confidence(r, metric) - _get_confidence(b, metric)
        label = int(it["label"])  # +1 FT, -1 TF, 0 otherwise
        # conv string for counts
        b_ok = _is_correct_record_math(b)
        r_ok = _is_correct_record_math(r)
        if (not b_ok) and r_ok:
            conv = "FT"
        elif b_ok and (not r_ok):
            conv = "TF"
        elif b_ok and r_ok:
            conv = "TT"
        else:
            conv = "FF"
        triplets.append((float(score), label, conv))

    # Best prefix by score
    triplets.sort(key=lambda x: x[0], reverse=True)
    cur = 0
    best = 0
    best_k = 0
    ft = tf = tt = ff = 0
    cur_ft = cur_tf = cur_tt = cur_ff = 0
    for k, (_, lab, conv) in enumerate(triplets, start=1):
        cur += int(lab)
        if conv == "FT":
            cur_ft += 1
        elif conv == "TF":
            cur_tf += 1
        elif conv == "TT":
            cur_tt += 1
        else:
            cur_ff += 1
        if cur > best:
            best = cur
            best_k = k
            ft, tf, tt, ff = cur_ft, cur_tf, cur_tt, cur_ff

    acc_base = ds["acc_base"]
    full_n = len(ds["all_base_ids"]) or 1
    acc_fused = acc_base + (best * 100.0 / full_n)
    gain_pp = acc_fused - acc_base

    # Always-override baseline
    always_net = sum(lab for _, lab, _ in triplets)
    acc_fused_always = acc_base + (always_net * 100.0 / full_n)

    return {
        "pair_index": idx,
        "base": ds["pair"].base_run_id,
        "rerun": ds["pair"].rerun_run_id,
        "metric": metric,
        "net": int(best),
        "overrides": int(best_k),
        "FT": int(ft),
        "TF": int(tf),
        "TT": int(tt),
        "FF": int(ff),
        "acc_base": float(acc_base),
        "acc_fused": float(acc_fused),
        "gain_pp": float(gain_pp),
        "always_net": int(always_net),
        "acc_fused_always": float(acc_fused_always),
    }


def run_metric_best_per_pair(metric: str, processes: int | None = None) -> None:
    """For a single metric, compute the best per-pair separator and compare to always-override.

    Prints a per-pair table sorted by gain and basic summary stats.
    """
    # Preload datasets once
    datasets = [_load_pair_overlap(p) for p in fusion_reruns]

    # Parallel over pairs if requested
    idxs = list(range(len(datasets)))
    if processes and processes > 1:
        console.print(
            Panel.fit(
                f"Per-pair best-separator for metric '{metric}' using {processes} processes",
                border_style="cyan",
            )
        )
        with ProcessPoolExecutor(max_workers=processes) as ex:
            futures = [
                ex.submit(
                    _metric_best_per_pair_worker,
                    {"pair_index": i, "ds": datasets[i], "metric": metric},
                )
                for i in idxs
            ]
            rows = [
                f.result()
                for f in tqdm(
                    as_completed(futures), total=len(futures), desc="Pairs", leave=False
                )
            ]
    else:
        rows = [
            _metric_best_per_pair_worker(
                {"pair_index": i, "ds": datasets[i], "metric": metric}
            )
            for i in idxs
        ]

    # Print per-pair table
    per = Table(title=f"Per-pair results for metric '{metric}'", box=box.SIMPLE_HEAVY)
    per.add_column("#", justify="right")
    per.add_column("base→rerun")
    per.add_column("net", justify="right")
    per.add_column("overrides", justify="right")
    per.add_column("acc_base", justify="right")
    per.add_column("acc_fused", justify="right")
    per.add_column("gain_pp", justify="right")
    per.add_column("always_net", justify="right")
    per.add_column("always_acc", justify="right")

    gains: List[float] = []
    for r in sorted(rows, key=lambda x: x["gain_pp"], reverse=True):
        gain = r["gain_pp"]
        gains.append(gain)
        gain_str = (
            f"[green]+{gain:.2f}[/green]"
            if gain > 0
            else (f"[red]{gain:.2f}[/red]" if gain < 0 else f"{gain:.2f}")
        )
        per.add_row(
            str(r["pair_index"]),
            f"{r['base']}→{r['rerun']}",
            str(r["net"]),
            str(r["overrides"]),
            f"{r['acc_base']:.2f}%",
            f"{r['acc_fused']:.2f}%",
            gain_str,
            str(r["always_net"]),
            f"{r['acc_fused_always']:.2f}%",
        )
    console.print(per)

    # Summary
    if gains:
        mean_gain = sum(gains) / len(gains)
        std_gain = _m.sqrt(sum((g - mean_gain) ** 2 for g in gains) / len(gains))
    else:
        mean_gain = 0.0
        std_gain = 0.0
    summ = Table(title="Summary", box=box.SIMPLE_HEAVY)
    summ.add_column("field", style="bold")
    summ.add_column("value", justify="right")
    summ.add_row("pairs", str(len(rows)))
    summ.add_row("mean gain (pp)", f"{mean_gain:.2f}")
    summ.add_row("std gain (pp)", f"{std_gain:.2f}")
    console.print(summ)


def _lopo_eval_for_metric(args: Dict[str, Any]) -> Dict[str, Any]:
    metric: str = args["metric"]
    datasets: List[Dict[str, Any]] = args["datasets"]
    per_pair: List[Dict[str, Any]] = []
    for holdout_idx in range(len(datasets)):
        # Build training pool from all except holdout
        train_scores: List[tuple] = []
        for i, ds in enumerate(datasets):
            if i == holdout_idx:
                continue
            train_scores.extend(_metric_scores(ds["items"], metric))
        t = _choose_threshold(train_scores)

        # Evaluate on holdout
        ds = datasets[holdout_idx]
        test_scores = _metric_scores(ds["items"], metric)
        net = sum(lab for sc, lab in test_scores if sc >= t)
        overrides = sum(1 for sc, _ in test_scores if sc >= t)
        acc_base = ds["acc_base"]
        full_n = len(ds["all_base_ids"]) or 1
        acc_fused = acc_base + (net * 100.0 / full_n)
        per_pair.append(
            {
                "pair_index": holdout_idx,
                "base": ds["pair"].base_run_id,
                "rerun": ds["pair"].rerun_run_id,
                "net": int(net),
                "overrides": int(overrides),
                "acc_base": float(acc_base),
                "acc_fused": float(acc_fused),
                "gain_pp": float(acc_fused - acc_base),
                "always_net": int(sum(lab for _, lab in test_scores)),
                "acc_fused_always": float(
                    acc_base + (sum(lab for _, lab in test_scores) * 100.0 / full_n)
                ),
            }
        )

    gains = [x["gain_pp"] for x in per_pair]
    mean_gain = sum(gains) / len(gains) if gains else 0.0
    std_gain = (
        _m.sqrt(sum((g - mean_gain) ** 2 for g in gains) / len(gains)) if gains else 0.0
    )
    return {
        "metric": metric,
        "mean_gain": float(mean_gain),
        "std_gain": float(std_gain),
        "details": per_pair,
    }


def run_lopo_metric_eval(
    metrics: Iterable[str] = ALL_METRICS, processes: int | None = None
) -> None:
    # Preload data for all pairs
    datasets = [_load_pair_overlap(p) for p in fusion_reruns]

    metric_list = list(metrics)
    # Parallelize across metrics if requested
    if processes and processes > 1:
        console.print(
            Panel.fit(
                f"LOPO metric eval using {processes} processes",
                border_style="cyan",
            )
        )
        with ProcessPoolExecutor(max_workers=processes) as ex:
            futures = [
                ex.submit(_lopo_eval_for_metric, {"metric": m, "datasets": datasets})
                for m in metric_list
            ]
            results_agg = [
                f.result()
                for f in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Metrics",
                    leave=False,
                )
            ]
    else:
        results_agg = [
            _lopo_eval_for_metric({"metric": m, "datasets": datasets})
            for m in metric_list
        ]

    # Print summary table
    summ = Table(title="LOPO global metric evaluation", box=box.SIMPLE_HEAVY)
    summ.add_column("metric")
    summ.add_column("mean gain (pp)", justify="right")
    summ.add_column("std gain (pp)", justify="right")
    for r in sorted(results_agg, key=lambda x: x["mean_gain"], reverse=True):
        mg = r["mean_gain"]
        mg_str = (
            f"[green]+{mg:.2f}[/green]"
            if mg > 0
            else (f"[red]{mg:.2f}[/red]" if mg < 0 else f"{mg:.2f}")
        )
        summ.add_row(r["metric"], mg_str, f"{r['std_gain']:.2f}")
    console.print(summ)

    # Optional: show best metric per pair
    per_pair_table = Table(title="Best metric per pair (LOPO)", box=box.SIMPLE_HEAVY)
    per_pair_table.add_column("#", justify="right")
    per_pair_table.add_column("base→rerun")
    per_pair_table.add_column("metric")
    # per_pair_table.add_column("net", justify="right")
    per_pair_table.add_column("overrides", justify="right")
    per_pair_table.add_column("acc_base", justify="right")
    per_pair_table.add_column("acc_fused", justify="right")
    per_pair_table.add_column("gain_pp", justify="right")
    per_pair_table.add_column("always_net", justify="right")
    per_pair_table.add_column("always_acc", justify="right")

    for i in range(len(datasets)):
        best = None
        for r in results_agg:
            d = next(x for x in r["details"] if x["pair_index"] == i)
            if (best is None) or (d["gain_pp"] > best["gain_pp"]):
                best = {"metric": r["metric"], **d}
        if best is not None:
            gain = best["gain_pp"]
            gain_str = (
                f"[green]+{gain:.2f}[/green]"
                if gain > 0
                else (f"[red]{gain:.2f}[/red]" if gain < 0 else f"{gain:.2f}")
            )
            per_pair_table.add_row(
                str(best["pair_index"]),
                f"{best['base']}→{best['rerun']}",
                best["metric"],
                str(best["net"]),
                str(best["overrides"]),
                f"{best['acc_base']:.2f}%",
                f"{best['acc_fused']:.2f}%",
                gain_str,
                str(best.get("always_net", 0)),
                f"{best.get('acc_fused_always', best['acc_base']):.2f}%",
            )
    console.print(per_pair_table)


# ----------------- Utility functions for statistics -----------------


def format_mean_std(values, decimal_places=3, use_ci=False, confidence=0.95):
    """Format a list of values as mean ± std or mean [CI_lower, CI_upper].

    Args:
        values: List of numerical values
        decimal_places: Number of decimal places to display
        use_ci: If True, return confidence interval instead of std
        confidence: Confidence level for CI (default 0.95 for 95% CI)

    Returns:
        String in format "mean ± std" or "mean [CI_lower, CI_upper]"

    Examples:
        >>> format_mean_std([1.2, 1.5, 1.8])
        "1.50 ± 0.30"
        >>> format_mean_std([1.2, 1.5, 1.8], use_ci=True)
        "1.50 [0.90, 2.10]"
    """
    if np is None:
        return "Error: NumPy required"

    if not values:
        return "No data"

    values = np.array(values)
    mean_val = np.mean(values)

    if use_ci and len(values) > 1:
        # Calculate confidence interval
        std_val = np.std(values, ddof=1)  # Sample standard deviation
        sem = std_val / np.sqrt(len(values))  # Standard error of mean

        if stats is not None:
            ci = stats.t.interval(confidence, len(values) - 1, loc=mean_val, scale=sem)
            return f"{mean_val:.{decimal_places}f} [{ci[0]:.{decimal_places}f}, {ci[1]:.{decimal_places}f}]"
        else:
            # Fallback to approximate normal CI if scipy not available
            z_score = (
                1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
            )
            margin = z_score * sem
            ci_lower = mean_val - margin
            ci_upper = mean_val + margin
            return f"{mean_val:.{decimal_places}f} [{ci_lower:.{decimal_places}f}, {ci_upper:.{decimal_places}f}]"
    else:
        # Calculate standard deviation
        if len(values) == 1:
            return f"{mean_val:.{decimal_places}f}"

        std_val = np.std(values, ddof=1)  # Sample standard deviation
        return f"{mean_val:.{decimal_places}f} ± {std_val:.{decimal_places}f}"


# ----------------- Top-3 metric analysis -----------------


def analyze_top3_metric_performance(
    metrics: Iterable[str] = ALL_METRICS,
    processes: int | None = None,
    num_delta_steps: int = 100,
) -> None:
    """Analyze each metric to find which gives the highest mean accuracy for top 3 performing runs.

    For each metric, we:
    1. Test different delta thresholds
    2. Calculate accuracy gains for each pair
    3. Find the top 3 performing runs
    4. Return mean ± std for the top 3
    """
    if np is None:
        console.print(Panel.fit("NumPy required for this analysis", border_style="red"))
        return

    # Preload all pair data
    datasets = [_load_pair_overlap(p) for p in fusion_reruns]
    metric_list = list(metrics)

    console.print(
        Panel.fit(
            f"Analyzing top-3 performance for {len(metric_list)} metrics",
            border_style="cyan",
        )
    )

    metric_results = []

    for metric in tqdm(metric_list, desc="Processing metrics"):
        # For this metric, collect all delta values and gains across all pairs
        all_results = []

        for ds in datasets:
            items = ds["items"]
            acc_base = ds["acc_base"]
            full_n = len(ds["all_base_ids"]) or 1

            # Get all delta values for this metric
            deltas = []
            labels = []
            for it in items:
                b = it["b"]
                r = it["r"]
                delta = _get_confidence(r, metric) - _get_confidence(b, metric)
                deltas.append(delta)
                labels.append(it["label"])

            if not deltas:
                continue

            deltas = np.array(deltas)
            labels = np.array(labels)

            # Test different delta thresholds
            min_delta, max_delta = np.min(deltas), np.max(deltas)
            if min_delta == max_delta:
                thresholds = [min_delta]
            else:
                thresholds = np.linspace(min_delta, max_delta, num_delta_steps)

            best_gain = -float("inf")
            best_acc = acc_base

            for threshold in thresholds:
                # Select items where delta >= threshold
                selected_mask = deltas >= threshold
                net_gain = np.sum(labels[selected_mask])
                acc_fused = acc_base + (net_gain * 100.0 / full_n)
                gain = acc_fused - acc_base

                if gain > best_gain:
                    best_gain = gain
                    best_acc = acc_fused

            all_results.append(
                {
                    "pair": ds["pair"],
                    "best_gain": best_gain,
                    "best_acc": best_acc,
                    "base_acc": acc_base,
                }
            )

        # Sort by accuracy gain and get top 3
        all_results.sort(key=lambda x: x["best_gain"], reverse=True)
        top3_results = all_results[:3]

        if len(top3_results) >= 3:
            top3_gains = [r["best_gain"] for r in top3_results]
            top3_accs = [r["best_acc"] for r in top3_results]

            mean_gain = np.mean(top3_gains)
            std_gain = np.std(top3_gains, ddof=1)  # Sample std
            mean_acc = np.mean(top3_accs)
            std_acc = np.std(top3_accs, ddof=1)

            # 95% confidence interval for mean (assuming normal distribution)
            from scipy import stats

            try:
                ci_gain = stats.t.interval(
                    0.95,
                    len(top3_gains) - 1,
                    loc=mean_gain,
                    scale=std_gain / np.sqrt(len(top3_gains)),
                )
                ci_acc = stats.t.interval(
                    0.95,
                    len(top3_accs) - 1,
                    loc=mean_acc,
                    scale=std_acc / np.sqrt(len(top3_accs)),
                )
            except ImportError:
                # Fallback if scipy not available
                margin_gain = (
                    1.96 * std_gain / np.sqrt(len(top3_gains))
                )  # Approximate 95% CI
                margin_acc = 1.96 * std_acc / np.sqrt(len(top3_accs))
                ci_gain = (mean_gain - margin_gain, mean_gain + margin_gain)
                ci_acc = (mean_acc - margin_acc, mean_acc + margin_acc)

            metric_results.append(
                {
                    "metric": metric,
                    "mean_gain": mean_gain,
                    "std_gain": std_gain,
                    "ci_gain": ci_gain,
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "ci_acc": ci_acc,
                    "top3_details": top3_results,
                }
            )

    # Sort by mean accuracy gain
    metric_results.sort(key=lambda x: x["mean_gain"], reverse=True)

    # Display results
    results_table = Table(
        title="Top-3 Metric Performance Analysis", box=box.SIMPLE_HEAVY
    )
    results_table.add_column("Rank", justify="right")
    results_table.add_column("Metric")
    results_table.add_column("Mean Gain (pp)", justify="right")
    results_table.add_column("95% CI Gain", justify="right")
    results_table.add_column("Mean Top-3 Acc (%)", justify="right")
    results_table.add_column("95% CI Acc", justify="right")

    for i, result in enumerate(metric_results, 1):
        mean_gain = result["mean_gain"]
        ci_gain = result["ci_gain"]
        mean_acc = result["mean_acc"]
        ci_acc = result["ci_acc"]

        gain_str = (
            f"[green]+{mean_gain:.2f}[/green]"
            if mean_gain > 0
            else f"[red]{mean_gain:.2f}[/red]"
        )
        ci_gain_str = f"[{ci_gain[0]:.2f}, {ci_gain[1]:.2f}]"
        ci_acc_str = f"[{ci_acc[0]:.2f}, {ci_acc[1]:.2f}]"

        results_table.add_row(
            str(i),
            result["metric"],
            gain_str,
            ci_gain_str,
            f"{mean_acc:.2f}",
            ci_acc_str,
        )

    console.print(results_table)

    # Show details for the best metric
    if metric_results:
        best = metric_results[0]
        # Extract top 3 values for cleaner formatting
        top3_gains = [r["best_gain"] for r in best["top3_details"]]
        top3_accs = [r["best_acc"] for r in best["top3_details"]]

        gain_formatted = format_mean_std(top3_gains)
        acc_formatted = format_mean_std(top3_accs)
        gain_ci_formatted = format_mean_std(top3_gains, use_ci=True)
        acc_ci_formatted = format_mean_std(top3_accs, use_ci=True)

        console.print(
            Panel.fit(
                f"Best metric: [bold]{best['metric']}[/bold]\n"
                f"Mean gain: {gain_formatted} pp\n"
                f"Mean accuracy: {acc_formatted}%\n"
                f"95% CI gain: {gain_ci_formatted} pp\n"
                f"95% CI accuracy: {acc_ci_formatted}%",
                border_style="green",
            )
        )

        # Show top 3 details for best metric
        details_table = Table(
            title=f"Top 3 pairs for {best['metric']}", box=box.SIMPLE_HEAVY
        )
        details_table.add_column("Rank", justify="right")
        details_table.add_column("Pair")
        details_table.add_column("Base Acc (%)", justify="right")
        details_table.add_column("Best Acc (%)", justify="right")
        details_table.add_column("Gain (pp)", justify="right")

        for i, detail in enumerate(best["top3_details"], 1):
            pair = detail["pair"]
            gain = detail["best_gain"]
            gain_str = (
                f"[green]+{gain:.2f}[/green]" if gain > 0 else f"[red]{gain:.2f}[/red]"
            )

            details_table.add_row(
                str(i),
                f"{pair.base_run_id}→{pair.rerun_run_id}",
                f"{detail['base_acc']:.2f}",
                f"{detail['best_acc']:.2f}",
                gain_str,
            )

        console.print(details_table)


def experiment():
    check_all_runs_exist(fusion_reruns)
    # run_single_pair_analysis(0)
    # run_classifier_pair_analysis(0)
    # run_greedy_cv_pair_analysis(2, max_features=2)
    # run_greedy_cv_all_pairs(max_features=1, processes=os.cpu_count())
    # run_lopo_metric_eval(processes=os.cpu_count())
    # run_metric_best_per_pair("consensus_support", processes=os.cpu_count())
    # run_metric_best_per_pair("prm_top_frac", processes=os.cpu_count())
    analyze_top3_metric_performance(processes=os.cpu_count())


if __name__ == "__main__":
    experiment()
