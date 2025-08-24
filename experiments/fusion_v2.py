import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from experiments.fusion import (
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
    FusionRerun(base_run_id="0hermenf", rerun_run_id="77pyab58", coverage=20),
    FusionRerun(base_run_id="58xqqffr", rerun_run_id="gfw8x07r", coverage=20),
    FusionRerun(base_run_id="6nuelycf", rerun_run_id="tqfyvf5w", coverage=20),
    FusionRerun(base_run_id="defwqu3v", rerun_run_id="77pyab58", coverage=20),
    FusionRerun(base_run_id="dtjaqwyg", rerun_run_id="tqfyvf5w", coverage=20),
    FusionRerun(base_run_id="whldvunb", rerun_run_id="tqfyvf5w", coverage=20),
    FusionRerun(base_run_id="22z91qa7", rerun_run_id="tqfyvf5w", coverage=10),
    FusionRerun(base_run_id="2jcdgx7c", rerun_run_id="tqfyvf5w", coverage=10),
    FusionRerun(base_run_id="8ff83v7m", rerun_run_id="77pyab58", coverage=10),
    FusionRerun(base_run_id="8yyge5wj", rerun_run_id="gfw8x07r", coverage=10),
    FusionRerun(base_run_id="ch6s6c0c", rerun_run_id="tqfyvf5w", coverage=10),
    FusionRerun(base_run_id="ylsdosex", rerun_run_id="77pyab58", coverage=10),
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

    metrics = (
        "agreement_ratio",
        "entropy_freq",
        "entropy_weighted",
        "prm_margin",
        "prm_top_frac",
        "group_top_frac",
        "prm_std",
        "prm_mean",
    )

    results_rows: List[Dict[str, Any]] = []
    for metric in metrics:
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
            }
        )

    return {
        "pair": pair,
        "all_base_ids": all_base_ids,
        "overlap_ids": overlap_ids,
        "acc_base": acc_base,
        "acc_rerun": acc_rerun,
        "conversions": conversions,
        "oracle_net_gain": oracle_net_gain,
        "results_rows": results_rows,
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
    ]:
        res_table.add_column(col, justify=(j or "left"))

    for row in sorted(results_rows, key=lambda x: x["best_net"], reverse=True):
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
        )
    console.print(res_table)


def run_single_pair_analysis(pair_index: int = 0) -> None:
    pair = fusion_reruns[pair_index]
    result = analyze_pair(pair)
    render_pair_analysis(result)


if __name__ == "__main__":
    check_all_runs_exist(fusion_reruns)
    run_single_pair_analysis(0)
