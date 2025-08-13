#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sal.const import PROBE_OUTPUT_ROOT

console = Console()


def _load_probe(run_id: str, index: int) -> Dict[str, Any]:
    path = PROBE_OUTPUT_ROOT / run_id / f"{index}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
        if not line:
            raise RuntimeError(f"Empty file: {path}")
        return json.loads(line)


def _shorten(text: str, max_len: int = 280) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n\n", "\n").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _score_style(value: float) -> str:
    try:
        return "green" if float(value) >= 0 else "red"
    except Exception:
        return "white"


def _rank_by_last(candidates: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    scores: List[Tuple[int, float]] = []
    for i, c in enumerate(candidates):
        agg = c.get("aggregates") or {}
        last = float(agg.get("last", 0.0))
        scores.append((i, last))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


@click.group(help="Probe visualiser utilities")
def cli() -> None:
    pass


@cli.command(name="detail", help="Inspect a single probe result")
@click.option("--run-id", required=True, type=str)
@click.option("--index", required=True, type=int)
def cmd_detail(run_id: str, index: int) -> None:
    record = _load_probe(run_id, index)

    header = Table.grid(padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Run", run_id)
    header.add_row("File", str(PROBE_OUTPUT_ROOT / run_id / f"{index}.jsonl"))
    header.add_row("Index", str(index))
    console.print(Panel(header, title="Probe Result", box=box.ROUNDED))

    problem = str(record.get("problem") or "")
    console.print(Panel(_shorten(problem, 800), title="Problem", box=box.SQUARE))

    prefix = str(record.get("prefix_text") or "")
    console.print(Panel(_shorten(prefix, 800), title="Prefix (shared)", box=box.SQUARE))

    strategies: List[Dict[str, Any]] = record.get("strategies", [])
    if not strategies:
        console.print(Text("No strategies present", style="red"))
        return

    for strat in strategies:
        name = str(strat.get("name") or "strategy")
        params = strat.get("params") or {}
        candidates: List[Dict[str, Any]] = strat.get("candidates", [])

        if not candidates:
            console.print(Panel(Text("No candidates"), title=name, box=box.ROUNDED))
            continue

        rank = _rank_by_last(candidates)
        if len(rank) >= 2:
            top_margin = float(rank[0][1] - rank[1][1])
        else:
            top_margin = float(rank[0][1]) if rank else 0.0

        table = Table(
            title=f"{name} (params={params}, margin={top_margin:+.4f})",
            box=box.MINIMAL_HEAVY_HEAD,
            show_lines=True,
        )
        table.add_column("#", style="bold cyan", justify="right")
        table.add_column("last", justify="right")
        table.add_column("mean", justify="right")
        table.add_column("min", justify="right")
        table.add_column("prod", justify="right")
        table.add_column("stop", justify="left")
        table.add_column("preview", overflow="fold")

        for i, _ in rank:
            c = candidates[i]
            agg = c.get("aggregates") or {}
            last = float(agg.get("last", 0.0))
            mean = float(agg.get("mean", 0.0))
            min_v = float(agg.get("min", 0.0))
            prod = float(agg.get("prod", 0.0))
            stop = str(c.get("stop_reason") or "")
            preview = str(c.get("text") or "")

            table.add_row(
                str(i),
                Text(f"{last:+.4f}", style=_score_style(last)),
                Text(f"{mean:+.4f}", style=_score_style(mean)),
                Text(f"{min_v:+.4f}", style=_score_style(min_v)),
                Text(f"{prod:+.4f}", style=_score_style(prod)),
                stop,
                preview,
            )

        console.print(table)


if __name__ == "__main__":
    cli()
