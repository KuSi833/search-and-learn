#!/usr/bin/env python3
"""
Selection experiments entry point script.
Clean command-line interface for uncertainty analysis, export, and visualization.

Usage examples:
    python scripts/selection.py analyze --runs run1,run2,run3
    python scripts/selection.py export --runs run1,run2 --coverage 20 --metric agreement_ratio
    python scripts/selection.py visualize --runs run1,run2,run3
    python scripts/selection.py all --runs run1,run2,run3
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from sal.utils.constants import Benchmarks
from sal_experiments.selection import (
    SelectionAnalyzer,
    SelectionExporter,
    SelectionVisualizer,
    get_default_run_ids,
)

console = Console()


def parse_run_ids(run_ids_str: str) -> List[str]:
    """Parse comma-separated run IDs string."""
    if not run_ids_str:
        return []
    return [rid.strip() for rid in run_ids_str.split(",") if rid.strip()]


def cmd_analyze(args) -> None:
    """Run uncertainty analysis."""
    run_ids = parse_run_ids(args.runs) if args.runs else get_default_run_ids()

    console.print(
        Panel(
            Text(f"Uncertainty Analysis\nRuns: {', '.join(run_ids)}", style="bold"),
            box=box.ROUNDED,
        )
    )

    analyzer = SelectionAnalyzer(run_ids)
    analyzer.analyze_uncertainty()


def cmd_export(args) -> None:
    """Export uncertain subsets."""
    run_ids = parse_run_ids(args.runs) if args.runs else get_default_run_ids()

    console.print(
        Panel(
            Text(
                f"Export Uncertain Subsets\n"
                f"Runs: {', '.join(run_ids)}\n"
                f"Coverage: {args.coverage}%\n"
                f"Metric: {args.metric}",
                style="bold",
            ),
            box=box.ROUNDED,
        )
    )

    benchmark = Benchmarks.MATH500.value  # Default benchmark
    SelectionExporter.export_multi_runs(run_ids, args.coverage, args.metric, benchmark)


def cmd_visualize(args) -> None:
    """Generate visualization figures."""
    run_ids = parse_run_ids(args.runs) if args.runs else get_default_run_ids()

    console.print(
        Panel(
            Text(f"Generate Figures\nRuns: {', '.join(run_ids)}", style="bold"),
            box=box.ROUNDED,
        )
    )

    visualizer = SelectionVisualizer(run_ids)
    visualizer.generate_figures()


def cmd_all(args) -> None:
    """Run all: analyze + visualize."""
    run_ids = parse_run_ids(args.runs) if args.runs else get_default_run_ids()

    console.print(
        Panel(
            Text(
                f"Complete Analysis Pipeline\nRuns: {', '.join(run_ids)}", style="bold"
            ),
            box=box.DOUBLE,
        )
    )

    # 1. Analysis
    console.print(Text("\n🔍 PHASE 1: Uncertainty Analysis", style="bold magenta"))
    analyzer = SelectionAnalyzer(run_ids)
    analyzer.analyze_uncertainty()

    # 2. Visualization
    console.print(Text("\n📊 PHASE 2: Generate Figures", style="bold magenta"))
    visualizer = SelectionVisualizer(run_ids)
    visualizer.generate_figures()

    console.print(Text("\n✅ Complete analysis finished!", style="bold green"))


def main():
    parser = argparse.ArgumentParser(
        description="Selection experiments: uncertainty analysis and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze --runs run1,run2,run3
  %(prog)s export --runs run1,run2 --coverage 20 --metric agreement_ratio
  %(prog)s visualize --runs run1,run2,run3
  %(prog)s all --runs run1,run2,run3
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments
    def add_common_args(p):
        p.add_argument(
            "--runs",
            type=str,
            help="Comma-separated list of run IDs (default: use fusion best runs)",
        )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run uncertainty analysis")
    add_common_args(analyze_parser)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export uncertain subsets")
    add_common_args(export_parser)
    export_parser.add_argument(
        "--coverage",
        type=float,
        default=20.0,
        help="Coverage percentage (default: 20.0)",
    )
    export_parser.add_argument(
        "--metric",
        type=str,
        default="agreement_ratio",
        choices=["agreement_ratio", "entropy_freq", "consensus_support"],
        help="Uncertainty metric (default: agreement_ratio)",
    )

    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Generate figures")
    add_common_args(visualize_parser)

    # All command (analyze + visualize)
    all_parser = subparsers.add_parser("all", help="Run complete analysis pipeline")
    add_common_args(all_parser)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "analyze":
            cmd_analyze(args)
        elif args.command == "export":
            cmd_export(args)
        elif args.command == "visualize":
            cmd_visualize(args)
        elif args.command == "all":
            cmd_all(args)
    except KeyboardInterrupt:
        console.print(Text("\n⚠️  Interrupted by user", style="yellow"))
    except Exception as e:
        console.print(Text(f"❌ Error: {str(e)}", style="red"))
        raise


if __name__ == "__main__":
    main()
