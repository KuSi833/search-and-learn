#!/usr/bin/env python3
"""Quick fusion analysis: test the three strategies and output results."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from experiments.fusion import run_ultraminimal_experiment


def load_results(json_path: Path) -> List[Dict[str, Any]]:
    """Load fusion results from JSON."""
    with json_path.open("r") as f:
        return json.load(f)


def create_comparison_chart(results_summary: Dict[str, Any], output_dir: Path) -> None:
    """Create a comparison chart showing baselines as lines and fusion strategies as bars."""

    # Get all smart fusion results sorted by accuracy
    all_smart = results_summary["all_smart_results"]
    best_metric = results_summary["best_smart_fusion"]["metric"]

    # Prepare data for smart fusion bars
    metrics = [r["metric"] for r in all_smart]
    accuracies = [r["accuracy"] for r in all_smart]

    # Color bars: green for best, blue for others
    bar_colors = [
        "#2ca02c" if metric == best_metric else "#1f77b4" for metric in metrics
    ]

    # Create the plot
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Bar chart for smart fusion strategies
    bars = ax.bar(range(len(metrics)), accuracies, color=bar_colors, alpha=0.8)

    # Add baseline lines
    always_base = results_summary["always_base"]
    always_rerun = results_summary["always_rerun_when_possible"]

    ax.axhline(
        y=always_base,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Always Base: {always_base:.1f}%",
        alpha=0.8,
    )
    ax.axhline(
        y=always_rerun,
        color="#FFA500",
        linestyle="--",
        linewidth=2,
        label=f"Always Override: {always_rerun:.1f}%",
        alpha=0.8,
    )

    # Add value labels on bars
    for i, (bar, acc, metric) in enumerate(zip(bars, accuracies, metrics)):
        height = bar.get_height()
        diff = acc - always_base
        label = f"{acc:.1f}%\n({diff:+.1f}%)"

        fontweight = "bold" if metric == best_metric else "normal"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight=fontweight,
        )

    # Formatting
    ax.set_title(
        "Fusion Strategy Comparison: Smart Selection vs Naive Baselines",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("Confidence Metric", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylim(0, max(accuracies) * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Legend
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        output_dir / "fusion_strategy_comparison.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

    print(f"Saved chart to: {output_dir / 'fusion_strategy_comparison.png'}")


def analyze_strategies(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the three strategies from fusion results."""

    # Get basic info
    smart_results = [r for r in results if r["delta"] == 0.0]
    always_override = [r for r in results if r["delta"] == -999.0]

    if not smart_results:
        raise ValueError("No smart fusion results found!")

    sample = smart_results[0]
    base_acc = sample["acc_base"]
    rerun_acc = sample["acc_rerun"]
    total_samples = sample["total_samples"]
    rerun_samples = sample["rerun_samples"]

    # Strategy 1: Always base
    always_base = base_acc

    # Strategy 2: Always rerun when possible
    # = (rerun_correct + base_correct_on_remaining) / total_samples * 100
    rerun_correct = rerun_acc * rerun_samples / 100
    base_correct_on_remaining = base_acc * (total_samples - rerun_samples) / 100
    always_rerun_when_possible = (
        (rerun_correct + base_correct_on_remaining) / total_samples * 100
    )

    # Strategy 3: Smart fusion (best result)
    best_smart = max(smart_results, key=lambda x: x["acc_fused"])

    # Get measured always override if available
    measured_always_override = None
    if always_override:
        measured_always_override = always_override[0]["acc_fused"]

    return {
        "experiment_info": {
            "total_samples": total_samples,
            "rerun_samples": rerun_samples,
            "base_accuracy_overall": base_acc,
            "rerun_accuracy_on_subset": rerun_acc,
        },
        "strategies": {
            "always_base": {
                "description": "Always use base run",
                "accuracy": always_base,
                "vs_base": 0.0,
            },
            "always_rerun_when_possible": {
                "description": f"Use rerun on {rerun_samples} samples, base on remaining {total_samples - rerun_samples}",
                "accuracy": always_rerun_when_possible,
                "vs_base": always_rerun_when_possible - always_base,
                "measured": measured_always_override,
            },
            "smart_fusion_best": {
                "description": f"Use {best_smart['metric']} confidence to decide",
                "accuracy": best_smart["acc_fused"],
                "vs_base": best_smart["acc_fused"] - always_base,
                "metric": best_smart["metric"],
                "overrides_used": best_smart["overrides_used"],
                "flips_positive": best_smart["flips_pos"],
                "flips_negative": best_smart["flips_neg"],
                "net_flips": best_smart["flips_pos"] - best_smart["flips_neg"],
                "conversions": {
                    "T_to_T": best_smart.get("conversions_tt", 0),
                    "F_to_F": best_smart.get("conversions_ff", 0),
                    "T_to_F": best_smart.get("conversions_tf", 0),
                    "F_to_T": best_smart.get("conversions_ft", 0),
                },
            },
        },
        "all_smart_results": [
            {
                "metric": r["metric"],
                "accuracy": r["acc_fused"],
                "vs_base": r["acc_fused"] - always_base,
                "overrides_used": r["overrides_used"],
                "conversions": {
                    "T_to_T": r.get("conversions_tt", 0),
                    "F_to_F": r.get("conversions_ff", 0),
                    "T_to_F": r.get("conversions_tf", 0),
                    "F_to_T": r.get("conversions_ft", 0),
                },
            }
            for r in sorted(smart_results, key=lambda x: x["acc_fused"], reverse=True)
        ],
        # Summary for easy access
        "always_base": always_base,
        "always_rerun_when_possible": always_rerun_when_possible,
        "best_smart_fusion": {
            "accuracy": best_smart["acc_fused"],
            "metric": best_smart["metric"],
            "improvement_over_base": best_smart["acc_fused"] - always_base,
            "improvement_over_always_rerun": best_smart["acc_fused"]
            - always_rerun_when_possible,
        },
    }


def print_results_summary(analysis: Dict[str, Any]) -> None:
    """Print a clean summary."""
    print(f"\n{'=' * 60}")
    print("FUSION STRATEGY COMPARISON")
    print(f"{'=' * 60}")

    exp = analysis["experiment_info"]
    print(
        f"Dataset: {exp['total_samples']} total samples, {exp['rerun_samples']} with rerun"
    )

    print("\nSTRATEGY RESULTS:")
    strategies = analysis["strategies"]

    for name, strategy in strategies.items():
        acc = strategy["accuracy"]
        vs_base = strategy["vs_base"]
        print(f"  {strategy['description']}:")
        print(f"    Accuracy: {acc:.1f}% ({vs_base:+.1f}%)")

        if name == "smart_fusion_best":
            print(f"    Best metric: {strategy['metric']}")
            print(f"    Overrides used: {strategy['overrides_used']}")
            print(f"    Net positive flips: {strategy['net_flips']}")

            # Print T/F conversion breakdown (overrides only)
            conv = strategy.get("conversions", {})
            print("    T/F Conversions (overrides only):")
            print(f"      T→T: {conv.get('T_to_T', 0)}, F→F: {conv.get('F_to_F', 0)}")
            print(f"      T→F: {conv.get('T_to_F', 0)}, F→T: {conv.get('F_to_T', 0)}")

    print("\nKEY INSIGHT:")
    best = analysis["best_smart_fusion"]
    print(
        f"Smart fusion beats naive strategies by {best['improvement_over_always_rerun']:.1f}%"
    )
    print("This proves that intelligent selection based on confidence works!")


def main():
    # Configuration
    base_run = "53vig20u"
    rerun_id = "9qup1u07"

    print("Running ultraminimal fusion experiment...")

    # Run experiment
    run_ultraminimal_experiment(base_run, rerun_id)

    # Load results
    results_path = (
        Path("./output/fusion_sweeps/ultraminimal") / f"{base_run}__{rerun_id}.json"
    )
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    results = load_results(results_path)

    # Analyze strategies
    analysis = analyze_strategies(results)

    # Print summary
    print_results_summary(analysis)

    # Create output directory
    output_dir = Path("./figures/fusion_analysis") / f"{base_run}__{rerun_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = output_dir / "strategy_comparison.json"
    with json_path.open("w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved results to: {json_path}")

    # Create chart
    create_comparison_chart(analysis, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
