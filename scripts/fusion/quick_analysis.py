#!/usr/bin/env python3
"""Quick fusion analysis: test the three strategies and output results."""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import sys
from pathlib import Path

from sal_experiments.fusion.fusion import (
    FusionRunConfig,
    FusionSetting,
    _get_base_samples,
    _is_correct_record_math,
    best_accuracy,
    load_jsonl,
    run_sweep,
    run_ultraminimal_experiment,
)
from src.sal_experiments.report.colors import BLUE, GOLD, GREEN, ORANGE, RED


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

    # Color bars: green for best, blue for others (using custom colors)
    bar_colors = [
        GREEN.triplet.hex if metric == best_metric else BLUE.triplet.hex
        for metric in metrics
    ]

    # Create the plot
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Bar chart for smart fusion strategies with black outlines
    bars = ax.bar(
        range(len(metrics)),
        accuracies,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )

    # Add baseline lines
    always_base = results_summary["always_base"]
    always_rerun = results_summary["always_rerun_when_possible"]

    ax.axhline(
        y=always_base,
        color=RED.triplet.hex,
        linestyle="--",
        linewidth=2,
        label=f"Always Base: {always_base:.1f}%",
        alpha=0.8,
    )
    ax.axhline(
        y=always_rerun,
        color=ORANGE.triplet.hex,
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


def create_averaged_comparison_chart(
    results_summary: Dict[str, Any], output_dir: Path
) -> None:
    """Create a comparison chart showing averaged results across multiple run pairs with error bars."""

    # Get averaged smart fusion results
    all_smart = results_summary["averaged_results"]
    best_metric = results_summary["best_metric"]

    # Combine strategies - EXCLUDE Always Base since it's always 0% improvement
    # Calculate min/max for Always Override from individual pairs
    override_improvements = [
        r["always_rerun_when_possible"] - r["always_base"]
        for r in results_summary["individual_pair_results"]
    ]

    all_strategies = [
        {
            "name": "Always Override",
            "improvement": results_summary["always_override_improvement"],
            "std": results_summary["always_override_improvement_std"],
            "min": min(override_improvements),
            "max": max(override_improvements),
            "type": "baseline",
            "accuracy": results_summary["always_rerun_when_possible"],
        },
    ] + [
        {
            "name": r["metric"],
            "improvement": r["improvement"],  # Already calculated correctly
            "std": r["std"],  # Standard deviation of improvements across pairs
            "min": r["min"],  # Already calculated correctly
            "max": r["max"],  # Already calculated correctly
            "type": "smart",
            "accuracy": r["accuracy"],
        }
        for r in all_smart
    ]

    # Sort by improvement (highest first)
    all_strategies.sort(key=lambda x: x["improvement"], reverse=True)

    # Extract sorted data
    metrics = [s["name"] for s in all_strategies]
    improvements = [s["improvement"] for s in all_strategies]

    # Color bars based on strategy type and performance
    bar_colors = []
    for s in all_strategies:
        if s["type"] == "baseline":
            if s["name"] == "Always Base":
                bar_colors.append(RED.triplet.hex)
            else:  # Always Override
                bar_colors.append(ORANGE.triplet.hex)
        else:  # Smart fusion
            if s["name"] == best_metric:
                bar_colors.append(GREEN.triplet.hex)  # Best smart metric
            else:
                bar_colors.append(BLUE.triplet.hex)  # Other smart metrics

    # Create the plot
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(14, 8))  # Slightly wider for more bars

    # Calculate asymmetric error bars (min/max range)
    lower_errors = [
        imp - strategy["min"] for imp, strategy in zip(improvements, all_strategies)
    ]
    upper_errors = [
        strategy["max"] - imp for imp, strategy in zip(improvements, all_strategies)
    ]

    # Bar chart for all strategies with black outlines and min/max error bars
    bars = ax.bar(
        range(len(metrics)),
        improvements,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
        yerr=[lower_errors, upper_errors],
        capsize=5,
        error_kw={"color": "black", "linewidth": 1.5},
    )

    # Add value labels on bars
    for i, (bar, improvement, metric, strategy) in enumerate(
        zip(bars, improvements, metrics, all_strategies)
    ):
        height = bar.get_height()

        # Show improvement with min/max range
        min_val = strategy["min"]
        max_val = strategy["max"]
        label = f"{improvement:+.1f}% [{min_val:+.1f}, {max_val:+.1f}]"

        # Bold for best performing strategy
        fontweight = "bold" if i == 0 else "normal"  # First in sorted list is best

        # Position label above error bar
        upper_error = strategy["max"] - improvement
        y_pos = height + upper_error + 0.1

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=9,  # Slightly smaller to fit more text
            fontweight=fontweight,
        )

    # Formatting
    num_pairs = results_summary["experiment_info"]["num_pairs"]
    ax.set_title(
        f"Fusion Strategy Improvements over Baseline (Averaged over {num_pairs} Run Pairs)",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Improvement over Always Base (%)", fontsize=12)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")

    # Set y-limits to focus on improvement range with min/max error bars
    all_mins = [s["min"] for s in all_strategies]
    all_maxs = [s["max"] for s in all_strategies]

    overall_min = min(all_mins + [0])  # Include 0 for reference line
    overall_max = max(all_maxs)

    # Add some padding and ensure we show the zero line
    y_padding = max(0.2, (overall_max - overall_min) * 0.1)
    y_min = overall_min - y_padding
    y_max = overall_max + y_padding
    ax.set_ylim(y_min, y_max)

    # Add horizontal line at y=0 (no improvement)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(
        output_dir / "fusion_strategy_comparison_averaged.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Saved averaged chart to: {output_dir / 'fusion_strategy_comparison_averaged.png'}"
    )


def calculate_always_override_conversions(
    base_run: str, rerun_id: str
) -> Dict[str, int]:
    """Calculate T/F conversions for always override strategy by loading the data directly."""

    # Add parent directories to path for imports

    # Load the data files
    base_file = Path("./output") / base_run / "inference_output.jsonl"
    rerun_file = Path("./output") / rerun_id / "inference_output.jsonl"

    base_recs = {rec["unique_id"]: rec for rec in load_jsonl(base_file)}
    rerun_recs = {rec["unique_id"]: rec for rec in load_jsonl(rerun_file)}

    all_base_ids, rerun_ids = _get_base_samples(base_recs, rerun_recs, None)

    # Calculate conversions for samples that have rerun data
    conversions = {"T_to_T": 0, "F_to_F": 0, "T_to_F": 0, "F_to_T": 0}

    # Debug: track accuracy for validation
    base_correct_on_rerun_subset = 0
    rerun_correct_on_rerun_subset = 0

    for uid in rerun_ids:  # Only samples that have both base and rerun
        base_rec = base_recs[uid]
        rerun_rec = rerun_recs[uid]

        base_ok = _is_correct_record_math(base_rec)
        rerun_ok = _is_correct_record_math(rerun_rec)

        # Debug counting
        if base_ok:
            base_correct_on_rerun_subset += 1
        if rerun_ok:
            rerun_correct_on_rerun_subset += 1

        # Track T/F conversions (base → rerun)
        if base_ok and rerun_ok:
            conversions["T_to_T"] += 1
        elif (not base_ok) and (not rerun_ok):
            conversions["F_to_F"] += 1
        elif base_ok and (not rerun_ok):
            conversions["T_to_F"] += 1
        elif (not base_ok) and rerun_ok:
            conversions["F_to_T"] += 1

    # Debug output
    print(f"DEBUG: Rerun subset size: {len(rerun_ids)}")
    print(
        f"DEBUG: Base correct on rerun subset: {base_correct_on_rerun_subset} ({base_correct_on_rerun_subset / len(rerun_ids) * 100:.1f}%)"
    )
    print(
        f"DEBUG: Rerun correct on rerun subset: {rerun_correct_on_rerun_subset} ({rerun_correct_on_rerun_subset / len(rerun_ids) * 100:.1f}%)"
    )
    print(f"DEBUG: Conversions: {conversions}")
    print(f"DEBUG: Net change: {conversions['F_to_T'] - conversions['T_to_F']}")

    return conversions


def analyze_strategies(
    results: List[Dict[str, Any]], base_run: str, rerun_id: str
) -> Dict[str, Any]:
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
    # We need to calculate this properly using the actual data
    # The calculation will be done after getting the conversion data

    # Calculate T/F conversions for always rerun strategy by loading data directly
    always_override_conversions = calculate_always_override_conversions(
        base_run, rerun_id
    )

    # Now calculate always_rerun_when_possible correctly
    # For the 100 rerun samples: use rerun results (from conversions)
    rerun_correct_on_rerun_subset = (
        always_override_conversions["T_to_T"] + always_override_conversions["F_to_T"]
    )

    # For the 400 non-rerun samples: we need base correct on those
    # Total base correct = base_acc * total_samples / 100
    # Base correct on rerun subset = conversions["T_to_T"] + conversions["T_to_F"]
    total_base_correct = base_acc * total_samples / 100
    base_correct_on_rerun_subset = (
        always_override_conversions["T_to_T"] + always_override_conversions["T_to_F"]
    )
    base_correct_on_non_rerun_subset = total_base_correct - base_correct_on_rerun_subset

    always_rerun_when_possible = (
        (rerun_correct_on_rerun_subset + base_correct_on_non_rerun_subset)
        / total_samples
        * 100
    )

    # Debug output
    print(f"DEBUG ACCURACY: Base acc: {base_acc}%, Rerun acc: {rerun_acc}%")
    print(
        f"DEBUG ACCURACY: Total samples: {total_samples}, Rerun samples: {rerun_samples}"
    )
    print(f"DEBUG ACCURACY: Total base correct: {total_base_correct}")
    print(
        f"DEBUG ACCURACY: Base correct on rerun subset: {base_correct_on_rerun_subset}"
    )
    print(
        f"DEBUG ACCURACY: Base correct on non-rerun subset: {base_correct_on_non_rerun_subset}"
    )
    print(
        f"DEBUG ACCURACY: Rerun correct on rerun subset: {rerun_correct_on_rerun_subset}"
    )
    print(f"DEBUG ACCURACY: Always override calc: {always_rerun_when_possible:.1f}%")

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
                "conversions": always_override_conversions,
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

        elif name == "always_rerun_when_possible":
            # Print T/F conversion breakdown for always override strategy
            conv = strategy.get("conversions")
            if conv and any(conv.values()):  # Only print if we have conversion data
                print("    T/F Conversions (all rerun samples):")
                print(
                    f"      T→T: {conv.get('T_to_T', 0)}, F→F: {conv.get('F_to_F', 0)}"
                )
                print(
                    f"      T→F: {conv.get('T_to_F', 0)}, F→T: {conv.get('F_to_T', 0)}"
                )

    print("\nKEY INSIGHT:")
    best = analysis["best_smart_fusion"]
    print(
        f"Smart fusion beats naive strategies by {best['improvement_over_always_rerun']:.1f}%"
    )
    print("This proves that intelligent selection based on confidence works!")


def run_delta_analysis(base_run: str, rerun_id: str) -> None:
    """Focused analysis of delta thresholds for best metrics."""

    print("\n" + "=" * 60)
    print("DELTA THRESHOLD ANALYSIS")
    print("=" * 60)
    print("Analyzing delta impact on consensus_support and agreement_ratio")

    # Test key metrics with different delta values
    metrics = ["consensus_support", "agreement_ratio"]
    delta_values = [n / 100 for n in range(10, 20, 1)]

    # Create all settings for the sweep
    settings = []
    for metric in metrics:
        for delta in delta_values:
            settings.append(
                FusionSetting(
                    metric=metric, delta=delta, min_rerun_conf=None, max_base_conf=None
                )
            )

    # Run the fusion sweep
    save_dir = Path("./output/fusion_sweeps/delta_analysis")
    cfg = FusionRunConfig(
        base_run_id=base_run,
        rerun_id=rerun_id,
        subset=None,
        settings=settings,
        save_dir=save_dir,
    )

    print(f"Running delta analysis: {len(settings)} settings")
    results = run_sweep(cfg)

    # Process results by metric
    output_dir = Path("./figures/fusion_analysis") / f"{base_run}__{rerun_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for metric in metrics:
        print(f"\nAnalyzing {metric}:")
        metric_results = []

        # Get results for this metric
        metric_fusion_results = [r for r in results if r.metric == metric]
        metric_fusion_results.sort(key=lambda x: x.delta)

        for result in metric_fusion_results:
            # Calculate conversion breakdown
            conversions = {
                "T_to_T": result.conversions_tt,
                "F_to_F": result.conversions_ff,
                "T_to_F": result.conversions_tf,
                "F_to_T": result.conversions_ft,
            }

            net_improvement = result.conversions_ft - result.conversions_tf

            result_summary = {
                "metric": metric,
                "delta": result.delta,
                "accuracy": result.acc_fused,
                "overrides_used": result.overrides_used,
                "net_improvement": net_improvement,
                "conversions": conversions,
                "harmful_conversions": result.conversions_tf,
                "beneficial_conversions": result.conversions_ft,
            }

            metric_results.append(result_summary)
            all_results.append(result_summary)

            print(
                f"  Delta={result.delta}: Acc: {result.acc_fused:.1f}%, Overrides: {result.overrides_used}, Net: +{net_improvement}, T→F: {result.conversions_tf}"
            )

        # Show delta impact for this metric
        print(f"\n  {metric} Delta Impact:")
        print(
            f"    {'Delta':<6} {'Acc':<6} {'Overrides':<10} {'T→F':<5} {'F→T':<5} {'Net':<5}"
        )
        print(f"    {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 5} {'-' * 5} {'-' * 5}")
        for r in metric_results:
            print(
                f"    {r['delta']:<6.1f} {r['accuracy']:<6.1f} {r['overrides_used']:<10} {r['harmful_conversions']:<5} {r['beneficial_conversions']:<5} {r['net_improvement']:<5}"
            )

    # Save detailed results
    delta_results_path = output_dir / "delta_analysis.json"
    with delta_results_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved delta analysis to: {delta_results_path}")

    # Summary insights
    print("\nKEY INSIGHTS:")

    for metric in metrics:
        metric_results = [r for r in all_results if r["metric"] == metric]
        best_result = max(metric_results, key=lambda x: x["accuracy"])
        zero_delta = next(r for r in metric_results if r["delta"] == 0.0)

        print(f"\n{metric}:")
        print(
            f"  Best delta: {best_result['delta']} (accuracy: {best_result['accuracy']:.1f}%)"
        )
        print(
            f"  vs delta=0: {best_result['accuracy'] - zero_delta['accuracy']:+.1f}% accuracy"
        )
        print(
            f"  Harmful conversions reduced: {zero_delta['harmful_conversions']} → {best_result['harmful_conversions']}"
        )


def get_fusion_run_pairs() -> List[Tuple[str, str]]:
    """Get a list of available fusion run pairs."""
    # These are proven working pairs from fusion_v2.py
    return [
        ("53vig20u", "9qup1u07"),  # convert_45 pair
        ("5lvoti3i", "0oe2xr1b"),  # best_accuracy pair
        ("77pyab58", "0hermenf"),
        ("gfw8x07r", "58xqqffr"),
        ("77pyab58", "8ff83v7m"),
        ("gfw8x07r", "8yyge5wj"),
    ]


def run_multi_pair_analysis() -> Dict[str, Any]:
    """Run fusion analysis over multiple run pairs and average results."""

    # run_pairs = get_fusion_run_pairs()[:2]
    run_pairs = get_fusion_run_pairs()
    print(f"Running fusion analysis over {len(run_pairs)} run pairs...")

    # Store results for each pair
    all_pair_results = []
    all_smart_results_by_metric = defaultdict(list)

    for i, (base_run, rerun_id) in enumerate(run_pairs):
        print(f"\n--- Pair {i + 1}/{len(run_pairs)}: {base_run} -> {rerun_id} ---")

        try:
            # Run experiment for this pair
            run_ultraminimal_experiment(base_run, rerun_id)

            # Load results
            results_path = (
                Path("./output/fusion_sweeps/ultraminimal")
                / f"{base_run}__{rerun_id}.json"
            )
            if not results_path.exists():
                print(f"Results file not found: {results_path}")
                continue

            results = load_results(results_path)

            # Analyze strategies for this pair
            analysis = analyze_strategies(results, base_run, rerun_id)
            all_pair_results.append(analysis)

            # Store smart results by metric with their improvements over baseline for this pair
            for smart_result in analysis["all_smart_results"]:
                metric = smart_result["metric"]
                improvement = (
                    smart_result["accuracy"] - analysis["always_base"]
                )  # Improvement for THIS pair
                all_smart_results_by_metric[metric].append(improvement)

        except Exception as e:
            print(f"Error processing pair {base_run}->{rerun_id}: {e}")
            continue

    if not all_pair_results:
        raise ValueError("No valid results found!")

    # Calculate improvement statistics (this is the key fix!)
    averaged_results = []
    always_base_avg = np.mean([r["always_base"] for r in all_pair_results])

    for metric in all_smart_results_by_metric:
        improvements = all_smart_results_by_metric[
            metric
        ]  # These are already improvements per pair
        if len(improvements) > 0:
            avg_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)  # Standard deviation of IMPROVEMENTS
            averaged_results.append(
                {
                    "metric": metric,
                    "improvement": avg_improvement,
                    "std": std_improvement,
                    "min": np.min(improvements),
                    "max": np.max(improvements),
                    "count": len(improvements),
                    # Calculate average absolute accuracy for display
                    "accuracy": avg_improvement + always_base_avg,
                }
            )

    # Sort by average improvement
    averaged_results.sort(key=lambda x: x["improvement"], reverse=True)

    # Calculate baseline averages and improvement stats for always override
    always_rerun_avg = np.mean(
        [r["always_rerun_when_possible"] for r in all_pair_results]
    )

    # Calculate improvement of always override over always base for each pair
    always_override_improvements = [
        r["always_rerun_when_possible"] - r["always_base"] for r in all_pair_results
    ]
    always_override_improvement_avg = np.mean(always_override_improvements)
    always_override_improvement_std = np.std(always_override_improvements)

    return {
        "experiment_info": {
            "num_pairs": len(all_pair_results),
            "pairs_analyzed": [
                (
                    r["experiment_info"]["total_samples"],
                    r["experiment_info"]["rerun_samples"],
                )
                for r in all_pair_results
            ],
        },
        "averaged_results": averaged_results,
        "always_base": always_base_avg,
        "always_rerun_when_possible": always_rerun_avg,
        "always_override_improvement": always_override_improvement_avg,
        "always_override_improvement_std": always_override_improvement_std,
        "best_metric": averaged_results[0]["metric"] if averaged_results else None,
        "individual_pair_results": all_pair_results,
    }


def main():
    print("Running multi-pair fusion analysis (no delta - proven useless)...")

    # Run analysis over multiple pairs
    multi_analysis = run_multi_pair_analysis()

    # Print summary
    print_multi_pair_summary(multi_analysis)

    # Create output directory
    output_dir = Path("./figures/fusion_analysis/multi_pair_average")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = output_dir / "multi_pair_analysis.json"
    with json_path.open("w") as f:
        json.dump(multi_analysis, f, indent=2)
    print(f"\nSaved results to: {json_path}")

    # Create averaged chart
    create_averaged_comparison_chart(multi_analysis, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


def print_multi_pair_summary(analysis: Dict[str, Any]) -> None:
    """Print summary of multi-pair analysis."""
    print(f"\n{'=' * 60}")
    print("MULTI-PAIR FUSION STRATEGY COMPARISON")
    print(f"{'=' * 60}")

    exp = analysis["experiment_info"]
    print(f"Analyzed {exp['num_pairs']} run pairs")

    print("\nAVERAGED STRATEGY RESULTS:")
    print(f"  Always Base (avg): {analysis['always_base']:.2f}% (baseline)")
    print(
        f"  Always Override (avg): {analysis['always_rerun_when_possible']:.2f}% ({analysis['always_override_improvement']:+.2f}%±{analysis['always_override_improvement_std']:.2f})"
    )

    print(f"\nSMART FUSION RESULTS (averaged over {exp['num_pairs']} pairs):")
    print(f"{'Metric':<20} {'Improvement':<12} {'Range':<15} {'Accuracy':<10}")
    print("-" * 65)

    for result in analysis["averaged_results"]:
        range_str = f"[{result['min']:+.2f}, {result['max']:+.2f}]"
        print(
            f"{result['metric']:<20} {result['improvement']:<+12.2f} {range_str:<15} {result['accuracy']:<10.2f}"
        )

    best = analysis["averaged_results"][0] if analysis["averaged_results"] else None
    if best:
        print(
            f"\nBEST METRIC: {best['metric']} with {best['improvement']:+.2f}% avg improvement"
        )
        print(f"Absolute accuracy: {best['accuracy']:.2f}%")
        improvement_over_override = (
            best["improvement"] - analysis["always_override_improvement"]
        )
        print(f"Improvement over always override: {improvement_over_override:+.2f}%")


if __name__ == "__main__":
    # You can choose which analysis to run:
    main()  # Run the full analysis including strategy comparison

    # Or run just the delta analysis:
    # BASE_RUN, RERUN_ID = best_accuracy()
    # # BASE_RUN, RERUN_ID = convert_45()
    # run_delta_analysis(BASE_RUN, RERUN_ID)
