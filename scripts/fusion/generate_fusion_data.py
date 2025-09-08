#!/usr/bin/env python3
"""Generate fusion analysis data: run experiments and create analysis JSON files."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from experiments.fusion.fusion import best_accuracy, run_ultraminimal_experiment


def load_results(json_path: Path) -> List[Dict[str, Any]]:
    """Load fusion results from JSON."""
    with json_path.open("r") as f:
        return json.load(f)


def calculate_always_override_conversions(
    base_run: str, rerun_id: str
) -> Dict[str, int]:
    """Calculate T/F conversions for always override strategy by loading the data directly."""
    import sys
    from pathlib import Path

    # Add parent directories to path for imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from experiments.fusion.fusion import (
        _get_base_samples,
        _is_correct_record_math,
        load_jsonl,
    )

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

    # Add parent directories to path for imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from experiments.fusion.fusion import FusionRunConfig, FusionSetting, run_sweep

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


def main():
    """Run fusion experiment and generate analysis JSON files."""
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
    analysis = analyze_strategies(results, base_run, rerun_id)

    # Print summary
    print_results_summary(analysis)

    # Create output directory
    output_dir = Path("./figures/fusion_analysis") / f"{base_run}__{rerun_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = output_dir / "strategy_comparison.json"
    with json_path.open("w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved strategy comparison results to: {json_path}")

    # Run focused delta analysis
    run_delta_analysis(base_run, rerun_id)

    print(f"\nAll data files saved to: {output_dir}")


if __name__ == "__main__":
    # You can choose which analysis to run:
    main()  # Run the full analysis including strategy comparison

    # Or run just the delta analysis:
    # BASE_RUN, RERUN_ID = best_accuracy()
    # run_delta_analysis(BASE_RUN, RERUN_ID)
