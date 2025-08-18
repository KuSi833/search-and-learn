import argparse
import json
from concurrent.futures import TimeoutError
from pathlib import Path

import numpy as np
import wandb
from datasets import Dataset
from pebble import ProcessPool
from tqdm import tqdm

from sal.evaluation.grader import math_equal_process
from sal.evaluation.parser import extract_answer_map, parse_ground_truth
from sal.evaluation.utils import load_jsonl


def detect_voting_ns(dataset):
    """Detect all n values from the dataset columns by looking for pred_naive@n pattern.
    Returns a sorted list of unique n values."""
    n_values = set()
    for col in dataset.column_names:
        if any(col.startswith(f"pred_{agg}@") for agg in ["naive", "weighted", "maj"]):
            try:
                n = int(col.split("@")[1])
                n_values.add(n)
            except (ValueError, IndexError):
                continue

    return sorted(list(n_values)) if n_values else None


def evaluate_config(params):
    n, agg, col, samples, benchmark = params
    samples_with_pred = samples.map(
        extract_answer_map,
        fn_kwargs={"data_name": benchmark, "col": col},
        desc=f"Parsing {agg}@{n} predictions",
        num_proc=4,
        load_from_cache_file=False,
    )

    params = [
        (idx, pred, gt)
        for idx, pred, gt in zip(
            samples_with_pred["idx"], samples_with_pred["pred"], samples_with_pred["gt"]
        )
    ]
    idx_sequence = [p[0] for p in params]
    # We only need overall mistakes (indices), not per-level breakdown
    with ProcessPool(max_workers=8) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        scores = []
        timeout_cnt = 0
        overall_mistakes = []
        with tqdm(total=len(params), desc=f"Evaluate {agg}@{n}") as progress_bar:
            pos = 0
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                    if pos < len(idx_sequence) and not result:
                        current_idx = idx_sequence[pos]
                        overall_mistakes.append(current_idx)
                    pos += 1
                except StopIteration:
                    break
                except TimeoutError as error:
                    timeout_cnt += 1
                    print(error)
                    if pos < len(idx_sequence):
                        current_idx = idx_sequence[pos]
                        overall_mistakes.append(current_idx)
                        scores.append(False)
                        pos += 1
                except Exception as error:
                    timeout_cnt += 1
                    print(error)
                    if pos < len(idx_sequence):
                        current_idx = idx_sequence[pos]
                        overall_mistakes.append(current_idx)
                        scores.append(False)
                        pos += 1

    level_scores = {}
    level_counts = {}
    total_samples = len(scores)
    overall_acc = sum(scores) / total_samples if total_samples > 0 else 0.0

    # If level column exists, calculate level-specific scores
    if "level" in samples_with_pred.column_names:
        level_params = {}
        for idx, pred, gt, level in zip(
            samples_with_pred["idx"],
            samples_with_pred["pred"],
            samples_with_pred["gt"],
            samples_with_pred["level"],
        ):
            if level not in level_params:
                level_params[level] = []
            level_params[level].append((idx, pred, gt))

        # Evaluate each level separately
        for level, level_param_list in level_params.items():
            with ProcessPool(max_workers=8) as pool:
                future = pool.map(math_equal_process, level_param_list, timeout=3)
                iterator = future.result()
                level_scores_list = []
                with tqdm(
                    total=len(level_param_list),
                    desc=f"Evaluate {agg}@{n} level {level}",
                ) as progress_bar:
                    while True:
                        try:
                            result = next(iterator)
                            level_scores_list.append(result)
                        except StopIteration:
                            break
                        except TimeoutError as error:
                            print(error)
                            level_scores_list.append(False)
                            timeout_cnt += 1
                        except Exception as error:
                            print(error)
                            exit()
                        progress_bar.update(1)
                level_scores[level] = (
                    np.mean(level_scores_list) * 100 if level_scores_list else 0.0
                )
                level_counts[level] = len(level_scores_list)

    return (n, agg, level_scores, overall_acc, timeout_cnt, overall_mistakes)


def evaluate_single_dataset(
    benchmark: str,
    dataset,
    dataset_col: str,
    output_file: Path,
    max_num_samples=None,
):
    samples = dataset

    # Sanity check we have unique number of MATH problems
    if benchmark == "math" and len(samples.unique("problem")) != len(samples):
        raise ValueError(
            f"Dataset contains duplicate math problems. Found {len(samples.unique('problem'))} unique problems out of {len(samples)} samples"
        )

    if "idx" not in samples.column_names:
        samples = samples.map(lambda x, idx: {"idx": idx}, with_indices=True)

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]

    def parse_gt(x):
        x["gt_cot"], x["gt"] = parse_ground_truth(x, benchmark)
        return x

    samples = samples.map(
        parse_gt, desc="Parsing ground truth", num_proc=4, load_from_cache_file=False
    )

    # Detect if this is a voting dataset
    voting_ns = detect_voting_ns(samples)
    if voting_ns is not None:
        # Prepare parameters for parallel evaluation
        eval_params = []
        for n in voting_ns:
            for agg in ["naive", "weighted", "maj"]:
                col = f"pred_{agg}@{n}"
                if col in samples.column_names:
                    eval_params.append((n, agg, col, samples, benchmark))

        # Run parallel evaluation
        voting_results = {}
        mistakes = {}
        total_timeout_cnt = 0
        with ProcessPool(max_workers=min(len(eval_params), 8)) as pool:
            future = pool.map(evaluate_config, eval_params)
            iterator = future.result()
            with tqdm(
                total=len(eval_params), desc="Evaluating configurations"
            ) as progress_bar:
                while True:
                    try:
                        (
                            n,
                            agg,
                            level_scores,
                            overall_acc,
                            timeout_cnt,
                            overall_mistakes,
                        ) = next(iterator)
                        if n not in voting_results:
                            voting_results[n] = {}
                        if n not in mistakes:
                            mistakes[n] = {}
                        voting_results[n][f"acc_{agg}"] = level_scores
                        voting_results[n][f"overall_acc_{agg}"] = overall_acc
                        mistakes[n][f"overall_mistakes_{agg}"] = overall_mistakes
                        total_timeout_cnt += timeout_cnt
                    except StopIteration:
                        break
                    except Exception as error:
                        print(f"Error in parallel evaluation: {str(error)}")
                        exit()
                    progress_bar.update(1)

        # Calculate average accuracy across all difficulty levels for each n and each aggregation method
        avg_acc = {}
        for n in voting_ns:
            for agg in ["naive", "weighted", "maj"]:
                if f"overall_acc_{agg}" in voting_results[n]:
                    avg_acc[f"overall_acc_{agg}_n{n}"] = voting_results[n][
                        f"overall_acc_{agg}"
                    ]

        # Save mistakes to file and upload to W&B as a file artifact
        mistakes_json = {
            "num_samples": len(samples),
            "timeout_samples": total_timeout_cnt,
            "voting_ns": voting_ns,
            "mistakes": mistakes,
        }
        mistakes_file = output_file.parent / "mistakes.json"
        mistakes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mistakes_file, "w") as f:
            json.dump(mistakes_json, f)
        try:
            if wandb.run is not None:
                wandb.run.save(str(mistakes_file))
        except Exception as error:
            print(f"W&B artifact upload failed: {error}")

        result_json = {
            "num_samples": len(samples),
            "num_scores": len(samples),
            "timeout_samples": total_timeout_cnt,
            "voting_results": voting_results,
            "voting_ns": voting_ns,
            "avg_acc": avg_acc,
        }
        if wandb.run is not None:
            wandb.log(result_json)

    else:
        # Regular evaluation
        samples = samples.map(
            extract_answer_map,
            fn_kwargs={"data_name": benchmark, "col": dataset_col},
            desc="Parsing predictions",
            num_proc=4,
            load_from_cache_file=False,
        )

        # Group samples by level
        level_params = {}
        for idx, pred, gt, level in zip(
            samples["idx"], samples["pred"], samples["gt"], samples["level"]
        ):
            if level not in level_params:
                level_params[level] = []
            level_params[level].append((idx, pred, gt))

        # Evaluate each level separately
        level_scores = {}
        timeout_cnt = 0
        overall_mistakes = []
        for level, params in level_params.items():
            with ProcessPool(max_workers=8) as pool:
                future = pool.map(math_equal_process, params, timeout=3)
                iterator = future.result()
                scores = []
                with tqdm(
                    total=len(params), desc=f"Evaluate level {level}"
                ) as progress_bar:
                    pos = 0
                    idx_sequence = [p[0] for p in params]
                    while True:
                        try:
                            result = next(iterator)
                            scores.append(result)
                            if pos < len(idx_sequence) and not result:
                                current_idx = idx_sequence[pos]
                                overall_mistakes.append(current_idx)
                            pos += 1
                        except StopIteration:
                            break
                        except TimeoutError as error:
                            print(error)
                            scores.append(False)
                            timeout_cnt += 1
                            if pos < len(idx_sequence):
                                current_idx = idx_sequence[pos]
                                overall_mistakes.append(current_idx)
                                pos += 1
                        except Exception as error:
                            print(error)
                            scores.append(False)
                            if pos < len(idx_sequence):
                                current_idx = idx_sequence[pos]
                                overall_mistakes.append(current_idx)
                                pos += 1
                        progress_bar.update(1)
                level_scores[level] = np.mean(scores) * 100 if scores else 0.0

        # Save mistakes to file and upload to W&B as a file artifact
        mistakes_json = {
            "num_samples": len(samples),
            "timeout_samples": timeout_cnt,
            "overall_mistakes": overall_mistakes,
        }
        mistakes_file = output_file.parent / "mistakes.json"
        mistakes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mistakes_file, "w") as f:
            json.dump(mistakes_json, f)
        try:
            if wandb.run is not None:
                wandb.run.save(str(mistakes_file))
        except Exception as error:
            print(f"W&B artifact upload failed: {error}")

        result_json = {
            "num_samples": len(samples),
            "num_scores": len(samples),
            "timeout_samples": timeout_cnt,
            "acc": level_scores,
        }

    print(result_json)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        json.dump(result_json, f)
        f.write("\n")

    return samples, result_json


def evaluate(
    benchmark: str,
    dataset_path: Path,
    dataset_col: str,
    output_file: Path,
    max_num_samples=None,
):
    jsonl_data = list(load_jsonl(str(dataset_path)))
    dataset = Dataset.from_list(jsonl_data)
    result_id = dataset_path.name

    samples, result = evaluate_single_dataset(
        benchmark=benchmark,
        dataset=dataset,
        dataset_col=dataset_col,
        max_num_samples=max_num_samples,
        output_file=output_file,
    )
    # Add id to result
    result["id"] = result_id
    return samples, result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="math")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_col", type=str, default="pred")
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        benchmark=args.benchmark,
        dataset_path=Path(args.dataset_path),
        dataset_col=args.dataset_col,
        max_num_samples=args.max_num_samples,
        output_file=Path(args.output_file),
    )
