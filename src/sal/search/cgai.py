#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams  # type: ignore

from sal.config import ExperimentConfig
from sal.models.reward_models import PRM
from sal.utils.math import memoized_canonical_form
from sal.utils.qwen_math_parser import extract_answer
from sal.utils.score import aggregate_scores


def _apply_chat_template(
    llm: LLM, system_prompt: str, prompts: List[str], custom_template: Optional[str]
) -> List[str]:
    tokenizer = llm.get_tokenizer()
    if custom_template is not None:
        tokenizer.chat_template = custom_template
    convs = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        for p in prompts
    ]
    return tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )


def _generate_n(
    llm: LLM, prompts: List[str], n: int, sampling: SamplingParams
) -> Tuple[List[List[str]], List[List[int]]]:
    # Duplicate prompts to generate n completions per prompt with continuous batching
    duplicated = [c for conv in prompts for c in [conv] * n]
    responses = llm.generate(duplicated, sampling_params=sampling, use_tqdm=True)
    if len(responses) != len(prompts) * n:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(prompts) * n}"
        )

    completions: List[List[str]] = [[] for _ in range(len(prompts))]
    completion_tokens: List[List[int]] = [[] for _ in range(len(prompts))]
    for i in range(len(prompts)):
        rng = slice(i * n, (i + 1) * n)
        completions[i] = [out.text for r in responses[rng] for out in r.outputs]
        completion_tokens[i] = [
            len(out.token_ids) for r in responses[rng] for out in r.outputs
        ]

    # Sanity check
    for c in completions:
        if len(c) != n:
            raise ValueError(f"Generated {len(c)} completions instead of {n}")

    return completions, completion_tokens


def _compute_metrics(answers: List[str], agg_scores: List[float]) -> Dict[str, float]:
    n = len(answers)
    if n == 0:
        return {"agreement_ratio": 0.0, "entropy_freq": 0.0, "consensus_support": 0.0}

    # Group by canonical form
    count_by_canon: Dict[str, int] = defaultdict(int)
    sumscore_by_canon: Dict[str, float] = defaultdict(float)
    for ans, s in zip(answers, agg_scores):
        canon = memoized_canonical_form(ans)
        count_by_canon[canon] += 1
        sumscore_by_canon[canon] += float(s)

    counts = list(count_by_canon.values())
    max_count = max(counts)
    agreement_ratio = float(max_count) / float(n)

    k = len(counts)
    if k <= 1:
        entropy_freq = 0.0
    else:
        probs = [c / float(n) for c in counts]
        h = -sum(p * math.log(p + 1e-12) for p in probs)
        max_h = math.log(k)
        entropy_freq = float(h / max_h) if max_h > 0 else 0.0

    total_score = sum(sumscore_by_canon.values())
    if total_score > 0:
        consensus_support = max(sumscore_by_canon.values()) / total_score
    else:
        consensus_support = agreement_ratio

    return {
        "agreement_ratio": float(agreement_ratio),
        "entropy_freq": float(entropy_freq),
        "consensus_support": float(consensus_support),
    }


def _should_select(
    metrics: Dict[str, float], thresholds: Dict[str, Tuple[str, float]]
) -> bool:
    # Disjunctive selection: select if ANY threshold condition is met
    for key, (op, thr) in thresholds.items():
        if key not in metrics:
            continue
        val = metrics[key]
        if op == "<=":
            if val <= thr:
                return True
        elif op == ">=":
            if val >= thr:
                return True
        else:
            raise ValueError(f"Unsupported operator in thresholds: {op}")
    return False


def cgai(
    x,
    experiment_config: ExperimentConfig,
    prm: PRM,
    llm: Optional[LLM] = None,
    draft_llm: Optional[LLM] = None,
    target_llm: Optional[LLM] = None,
):
    # Stage 1: Initial inference with diversity (Best-of-N on baseline llm)
    initial_llm = draft_llm if draft_llm is not None else llm
    if initial_llm is None:
        raise ValueError("No LLM provided for CGAI initial inference")

    prompts = x["problem"]
    maybe_idx: Optional[List[int]] = x.get("idx")
    templated = _apply_chat_template(
        initial_llm,
        experiment_config.system_prompt,
        prompts,
        experiment_config.custom_chat_template,
    )

    sampling_params = SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=experiment_config.search_config.max_tokens,
        top_p=experiment_config.search_config.top_p,
        n=1,
    )

    n = experiment_config.search_config.n
    completions_base, tokens_base = _generate_n(
        initial_llm, templated, n, sampling_params
    )

    # PRM scoring for base
    scores_base = prm.score(prompts, completions_base)
    agg_scores_base: List[List[float]] = [
        [
            aggregate_scores(s, experiment_config.search_config.agg_strategy)
            for s in per_ex
        ]
        for per_ex in scores_base
    ]

    # Compute answers and metrics, and determine base predictions
    answers_base: List[List[str]] = [
        [extract_answer(c, "math") for c in comp_list] for comp_list in completions_base
    ]
    metrics_base: List[Dict[str, float]] = [
        _compute_metrics(ans_list, agg_list)
        for ans_list, agg_list in zip(answers_base, agg_scores_base)
    ]
    base_pred: List[str] = [
        comp_list[int(max(range(len(aggs)), key=lambda j: aggs[j]))]
        for comp_list, aggs in zip(completions_base, agg_scores_base)
    ]

    # Stage 2: Selection based on thresholds
    thresholds = experiment_config.confidence_selection.thresholds
    selected: List[bool] = [_should_select(m, thresholds) for m in metrics_base]
    for i, is_sel in enumerate(selected):
        if is_sel:
            idx_str = f"idx={maybe_idx[i]}" if maybe_idx is not None else f"i={i}"
            m = metrics_base[i]
            thr_str = ", ".join(
                [f"{k} {op} {thr}" for k, (op, thr) in thresholds.items()]
            )
            met_str = (
                f"agreement_ratio={m['agreement_ratio']:.3f}, "
                f"entropy_freq={m['entropy_freq']:.3f}, "
                f"consensus_support={m['consensus_support']:.3f}"
            )
            tqdm.write(f"CGAI select {idx_str}: {met_str} | thresholds: {thr_str}")

    # Stage 3: Adaptive recomputation (hyperparameter scaling and optional model scaling)
    recompute_llm = target_llm if target_llm is not None else llm
    recompute_multiplier = max(
        1, experiment_config.confidence_selection.recompute_n_multiplier
    )
    n_recompute = n * recompute_multiplier

    recomputed_indices = [i for i, s in enumerate(selected) if s]
    completions_new: Dict[int, List[str]] = {}
    tokens_new: Dict[int, List[int]] = {}
    scores_new: Dict[int, List[List[float]]] = {}
    agg_scores_new: Dict[int, List[float]] = {}
    metrics_new: Dict[int, Dict[str, float]] = {}
    new_pred: Dict[int, str] = {}

    if len(recomputed_indices) > 0 and recompute_llm is not None:
        subset_prompts = [prompts[i] for i in recomputed_indices]
        templated_subset = _apply_chat_template(
            recompute_llm,
            experiment_config.system_prompt,
            subset_prompts,
            experiment_config.custom_chat_template,
        )
        completions_sub, tokens_sub = _generate_n(
            recompute_llm, templated_subset, n_recompute, sampling_params
        )
        scores_sub = prm.score(subset_prompts, completions_sub)
        agg_scores_sub: List[List[float]] = [
            [
                aggregate_scores(s, experiment_config.search_config.agg_strategy)
                for s in per_ex
            ]
            for per_ex in scores_sub
        ]
        answers_sub: List[List[str]] = [
            [extract_answer(c, "math") for c in comp_list]
            for comp_list in completions_sub
        ]

        for local_idx, global_idx in enumerate(recomputed_indices):
            completions_new[global_idx] = completions_sub[local_idx]
            tokens_new[global_idx] = tokens_sub[local_idx]
            scores_new[global_idx] = scores_sub[local_idx]
            agg_scores_new[global_idx] = agg_scores_sub[local_idx]
            metrics_new[global_idx] = _compute_metrics(
                answers_sub[local_idx], agg_scores_sub[local_idx]
            )
            # Choose recompute winner by aggregated score
            aggs = agg_scores_sub[local_idx]
            best_j = int(max(range(len(aggs)), key=lambda j: aggs[j]))
            new_pred[global_idx] = completions_sub[local_idx][best_j]
            # Log recompute metrics comparison
            idx_str = (
                f"idx={maybe_idx[global_idx]}"
                if maybe_idx is not None
                else f"i={global_idx}"
            )
            mb = metrics_base[global_idx]["consensus_support"]
            mn = metrics_new[global_idx]["consensus_support"]
            tqdm.write(
                f"CGAI recompute {idx_str}: consensus_support base={mb:.3f} -> new={mn:.3f} (n={n} -> {n_recompute})"
            )

    # Stage 4: Confidence-based fusion
    final_pred: List[str] = []
    consensus_base = [m["consensus_support"] for m in metrics_base]
    for i in range(len(prompts)):
        if i in metrics_new:
            if metrics_new[i]["consensus_support"] > consensus_base[i]:
                final_pred.append(new_pred[i])
            else:
                final_pred.append(base_pred[i])
        else:
            final_pred.append(base_pred[i])

    # Populate output record
    x["completions"] = completions_base
    x["scores"] = scores_base
    x["completion_tokens"] = tokens_base
    x["pred"] = final_pred

    # Optional diagnostics
    x["cgai_selected"] = selected
    x["cgai_metrics_base"] = metrics_base
    x["cgai_metrics_new"] = [metrics_new.get(i) for i in range(len(prompts))]
    x["cgai_consensus_base"] = consensus_base
    x["cgai_consensus_new"] = [
        metrics_new[i]["consensus_support"] if i in metrics_new else None
        for i in range(len(prompts))
    ]

    return x
