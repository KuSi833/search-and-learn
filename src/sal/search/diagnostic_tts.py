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

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

from sal.config import ExperimentConfig
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.perf_counter() * 1000)


def _normalise_text(text: str) -> str:
    # Lightweight normalisation for overlap checks
    return " ".join(text.strip().split())


def _summarise_scores(
    score_vecs: List[List[float]], agg_strategy: Literal["min", "prod", "last"]
) -> Tuple[List[float], float, float]:
    """Return per-candidate aggregates, variance across aggregates, and top-2 margin.

    - score_vecs: list of PRM score vectors, one per candidate
    - returns: (aggregates, variance_of_aggregates, top_minus_second_margin)
    """
    if len(score_vecs) == 0:
        return [], 0.0, 0.0
    aggregates = [aggregate_scores(s, agg_strategy) for s in score_vecs]
    if len(aggregates) == 1:
        return aggregates, 0.0, aggregates[0]
    variance = float(np.var(aggregates))
    top_two = sorted(aggregates, reverse=True)[:2]
    margin = float(top_two[0] - top_two[1])
    return aggregates, variance, margin


def _build_sampling_params(
    experiment_config: ExperimentConfig, final_iter: bool
) -> SamplingParams:
    if final_iter:
        return SamplingParams(
            temperature=experiment_config.search_config.temperature,
            max_tokens=experiment_config.search_config.max_tokens,
            top_p=experiment_config.search_config.top_p,
            n=1,
        )
    return SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=experiment_config.search_config.max_tokens,
        top_p=experiment_config.search_config.top_p,
        stop=["\n\n"],  # step boundary
        include_stop_str_in_output=True,
        n=1,
    )


def _apply_chat_template(
    llm: LLM,
    convs: List[List[Dict[str, str]]],
    experiment_config: ExperimentConfig,
    add_generation_prompt: bool,
    continue_final_message: bool,
) -> List[str]:
    tokenizer = llm.get_tokenizer()
    if experiment_config.custom_chat_template is not None:
        tokenizer.chat_template = experiment_config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tokenize=False,
    )
    return templated_convs


def diagnostic_tts(
    examples: Dict[str, List[Any]],
    experiment_config: ExperimentConfig,
    draft_llm: LLM,
    target_llm: LLM,
    prm: PRM,
    output_dir: str,
) -> Dict[str, Any]:
    """Diagnostic TTC that mirrors a standard beam search but logs dual-model telemetry.

    Selection policy: uses draft model candidates only. Target generations are counterfactuals for telemetry.

    Writes one JSONL line per problem with rich per-iteration telemetry, including latency measurements for
    generation and scoring.
    """

    problems: List[str] = examples["problem"]
    levels: List[Any] = examples.get("level", [None] * len(problems))

    beam_width = experiment_config.beam_search_config.beam_width
    if experiment_config.search_config.n % beam_width != 0:
        raise ValueError(
            "search_config.n must be a multiple of beam_search_config.beam_width"
        )
    n_beams = experiment_config.search_config.n // beam_width
    num_iterations = experiment_config.beam_search_config.num_iterations
    lookahead = experiment_config.beam_search_config.lookahead

    # Output directory handling
    base_dir = Path(output_dir)
    diag_dir = base_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = diag_dir / "results.jsonl"

    # Config is tracked externally (e.g., Weights & Biases). No local config.json persisted here.

    results: Dict[str, Any] = {
        "completions": [],
        "pred": [],
        "completion_tokens": [],
        "scores": [],
    }

    # Append mode so multiple runs/shards can add lines without truncation
    with open(jsonl_path, "a", encoding="utf-8") as writer:
        for problem_idx, (prompt, level) in enumerate(
            tqdm(
                zip(problems, levels, strict=True),
                total=len(problems),
                desc="Diagnostic TTC problems",
            )
        ):
            # Initialise beams for a single problem
            beams: List[Beam] = []
            for i in range(n_beams):
                beams.append(
                    Beam(
                        prompt=prompt,
                        index=i,
                        current_text="",
                        next_texts=None,
                        lookahead_texts=None,
                        best_scores=[0.0],
                        all_scores=[],
                        previous_text=None,
                        pruned=False,
                        stop_reasons=None,
                        history=[],
                    )
                )

            # Per-beam previous best aggregate to compute slopes
            prev_best_agg: Dict[int, Optional[float]] = {b.index: None for b in beams}

            telemetry_problem: Dict[str, Any] = {
                "problem_id": problem_idx,
                "level": level,
                "search_config": asdict(experiment_config.search_config),
                "beam_search_config": asdict(experiment_config.beam_search_config),
                "steps": [],
                "final": {},
            }

            for it in range(num_iterations):
                # Select active beams
                active_beams = [b for b in beams if not b.pruned]
                if len(active_beams) == 0:
                    break

                final_iter = it == num_iterations - 1
                sampling_params = _build_sampling_params(experiment_config, final_iter)

                convs = [
                    build_conv(
                        b.prompt, b.current_text, experiment_config.system_prompt
                    )
                    for b in active_beams
                ]
                add_generation_prompt = it == 0
                continue_final_message = it > 0

                # Template per model (tokenizers may differ)
                t0 = _now_ms()
                templated_draft = _apply_chat_template(
                    draft_llm,
                    convs,
                    experiment_config,
                    add_generation_prompt,
                    continue_final_message,
                )
                t1 = _now_ms()
                templated_target = _apply_chat_template(
                    target_llm,
                    convs,
                    experiment_config,
                    add_generation_prompt,
                    continue_final_message,
                )
                t2 = _now_ms()

                # Generation (draft)
                gen_lookahead = 0 if final_iter else lookahead
                g0 = _now_ms()
                gen_draft = generate_k_steps(
                    templated_draft,
                    gen_lookahead,
                    draft_llm,
                    sampling_params,
                    beam_width,
                )
                g1 = _now_ms()

                # Generation (target counterfactual)
                g2 = _now_ms()
                gen_target = generate_k_steps(
                    templated_target,
                    gen_lookahead,
                    target_llm,
                    sampling_params,
                    beam_width,
                )
                g3 = _now_ms()

                # Prepare PRM inputs
                prm_prompts: List[str] = []
                draft_completions: List[List[str]] = []
                target_completions: List[List[str]] = []
                # Tokenizers for counting generated tokens per candidate
                draft_tokenizer = draft_llm.get_tokenizer()
                target_tokenizer = target_llm.get_tokenizer()
                for beam_obj, gd, gt in zip(
                    active_beams, gen_draft, gen_target, strict=True
                ):
                    # Note: generate_k_steps returns strings for next_texts and lookahead_texts
                    # Attach to beams for reference
                    beam_obj.next_texts = gd.next_texts
                    beam_obj.stop_reasons = gd.stop_reasons
                    beam_obj.lookahead_texts = gd.lookahead_texts
                    prm_prompts.append(beam_obj.prompt)
                    prefix = beam_obj.current_text or ""
                    gd_look_list = gd.lookahead_texts or []
                    gt_look_list = gt.lookahead_texts or []
                    draft_completions.append([prefix + t for t in gd_look_list])
                    target_completions.append([prefix + t for t in gt_look_list])

                # PRM scoring (draft)
                s0 = _now_ms()
                draft_scores = prm.score(prm_prompts, draft_completions)
                s1 = _now_ms()

                # PRM scoring (target)
                s2 = _now_ms()
                target_scores = prm.score(prm_prompts, target_completions)
                s3 = _now_ms()

                # Selection based on draft only, and telemetry assembly
                step_record: Dict[str, Any] = {
                    "iteration": it,
                    "latency_ms": {
                        "templating_draft": t1 - t0,
                        "templating_target": t2 - t1,
                        "generation_draft": g1 - g0,
                        "generation_target": g3 - g2,
                        "prm_scoring_draft": s1 - s0,
                        "prm_scoring_target": s3 - s2,
                    },
                    "beam_states": [],
                }

                for local_idx, (beam_obj, gd, gt, d_scores, t_scores) in enumerate(
                    zip(
                        active_beams,
                        gen_draft,
                        gen_target,
                        draft_scores,
                        target_scores,
                        strict=True,
                    )
                ):
                    gd_next = gd.next_texts or []
                    gd_look = gd.lookahead_texts or []
                    gd_stop = gd.stop_reasons or []
                    gt_next = gt.next_texts or []
                    gt_look = gt.lookahead_texts or []
                    gt_stop = gt.stop_reasons or []
                    # Aggregates and summaries
                    d_aggs, d_var, d_margin = _summarise_scores(
                        cast(List[List[float]], d_scores),
                        experiment_config.search_config.agg_strategy,
                    )
                    t_aggs, t_var, t_margin = _summarise_scores(
                        cast(List[List[float]], t_scores),
                        experiment_config.search_config.agg_strategy,
                    )
                    d_best_idx = int(np.argmax(d_aggs)) if len(d_aggs) > 0 else 0
                    t_best_idx = int(np.argmax(t_aggs)) if len(t_aggs) > 0 else 0

                    prev_agg = prev_best_agg.get(beam_obj.index)
                    d_best_agg = float(d_aggs[d_best_idx]) if len(d_aggs) > 0 else 0.0
                    slope = None if prev_agg is None else float(d_best_agg - prev_agg)
                    prev_best_agg[beam_obj.index] = d_best_agg

                    # Diversity and overlap proxies
                    draft_texts_norm = [_normalise_text(t) for t in gd_next]
                    target_texts_norm = [_normalise_text(t) for t in gt_next]
                    overlap = len(
                        set(draft_texts_norm).intersection(set(target_texts_norm))
                    )
                    overlap_jaccard = overlap / max(
                        1, len(set(draft_texts_norm).union(set(target_texts_norm)))
                    )
                    best_text_match = (
                        1
                        if (
                            len(gd_next) > 0
                            and len(gt_next) > 0
                            and _normalise_text(gd_next[d_best_idx])
                            == _normalise_text(gt_next[t_best_idx])
                        )
                        else 0
                    )

                    beam_state = {
                        "beam_index": beam_obj.index,
                        "context_text": beam_obj.current_text,
                        "draft_candidates": [
                            {
                                "first_step_text": gd_next[j],
                                "lookahead_text": gd_look[j],
                                "stop_reason": gd_stop[j],
                                "prm_agg": float(d_aggs[j])
                                if j < len(d_aggs)
                                else None,
                                # Generated token count (first step + lookahead) using draft tokenizer
                                "tokens": int(len(draft_tokenizer.encode(gd_look[j])))
                                if j < len(gd_look)
                                else 0,
                            }
                            for j in range(len(gd_next))
                        ],
                        "target_candidates": [
                            {
                                "first_step_text": gt_next[j],
                                "lookahead_text": gt_look[j],
                                "stop_reason": gt_stop[j],
                                "prm_agg": float(t_aggs[j])
                                if j < len(t_aggs)
                                else None,
                                # Generated token count using target tokenizer
                                "tokens": int(len(target_tokenizer.encode(gt_look[j])))
                                if j < len(gt_look)
                                else 0,
                            }
                            for j in range(len(gt_next))
                        ],
                        "draft_summary": {
                            "best_idx": d_best_idx,
                            "best_prm_agg": d_best_agg,
                            "variance": d_var,
                            "top2_margin": d_margin,
                            "slope_since_prev": slope,
                        },
                        "target_summary": {
                            "best_idx": t_best_idx,
                            "best_prm_agg": float(t_aggs[t_best_idx])
                            if len(t_aggs) > 0
                            else 0.0,
                            "variance": t_var,
                            "top2_margin": t_margin,
                        },
                        "cross_model": {
                            "best_prm_delta": float(
                                (t_aggs[t_best_idx] if len(t_aggs) > 0 else 0.0)
                                - d_best_agg
                            ),
                            "overlap_jaccard": float(overlap_jaccard),
                            "best_text_match": int(best_text_match),
                        },
                    }

                    step_record["beam_states"].append(beam_state)

                    # Selection and beam update (draft only)
                    if len(d_aggs) > 0:
                        best_idx = d_best_idx
                    else:
                        best_idx = 0
                    beam_obj.previous_text = beam_obj.current_text
                    chosen_text = gd_next[best_idx] if len(gd_next) > 0 else ""
                    beam_obj.current_text = (beam_obj.current_text or "") + chosen_text
                    beam_obj.history.append(chosen_text)
                    beam_obj.best_scores = (
                        cast(List[List[float]], d_scores)[best_idx]
                        if len(d_scores) > 0
                        else [0.0]
                    )
                    beam_obj.all_scores = cast(List[List[float]], d_scores)
                    if chosen_text == "" or (
                        len(gd_stop) > best_idx and gd_stop[best_idx] == "EOS"
                    ):
                        beam_obj.pruned = True

                # Prune if final answer marker present
                for b in active_beams:
                    if (b.current_text or "").find("boxed{") != -1:
                        b.pruned = True

                telemetry_problem["steps"].append(step_record)

            # Build per-problem outputs similar to dvts
            # Expand last iteration texts to beam_width variants if needed
            per_beam_outputs: List[Beam] = []
            for b in beams:
                # If beam.next_texts is None (never generated), skip replication
                if not b.next_texts or not b.all_scores:
                    continue
                for j in range(beam_width):
                    j_idx = j if j < len(b.next_texts) else -1
                    next_text = b.next_texts[j_idx] if b.next_texts is not None else ""
                    scores_j = (
                        b.all_scores[j_idx] if j < len(b.all_scores) else b.best_scores
                    )
                    per_beam_outputs.append(
                        Beam(
                            prompt=b.prompt,
                            index=b.index,
                            current_text=(b.previous_text or "") + next_text,
                            next_texts=None,
                            lookahead_texts=None,
                            stop_reasons=None,
                            best_scores=scores_j,
                            all_scores=b.all_scores,
                            previous_text=b.current_text,
                            pruned=b.pruned,
                            history=b.history,
                        )
                    )

            # Summarise final prediction using best aggregate across replicated beams
            if len(per_beam_outputs) == 0:
                # Fallback: take the longest current_text across beams
                final_texts = [b.current_text for b in beams]
                final_text = (
                    max(final_texts, key=lambda x: len(x) if x is not None else 0) or ""
                )
                final_scores = [0.0]
                results["completions"].append([final_text])
                results["pred"].append(final_text)
                results["scores"].append([final_scores])
                results["completion_tokens"].append(-1)
            else:
                grouped = per_beam_outputs
                completions_texts = [b.current_text for b in grouped]
                agg_vals = [
                    aggregate_scores(
                        b.best_scores, experiment_config.search_config.agg_strategy
                    )
                    for b in grouped
                ]
                best_idx_global = int(np.argmax(agg_vals)) if len(agg_vals) > 0 else 0
                results["completions"].append(completions_texts)
                results["pred"].append(completions_texts[best_idx_global])
                results["scores"].append([b.best_scores for b in grouped])
                results["completion_tokens"].append(-1)

            telemetry_problem["final"] = {
                "chosen_text": results["pred"][-1],
                "solution_marker_present": ("boxed{" in results["pred"][-1])
                if isinstance(results["pred"][-1], str)
                else False,
            }

            # Write telemetry for this problem
            writer.write(json.dumps(telemetry_problem, ensure_ascii=False) + "\n")
            writer.flush()

    return results
