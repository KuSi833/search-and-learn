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
import copy
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams  # type: ignore

from sal.config import ExperimentConfig
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps

logger = logging.getLogger()


def _trim(t: str, n: int = 160) -> str:
    return (t[:n] + "…") if len(t) > n else t


def _beam_search(
    batch_of_prompts, config: ExperimentConfig, llm: LLM, prm: PRM
) -> list[Beam]:
    sampling = config.beam.sampling
    sampling_params = SamplingParams(
        temperature=sampling.temperature,
        max_tokens=sampling.max_tokens,
        top_p=sampling.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    # Keep K beams after each expansion
    keep_k = max(1, sampling.n // config.beam.beam_width)

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(keep_k):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                )
            )

    completed_beams: list[Beam] = []

    for i in tqdm(range(config.beam.num_iterations), desc="Beam search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # No replication: we explicitly control branching with beam_width

        if i == config.beam.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=sampling.temperature,
                max_tokens=sampling.max_tokens,
                top_p=sampling.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        # Determine lookahead depth for ranking-only vs final full decoding
        lookahead = (
            sampling.max_tokens
            if i == config.beam.num_iterations - 1
            else config.beam.lookahead
        )

        # Expand each active beam into beam_width candidates
        gen_results = generate_k_steps(
            templated_convs,
            lookahead,
            llm,
            sampling_params,
            config.beam.beam_width,
        )

        # Build child candidates
        candidate_beams: list[Beam] = []
        for parent_beam, gen_result in zip(active_beams, gen_results, strict=True):
            next_texts = gen_result.next_texts or []
            stop_reasons = gen_result.stop_reasons or [""] * len(next_texts)
            for j, (delta_text, stop_reason) in enumerate(
                zip(next_texts, stop_reasons, strict=True)
            ):
                new_text = parent_beam.current_text + delta_text
                child = Beam(
                    prompt=parent_beam.prompt,
                    index=j,
                    current_text=new_text,
                    next_texts=None,
                    lookahead_texts=None,
                    stop_reasons=None,
                    pruned=False,
                    completed=(stop_reason in ["EOS", "length"] or delta_text == ""),
                    history=parent_beam.history
                    + ([delta_text] if delta_text != "" else []),
                    best_scores=[],
                    all_scores=[],
                    previous_text=parent_beam.current_text,
                    completion_tokens=0,
                )
                candidate_beams.append(child)

        if len(candidate_beams) == 0:
            break

        # Score candidates using PRM
        prm_prompts = [b.prompt for b in candidate_beams]
        prm_completions = [[b.current_text] for b in candidate_beams]
        scores = prm.score(prm_prompts, prm_completions)

        # Attach scores and aggregate
        aggregated: list[float] = []
        for b, per_completion in zip(candidate_beams, scores, strict=True):
            # PRM returns list[list[float]] per question, with one entry per completion
            b.all_scores = per_completion[0]  # type: ignore[assignment]
            aggregated.append(aggregate_scores(b.all_scores, sampling.agg_strategy))

        if config.beam.debug:
            logger.debug(
                f"Iteration {i}: expanded {len(active_beams)} beams -> {len(candidate_beams)} candidates"
            )
            for parent_idx, (parent, gen_result) in enumerate(
                zip(active_beams, gen_results, strict=True)
            ):
                child_slice = candidate_beams[
                    parent_idx * config.beam.beam_width : (parent_idx + 1)
                    * config.beam.beam_width
                ]
                scores_slice = aggregated[
                    parent_idx * config.beam.beam_width : (parent_idx + 1)
                    * config.beam.beam_width
                ]
                logger.debug(
                    "Parent[%d]: '%s'",
                    parent_idx,
                    _trim(parent.current_text.replace("\n", " ")),
                )
                for k, (child, a) in enumerate(
                    zip(child_slice, scores_slice, strict=True)
                ):
                    logger.debug(
                        "  ├─ Child[%d]: agg=%.4f stop=%s text='%s'",
                        k,
                        a,
                        "completed" if child.completed else "",
                        _trim(
                            (child.current_text[len(parent.current_text) :]).replace(
                                "\n", " "
                            )
                        ),
                    )

        # Separate completed vs continuing
        continuing: list[tuple[float, Beam]] = []
        for b, a in zip(candidate_beams, aggregated, strict=True):
            if b.completed:
                completed_beams.append(b)
            else:
                continuing.append((a, b))

        # If enough completed, we can stop early
        if len(completed_beams) >= sampling.n:
            break

        # Optional deduplication
        if config.beam.filter_duplicates:
            seen = set()
            filtered: list[tuple[float, Beam]] = []
            for a, b in continuing:
                if b.current_text not in seen:
                    seen.add(b.current_text)
                    filtered.append((a, b))
            continuing = filtered

        if len(continuing) == 0:
            break

        # Select top-K beams to continue
        continuing.sort(key=lambda t: t[0], reverse=True)
        chosen = continuing[:keep_k]
        active_beams = [b for _, b in chosen]

        if config.beam.debug:
            logger.debug("Selected top-%d beams:", keep_k)
            for rank, (a, bsel) in enumerate(chosen):
                logger.debug(
                    "  #%d agg=%.4f text='%s'",
                    rank,
                    a,
                    _trim(bsel.current_text.replace("\n", " ")),
                )

    # Filter completed beams for those with top config.n scores
    if config.beam.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, sampling.agg_strategy),
            reverse=True,
        )[: sampling.n]
    else:
        completed_beams = completed_beams[: sampling.n]

    if config.beam.debug:
        logger.debug("Final completed beams (top %d):", len(completed_beams))
        for idx, b in enumerate(completed_beams):
            logger.debug(
                "  [%d] agg=%.4f text='%s'",
                idx,
                aggregate_scores(b.all_scores, sampling.agg_strategy),
                (
                    b.current_text.replace("\n", " ")[:200]
                    + ("…" if len(b.current_text) > 200 else "")
                ),
            )

    if len(completed_beams) != sampling.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (sampling.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {sampling.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: sampling.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams


def beam_search(examples, experiment_config: ExperimentConfig, llm: LLM, prm: PRM):
    problems = examples["problem"]
    beam_results = _beam_search(problems, experiment_config, llm, prm)

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, experiment_config.beam.sampling.agg_strategy)
            for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
