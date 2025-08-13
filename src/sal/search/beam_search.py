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


def _beam_search(
    batch_of_prompts, config: ExperimentConfig, llm: LLM, prm: PRM
) -> list[Beam]:
    sampling_params = SamplingParams(
        temperature=config.search_config.temperature,
        max_tokens=config.search_config.max_tokens,
        top_p=config.search_config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    # Keep K beams after each expansion
    keep_k = max(1, config.search_config.n // config.beam_search_config.beam_width)

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

    for i in tqdm(
        range(config.beam_search_config.num_iterations), desc="Beam search iterations"
    ):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # No replication: we explicitly control branching with beam_width

        if i == config.beam_search_config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.search_config.temperature,
                max_tokens=config.search_config.max_tokens,
                top_p=config.search_config.top_p,
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
            config.search_config.max_tokens
            if i == config.beam_search_config.num_iterations - 1
            else config.beam_search_config.lookahead
        )

        # Expand each active beam into beam_width candidates
        gen_results = generate_k_steps(
            templated_convs,
            lookahead,
            llm,
            sampling_params,
            config.beam_search_config.beam_width,
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
        for b, score in zip(candidate_beams, scores, strict=True):
            b.all_scores = score[0]
            aggregated.append(
                aggregate_scores(b.all_scores, config.search_config.agg_strategy)
            )

        # Separate completed vs continuing
        continuing: list[tuple[float, Beam]] = []
        for b, a in zip(candidate_beams, aggregated, strict=True):
            if b.completed:
                completed_beams.append(b)
            else:
                continuing.append((a, b))

        # If enough completed, we can stop early
        if len(completed_beams) >= config.search_config.n:
            break

        # Optional deduplication
        if config.filter_duplicates:
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
        active_beams = [b for _, b in continuing[:keep_k]]

    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(
                b.all_scores, config.search_config.agg_strategy
            ),
            reverse=True,
        )[: config.search_config.n]
    else:
        completed_beams = completed_beams[: config.search_config.n]

    if len(completed_beams) != config.search_config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.search_config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.search_config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b)
            for b in (completed_beams * repeats)[: config.search_config.n]
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
            aggregate_scores(b.all_scores, experiment_config.search_config.agg_strategy)
            for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
