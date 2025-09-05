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


def _dvts(
    batch_of_prompts: list[str], experiment_config: ExperimentConfig, llm: LLM, prm: PRM
):
    sampling_params = SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=experiment_config.search_config.max_tokens,
        top_p=experiment_config.search_config.top_p,
        stop=[
            "\n\n"
        ],  # we consider that a step in the problem is indicated by a double newline
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(experiment_config.n_beams):
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
                    completed=False,
                    completion_tokens=0,
                )
            )

    for i in tqdm(
        range(experiment_config.beam_search_config.num_iterations),
        desc="Beam search iterations",
    ):
        # generation
        gen_beams = [b for b in beams if not b.pruned]
        if len(gen_beams) == 0:
            break

        # Check token limits before generation to avoid vLLM errors
        tokenizer = llm.get_tokenizer()
        max_context_length = 4096  # vLLM limit

        # Calculate current token usage for each beam and adjust max_tokens accordingly
        remaining_tokens = []
        for beam in gen_beams:
            # Build the conversation to get accurate token count
            conv = build_conv(
                beam.prompt, beam.current_text or "", experiment_config.system_prompt
            )
            continue_final_message = i > 0
            add_generation_prompt = i == 0

            if experiment_config.custom_chat_template is not None:
                tokenizer.chat_template = experiment_config.custom_chat_template
            templated_conv = tokenizer.apply_chat_template(
                [conv],
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )[0]

            current_tokens = len(
                tokenizer.encode(templated_conv, add_special_tokens=False)
            )
            remaining = max_context_length - current_tokens - 50  # Leave some buffer
            remaining_tokens.append(max(1, remaining))

        # Use the minimum remaining tokens to ensure no beam exceeds the limit
        safe_max_tokens = min(remaining_tokens) if remaining_tokens else 1
        if safe_max_tokens <= 1:
            # If we're at the token limit, prune these beams and continue
            for beam in gen_beams:
                beam.pruned = True
            continue

        if i == experiment_config.beam_search_config.num_iterations - 1:
            # last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=experiment_config.search_config.temperature,
                max_tokens=min(
                    experiment_config.search_config.max_tokens, safe_max_tokens
                ),
                top_p=experiment_config.search_config.top_p,
                n=1,
            )
        else:
            sampling_params = SamplingParams(
                temperature=experiment_config.search_config.temperature,
                max_tokens=min(
                    experiment_config.search_config.max_tokens, safe_max_tokens
                ),
                top_p=experiment_config.search_config.top_p,
                stop=[
                    "\n\n"
                ],  # we consider that a step in the problem is indicated by a double newline
                include_stop_str_in_output=True,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, experiment_config.system_prompt)
            for b in gen_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        # TODO: set the augmented template from a file
        if experiment_config.custom_chat_template is not None:
            tokenizer.chat_template = experiment_config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = (
            0
            if i == experiment_config.beam_search_config.num_iterations - 1
            else experiment_config.beam_search_config.lookahead
        )
        gen_results = generate_k_steps(
            templated_convs,
            lookahead,
            llm,
            sampling_params,
            experiment_config.beam_search_config.beam_width,
        )

        prompts, completions = [], []
        for beam, gen_result in zip(gen_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            if (
                beam.next_texts is None
                or len(beam.next_texts)
                != experiment_config.beam_search_config.beam_width
            ):
                beam.pruned = True
                # rarely ~1/1000 the model will generate few beams than expected. #TODO: investigate why
                logger.warning(
                    f"beam {beam.index} has {len(beam.next_texts) if beam.next_texts else 0} completions"
                )
            prompts.append(beam.prompt)
            if beam.lookahead_texts is not None:
                current_text = beam.current_text or ""
                completions.append([current_text + t for t in beam.lookahead_texts])
            else:
                completions.append([beam.current_text or ""])

        # scoring and chose best generation per beam TODO: add option for selection across beams within the same prompt

        all_scores = prm.score(prompts, completions)

        for beam, scores in zip(gen_beams, all_scores, strict=True):
            if not beam.next_texts or not beam.stop_reasons:
                beam.pruned = True
                continue

            agg_scores = [
                aggregate_scores(
                    s if isinstance(s, list) else [s],
                    experiment_config.search_config.agg_strategy,
                )
                for s in scores
            ]
            best_score_ind = np.argmax(agg_scores)
            beam.all_scores = [
                [float(s)] if not isinstance(s, list) else s for s in scores
            ]
            beam.previous_text = beam.current_text
            current_text = beam.current_text or ""
            beam.current_text = current_text + beam.next_texts[best_score_ind]
            beam.history.append(beam.next_texts[best_score_ind])
            selected_score = scores[best_score_ind]
            beam.best_scores = (
                [float(selected_score)]
                if not isinstance(selected_score, list)
                else selected_score
            )

            if (
                beam.next_texts[best_score_ind] == ""
                or beam.stop_reasons[best_score_ind] == "EOS"
            ):
                # stopped on EOS, prune
                beam.pruned = True

        # filter / prune
        for beam in gen_beams:
            if beam.current_text and "boxed{" in beam.current_text:
                beam.pruned = True

    # we need to copy the results from the last iteration in to beam_width beams as otherwise we would only have n/m results
    output: list[Beam] = []
    for beam in beams:
        for i in range(experiment_config.beam_search_config.beam_width):
            previous_text = beam.previous_text or ""
            next_text = (
                beam.next_texts[i]
                if beam.next_texts and i < len(beam.next_texts)
                else ""
            )
            current_text = previous_text + next_text
            best_scores = (
                beam.all_scores[i]
                if beam.all_scores and i < len(beam.all_scores)
                else [0.0]
            )

            candidate = Beam(
                prompt=beam.prompt,
                index=beam.index,
                current_text=current_text,
                next_texts=None,
                lookahead_texts=None,
                stop_reasons=None,
                best_scores=best_scores,
                all_scores=beam.all_scores,
                previous_text=beam.current_text,
                pruned=beam.pruned,
                history=beam.history,
                completed=False,
                completion_tokens=0,
            )

            # Apply duplicate filtering if enabled
            if experiment_config.filter_duplicates:
                # Check if this current_text already exists for this prompt
                if any(
                    b.prompt == candidate.prompt
                    and b.current_text == candidate.current_text
                    for b in output
                ):
                    continue

            output.append(candidate)

    return output


def dvts(examples, experiment_config: ExperimentConfig, llm: LLM, prm: PRM):
    problems = examples["problem"]
    beam_results = _dvts(problems, experiment_config, llm, prm)

    # group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        results["completions"].append([b.current_text for b in beams])
        results["pred"].append(
            beams[
                np.argmax(
                    [
                        aggregate_scores(
                            b.best_scores, experiment_config.search_config.agg_strategy
                        )
                        for b in beams
                    ]
                )
            ].current_text
        )
        results["scores"].append([b.best_scores for b in beams])
        results["completion_tokens"].append(-1)

    # TODO: construct and store the tree

    return results
