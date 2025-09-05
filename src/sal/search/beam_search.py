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
from typing import List

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams  #  type: ignore

from sal.config import ExperimentConfig
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps

logger = logging.getLogger()


def _beam_search(
    batch_of_prompts: List[str], config: ExperimentConfig, llm: LLM, prm: PRM
) -> List[Beam]:
    """Classical beam search, using DVTS-style expand/score/select.

    At each iteration, expand every active beam into ``beam_width`` candidates
    using ``generate_k_steps`` (optionally with lookahead). Score candidates with
    the PRM and select the top-``n`` globally across all beams.
    """

    all_outputs: List[Beam] = []

    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    num_iterations = int(config.beam_search_config.num_iterations)
    beam_width = int(config.beam_search_config.beam_width)

    for prompt in batch_of_prompts:
        # Initialise with n empty beams for this prompt
        beams: List[Beam] = [
            Beam(
                prompt=prompt,
                index=i,
                current_text="",
                next_texts=None,
                lookahead_texts=None,
                stop_reasons=None,
                best_scores=[],
                all_scores=[],
                previous_text=None,
                pruned=False,
                history=[],
                completed=False,
                completion_tokens=0,
            )
            for i in range(config.search_config.n)
        ]

        for i in tqdm(
            range(num_iterations), desc="Beam search iterations", leave=False
        ):
            # Active beams for this prompt
            gen_beams = [b for b in beams if not b.pruned and not b.completed]
            if len(gen_beams) == 0:
                break

            # Enforce per-beam completion token budget (best_of_n-style)
            max_new_tokens = int(config.search_config.max_tokens)
            for b in gen_beams:
                if b.completion_tokens >= max_new_tokens:
                    b.completed = True
            gen_beams = [b for b in gen_beams if not b.completed]
            if len(gen_beams) == 0:
                break
            remainings = [max_new_tokens - b.completion_tokens for b in gen_beams]
            step_cap = int(max(1, min(remainings)))

            # Sampling params: step-wise vs final (generate to EOS)
            if i == num_iterations - 1:
                sampling_params = SamplingParams(
                    temperature=config.search_config.temperature,
                    max_tokens=config.search_config.max_tokens,
                    top_p=config.search_config.top_p,
                    n=1,
                )
            else:
                sampling_params = SamplingParams(
                    temperature=config.search_config.temperature,
                    max_tokens=config.search_config.max_tokens,
                    top_p=config.search_config.top_p,
                    stop=["\n\n"],
                    include_stop_str_in_output=True,
                    n=1,
                )

            # Cap per-step generation to not exceed any beam's remaining budget
            sampling_params.max_tokens = int(
                min(int(sampling_params.max_tokens), step_cap)
            )

            # Build conversations and template
            convs = [
                build_conv(prompt, b.current_text, config.system_prompt)
                for b in gen_beams
            ]
            add_generation_prompt = i == 0
            continue_final_message = i > 0
            templated_convs = tokenizer.apply_chat_template(
                convs,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )

            # Lookahead for scoring
            lookahead = (
                0
                if i == num_iterations - 1
                else int(config.beam_search_config.lookahead)
            )

            gen_results = generate_k_steps(
                templated_convs,
                lookahead,
                llm,
                sampling_params,
                beam_width,
            )

            # Build candidate children for this prompt
            candidate_children: List[Beam] = []
            prm_prompts: List[str] = []
            prm_completions: List[List[str]] = []
            for parent, gen_result in zip(gen_beams, gen_results, strict=True):
                assert gen_result.next_texts is not None
                assert gen_result.lookahead_texts is not None
                assert gen_result.stop_reasons is not None
                for k in range(beam_width):
                    next_piece = gen_result.next_texts[k]
                    lookahead_piece = gen_result.lookahead_texts[k]
                    stop_reason = gen_result.stop_reasons[k] or "EOS"
                    child_text = (parent.current_text or "") + next_piece
                    # Estimate tokens generated this step and track completion tokens
                    generated_token_count = len(
                        tokenizer.encode(next_piece, add_special_tokens=False)
                    )
                    new_completion_tokens = (
                        parent.completion_tokens + generated_token_count
                    )
                    score_text = (parent.current_text or "") + lookahead_piece

                    child = Beam(
                        prompt=prompt,
                        index=k,
                        current_text=child_text,
                        next_texts=None,
                        lookahead_texts=None,
                        stop_reasons=[stop_reason],
                        best_scores=[],
                        all_scores=[],
                        previous_text=parent.current_text,
                        pruned=False,
                        history=list(parent.history) + [next_piece],
                        completed=(
                            ("boxed{" in child_text)
                            or (stop_reason == "EOS")
                            or (i == num_iterations - 1)
                            or (new_completion_tokens >= max_new_tokens)
                        ),
                        completion_tokens=new_completion_tokens,
                    )
                    candidate_children.append(child)
                    prm_prompts.append(prompt)
                    prm_completions.append([score_text])

            if len(candidate_children) == 0:
                break

            # Score candidates using PRM; aggregate and select global top-n for this prompt
            all_scores_per_child = prm.score(prm_prompts, prm_completions)

            agg_scores: List[float] = []
            for c, s in zip(candidate_children, all_scores_per_child, strict=True):
                s0 = s[0]
                if isinstance(s0, list):
                    c.best_scores = s0
                else:
                    c.best_scores = [float(s0)]
                c.all_scores = [c.best_scores]
                agg_scores.append(
                    aggregate_scores(c.best_scores, config.search_config.agg_strategy)
                )

            ranks = np.argsort(np.array(agg_scores, dtype=np.float64))[::-1]
            next_beams: List[Beam] = []
            for r in ranks:
                if len(next_beams) >= config.search_config.n:
                    break
                candidate = candidate_children[int(r)]
                if config.filter_duplicates:
                    if any(
                        b.current_text == candidate.current_text for b in next_beams
                    ):
                        continue
                next_beams.append(candidate)

            # Accumulate completed results
            for b in next_beams:
                if b.completed:
                    all_outputs.append(b)

            beams = next_beams
            if all(b.completed for b in beams):
                break

        # Ensure we include any remaining completed beams for this prompt
        all_outputs.extend([b for b in beams if b.completed])

        # If fewer than n outputs for this prompt, duplicate to reach n
        prompt_outputs = [b for b in all_outputs if b.prompt == prompt]
        if len(prompt_outputs) < config.search_config.n and len(prompt_outputs) > 0:
            repeats = (config.search_config.n // len(prompt_outputs)) + 1
            extended = (prompt_outputs * repeats)[: config.search_config.n]
            all_outputs = [b for b in all_outputs if b.prompt != prompt] + extended

    return all_outputs


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
        # b.best_scores contains the PRM scores along the trace; aggregate them
        agg_scores = [
            aggregate_scores(
                b.best_scores, experiment_config.search_config.agg_strategy
            )
            for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.best_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
