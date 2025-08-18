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
from typing import List

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams  #  type: ignore

from sal.config import ExperimentConfig
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv

logger = logging.getLogger()


def _beam_search(
    batch_of_prompts: List[str], config: ExperimentConfig, llm: LLM, prm: PRM
) -> List[Beam]:
    """Beam search with per-beam token budgeting to avoid context overflow.

    This mirrors the robust budgeting in particles.py, ensuring each beam's
    templated prompt stays within the model context by truncating assistant
    text on the left and capping per-call max_tokens.
    """

    step_params_template = SamplingParams(
        temperature=config.search_config.temperature,
        max_tokens=config.search_config.max_tokens,
        top_p=config.search_config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        seed=config.seed,
        n=config.beam_search_config.beam_width,
    )

    final_params_template = SamplingParams(
        temperature=config.search_config.temperature,
        max_tokens=config.search_config.max_tokens,
        top_p=config.search_config.top_p,
        seed=config.seed,
        n=config.beam_search_config.beam_width,
    )

    all_outputs: List[Beam] = []

    # Process prompts independently; rescore and select beams per prompt
    for prompt in batch_of_prompts:
        # Initialise beams: n identical beams with empty assistant text
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

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        num_iterations = int(config.beam_search_config.num_iterations)
        beam_width = int(config.beam_search_config.beam_width)

        for iteration in tqdm(
            range(num_iterations), desc="Beam search iterations", leave=False
        ):
            # Select active (non-pruned, non-completed) beams
            active_indices = [
                i
                for i, b in enumerate(beams)
                if (not getattr(b, "pruned", False)) and (not b.completed)
            ]
            if len(active_indices) == 0:
                break

            # On the last iteration, force completion to EOS
            use_final = iteration == num_iterations - 1
            params_template = (
                final_params_template if use_final else step_params_template
            )

            # Build conversations and flags
            convs: List[list[dict[str, str]]] = []
            is_first_step_flags: List[bool] = []
            for i_idx in active_indices:
                b = beams[i_idx]
                conv = build_conv(prompt, b.current_text or "", config.system_prompt)
                convs.append(conv)
                is_first_step_flags.append(len(b.current_text or "") == 0)

            # Determine model max context
            max_ctx = getattr(tokenizer, "model_max_length", 4096)
            try:
                max_ctx = int(max_ctx)
            except Exception:
                max_ctx = 4096
            if max_ctx is None or max_ctx > 100000:
                max_ctx = 4096
            gen_reserve = int(min(64, max(1, params_template.max_tokens)))

            # Tokenise current prompts to get accurate lengths with correct flags
            first_idxs = [j for j, f in enumerate(is_first_step_flags) if f]
            cont_idxs = [j for j, f in enumerate(is_first_step_flags) if not f]
            prompt_token_lens_map: dict[int, int] = {}
            if first_idxs:
                first_convs = [convs[j] for j in first_idxs]
                first_tok = tokenizer.apply_chat_template(
                    first_convs,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    tokenize=True,
                )
                for local_idx, ids in zip(first_idxs, first_tok, strict=True):
                    prompt_token_lens_map[local_idx] = len(ids)
            if cont_idxs:
                cont_convs = [convs[j] for j in cont_idxs]
                cont_tok = tokenizer.apply_chat_template(
                    cont_convs,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=True,
                )
                for local_idx, ids in zip(cont_idxs, cont_tok, strict=True):
                    prompt_token_lens_map[local_idx] = len(ids)

            # Truncate assistant text from the left if over budget
            for local_idx, global_idx in enumerate(active_indices):
                current_len = int(prompt_token_lens_map.get(local_idx, 0))
                if current_len <= max_ctx:
                    continue
                assistant_text = beams[global_idx].current_text or ""
                assistant_token_ids = tokenizer.encode(
                    assistant_text, add_special_tokens=False
                )
                base_len_est = max(0, current_len - len(assistant_token_ids))
                budget_for_assistant = max(0, max_ctx - gen_reserve - base_len_est)
                if budget_for_assistant <= 0:
                    beams[global_idx].current_text = ""
                    beams[global_idx].completed = True
                    continue
                if len(assistant_token_ids) > budget_for_assistant:
                    keep_ids = assistant_token_ids[-budget_for_assistant:]
                    beams[global_idx].current_text = tokenizer.decode(
                        keep_ids, skip_special_tokens=True
                    )

            # Rebuild convs and templated prompts after possible truncation
            convs = []
            is_first_step_flags = []
            for i_idx in active_indices:
                b = beams[i_idx]
                conv = build_conv(prompt, b.current_text or "", config.system_prompt)
                convs.append(conv)
                is_first_step_flags.append(len(b.current_text or "") == 0)

            first_idxs = [j for j, f in enumerate(is_first_step_flags) if f]
            cont_idxs = [j for j, f in enumerate(is_first_step_flags) if not f]
            templated_convs_map: dict[int, str] = {}
            if first_idxs:
                first_convs = [convs[j] for j in first_idxs]
                templated_first = tokenizer.apply_chat_template(
                    first_convs,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    tokenize=False,
                )
                for local_idx, text in zip(first_idxs, templated_first, strict=True):
                    templated_convs_map[local_idx] = text
            if cont_idxs:
                cont_convs = [convs[j] for j in cont_idxs]
                templated_cont = tokenizer.apply_chat_template(
                    cont_convs,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=False,
                )
                for local_idx, text in zip(cont_idxs, templated_cont, strict=True):
                    templated_convs_map[local_idx] = text
            templated_convs = [templated_convs_map[j] for j in range(len(convs))]

            # Compute accurate prompt lengths to derive per-beam max_tokens
            prompt_token_lens_map = {}
            if first_idxs:
                first_convs = [convs[j] for j in first_idxs]
                first_tok = tokenizer.apply_chat_template(
                    first_convs,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    tokenize=True,
                )
                for local_idx, ids in zip(first_idxs, first_tok, strict=True):
                    prompt_token_lens_map[local_idx] = len(ids)
            if cont_idxs:
                cont_convs = [convs[j] for j in cont_idxs]
                cont_tok = tokenizer.apply_chat_template(
                    cont_convs,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=True,
                )
                for local_idx, ids in zip(cont_idxs, cont_tok, strict=True):
                    prompt_token_lens_map[local_idx] = len(ids)

            # Generate children for each active beam with per-beam budgets
            candidate_children: List[Beam] = []
            for local_idx, conv_text in enumerate(templated_convs):
                global_idx = active_indices[local_idx]
                parent = beams[global_idx]
                per_params = copy.deepcopy(params_template)
                curr_prompt_len = int(prompt_token_lens_map.get(local_idx, 0))
                if curr_prompt_len >= max_ctx:
                    # Skip generation; mark as length-completed
                    parent.completed = True
                    continue
                available_new = max(0, max_ctx - curr_prompt_len)
                per_params.max_tokens = int(min(per_params.max_tokens, available_new))
                per_params.seed = (
                    int(config.seed) + int(iteration) * 1000003 + int(global_idx)
                )

                resp_list = llm.generate(
                    [conv_text], sampling_params=per_params, use_tqdm=False
                )
                outputs = resp_list[0].outputs
                for k in range(min(len(outputs), beam_width)):
                    gen_text = outputs[k].text
                    stop_reason = outputs[k].stop_reason or "EOS"
                    child = Beam(
                        prompt=prompt,
                        index=k,
                        current_text=(parent.current_text or "") + gen_text,
                        next_texts=None,
                        lookahead_texts=None,
                        stop_reasons=[stop_reason],
                        best_scores=[],
                        all_scores=[],
                        previous_text=parent.current_text,
                        pruned=False,
                        history=list(parent.history) + [gen_text],
                        completed=(
                            ("boxed{" in ((parent.current_text or "") + gen_text))
                            or stop_reason == "EOS"
                            or per_params.max_tokens == 0
                        ),
                        completion_tokens=-1,
                    )
                    candidate_children.append(child)

            # Score all candidates and select top-n beams for next iteration
            if len(candidate_children) == 0:
                break

            prm_prompts: List[str] = [prompt for _ in candidate_children]
            prm_completions: List[List[str]] = [
                [c.current_text or ""] for c in candidate_children
            ]
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

            # Select top-n beams globally (keep diversity filtering optional)
            ranks = np.argsort(np.array(agg_scores, dtype=np.float64))[::-1]
            next_beams: List[Beam] = []
            for r in ranks:
                if len(next_beams) >= config.search_config.n:
                    break
                candidate = candidate_children[int(r)]
                if config.filter_duplicates:
                    # Ensure unique by current_text
                    if any(
                        b.current_text == candidate.current_text for b in next_beams
                    ):
                        continue
                next_beams.append(candidate)

            # Collect completed beams aside; keep unfinished for next iteration
            for b in next_beams:
                if b.completed:
                    all_outputs.append(copy.deepcopy(b))

            beams = next_beams

            # Early stop if all selected beams have completed
            if all(b.completed for b in beams):
                break

        # If after iterations not enough completed beams, add remaining beams
        remaining = [b for b in beams if b.completed]
        all_outputs.extend(remaining)

        # If still fewer than n outputs for this prompt, duplicate to reach n
        prompt_outputs = [b for b in all_outputs if b.prompt == prompt]
        if len(prompt_outputs) < config.search_config.n and len(prompt_outputs) > 0:
            repeats = (config.search_config.n // len(prompt_outputs)) + 1
            extended = [
                copy.deepcopy(b)
                for b in (prompt_outputs * repeats)[: config.search_config.n]
            ]
            # Replace only this prompt's outputs in all_outputs
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
