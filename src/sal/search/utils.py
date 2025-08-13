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
from dataclasses import dataclass

import numpy as np
from vllm import LLM, SamplingParams  # type: ignore

logger = logging.getLogger()


def build_conv(
    prompt: str, response: str | None, system_prompt: str
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Only include an assistant turn if we actually have content
    if response is not None and response != "":
        conversation.append({"role": "assistant", "content": response})

    return conversation


def last(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return x[-1]


def list_mean(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return np.mean(x)


@dataclass
class Beam:
    prompt: str
    index: int
    current_text: str
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str] | None
    best_scores: list[float]  # the PRM scores (optional, may be unused)
    all_scores: list[float]  # step-wise PRM scores for this beam's current_text
    previous_text: str | None
    pruned: bool
    history: list[str]
    completed: bool = False
    completion_tokens: int = 0


def generate_k_steps(
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
) -> list[Beam]:
    # First step: sample beam_width candidates per prompt
    first_params = copy.deepcopy(sampling_params)
    first_params.n = beam_width
    first_outputs = llm.generate(templated_convs, first_params, use_tqdm=False)

    # Collect first-step candidates
    next_texts_all: list[list[str]] = []
    stop_reasons_all: list[list[str]] = []
    for out in first_outputs:
        texts = [o.text for o in out.outputs]
        reasons = [(o.stop_reason or "EOS") for o in out.outputs]
        next_texts_all.append(texts)
        stop_reasons_all.append(reasons)

    # Optional greedy lookahead to help ranking
    lookahead_texts_all: list[list[str]] = [
        ["" for _ in range(beam_width)] for _ in templated_convs
    ]
    if lookahead_steps > 0:
        greedy_params = copy.deepcopy(sampling_params)
        greedy_params.n = 1
        greedy_params.temperature = 0.0
        for _ in range(lookahead_steps):
            prompts = []
            indices: list[tuple[int, int]] = []
            for i, base in enumerate(templated_convs):
                for j in range(beam_width):
                    # skip if already EOS
                    if stop_reasons_all[i][j] == "EOS":
                        continue
                    prefix = base + next_texts_all[i][j] + lookahead_texts_all[i][j]
                    prompts.append(prefix)
                    indices.append((i, j))
            if not prompts:
                break
            outs = llm.generate(prompts, greedy_params, use_tqdm=False)
            for (i, j), out in zip(indices, outs, strict=True):
                delta = out.outputs[0].text
                sr = out.outputs[0].stop_reason or "EOS"
                lookahead_texts_all[i][j] += delta
                stop_reasons_all[i][j] = sr

    # Package into Beam-like containers expected by beam_search
    results: list[Beam] = []
    for i, base in enumerate(templated_convs):
        results.append(
            Beam(
                prompt=base,
                index=i,
                current_text="",
                next_texts=next_texts_all[i],
                lookahead_texts=lookahead_texts_all[i],
                stop_reasons=stop_reasons_all[i],
                best_scores=[0.0],
                all_scores=[],
                previous_text=None,
                pruned=False,
                history=[],
            )
        )

    return results
