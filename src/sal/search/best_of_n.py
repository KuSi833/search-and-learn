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

import numpy as np
import torch.profiler
from torch.profiler import record_function
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores


def best_of_n(x, config: Config, llm: LLM, prm: PRM):
    tokenizer = llm.get_tokenizer()

    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]

    # TODO: set the augmented template from a file
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [
        c for conv in templated_convs for c in [conv] * config.search_config.n
    ]

    # Initialize empty lists for completions and completion tokens
    completions = [[] for _ in range(len(x["problem"]))]
    completion_tokens = [[] for _ in range(len(x["problem"]))]

    sampling_params = SamplingParams(
        temperature=config.search_config.temperature,
        max_tokens=config.search_config.max_tokens,
        top_p=config.search_config.top_p,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
    )

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=False,  # Don't need shapes for memory
        with_stack=False,  # No stack traces
    ) as prof:
        responses = llm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        prof.export_chrome_trace("./trace/memory_trace.json")

    if len(responses) != len(x["problem"]) * config.search_config.n:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(x['problem'] * config.n)}"
        )

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[
                i * config.search_config.n : (i + 1) * config.search_config.n
            ]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[
                i * config.search_config.n : (i + 1) * config.search_config.n
            ]
            for output in r.outputs
        ]

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != config.search_config.n:
            raise ValueError(
                f"Generated {len(c)} completions instead of {config.search_config.n}"
            )

    scores = prm.score(x["problem"], completions)
    agg_scores = [
        [aggregate_scores(s, config.search_config.agg_strategy) for s in score]
        for score in scores
    ]

    # Select the completion with the highest score
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens

    return x
