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
from vllm import LLM, SamplingParams  # type: ignore

from sal.config import ExperimentConfig
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores


def best_of_n(x, experiment_config: ExperimentConfig, llm: LLM, prm: PRM):
    tokenizer = llm.get_tokenizer()

    convs = [
        [
            {"role": "system", "content": experiment_config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]

    # TODO: set the augmented template from a file
    if experiment_config.custom_chat_template is not None:
        tokenizer.chat_template = experiment_config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [
        c for conv in templated_convs for c in [conv] * experiment_config.bon.sampling.n
    ]

    # Initialize empty lists for completions and completion tokens
    completions = [[] for _ in range(len(x["problem"]))]
    completion_tokens = [[] for _ in range(len(x["problem"]))]

    sampling_params = SamplingParams(
        temperature=experiment_config.bon.sampling.temperature,
        max_tokens=experiment_config.bon.sampling.max_tokens,
        top_p=experiment_config.bon.sampling.top_p,
        n=1,
    )

    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    if len(responses) != len(x["problem"]) * experiment_config.bon.sampling.n:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(x['problem']) * experiment_config.bon.sampling.n}"
        )

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[
                i * experiment_config.bon.sampling.n : (i + 1)
                * experiment_config.bon.sampling.n
            ]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[
                i * experiment_config.bon.sampling.n : (i + 1)
                * experiment_config.bon.sampling.n
            ]
            for output in r.outputs
        ]

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != experiment_config.bon.sampling.n:
            raise ValueError(
                f"Generated {len(c)} completions instead of {experiment_config.bon.sampling.n}"
            )

    scores = prm.score(x["problem"], completions)
    agg_scores = [
        [
            aggregate_scores(step_scores, experiment_config.bon.sampling.agg_strategy)
            for step_scores in per_completion
        ]
        for per_completion in scores
    ]

    # Select the completion with the highest score
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens

    return x
