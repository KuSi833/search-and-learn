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

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import ExperimentConfig
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    prompt: str
    current_text: str
    completed: bool
    last_stop_reason: str | None
    best_scores: List[float]


def _softmax(x: List[float]) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    x_arr = np.array(x, dtype=np.float64)
    # stabilise
    x_arr = x_arr - np.max(x_arr)
    exps = np.exp(x_arr)
    s = np.sum(exps)
    if not np.isfinite(s) or s <= 0.0:
        # fallback to uniform
        return np.ones_like(exps) / len(exps)
    return exps / s


def _particles(
    batch_of_prompts: List[str], experiment_config: ExperimentConfig, llm: LLM, prm: PRM
) -> List[Beam]:
    # One particle per requested sample (n particles total per prompt)
    n_particles = experiment_config.search_config.n

    step_sampling_params = SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=experiment_config.search_config.max_tokens,
        top_p=experiment_config.search_config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        seed=experiment_config.seed,
        n=1,
    )

    # When forcing completion at the final iteration
    final_sampling_params = SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=experiment_config.search_config.max_tokens,
        top_p=experiment_config.search_config.top_p,
        seed=experiment_config.seed,
        n=1,
    )

    all_outputs: List[Beam] = []

    # Process each prompt independently (resampling is per-prompt)
    for prompt in batch_of_prompts:
        # Initialise independent particles
        particles: List[Particle] = [
            Particle(
                prompt=prompt,
                current_text="",
                completed=False,
                last_stop_reason=None,
                best_scores=[],
            )
            for _ in range(n_particles)
        ]

        tokenizer = llm.get_tokenizer()
        if experiment_config.custom_chat_template is not None:
            tokenizer.chat_template = experiment_config.custom_chat_template

        num_iterations = experiment_config.beam_search_config.num_iterations

        for iteration in tqdm(
            range(num_iterations), desc="Particle filter iterations", leave=False
        ):
            # Identify active particles needing a new step
            active_indices = [i for i, p in enumerate(particles) if not p.completed]

            if len(active_indices) == 0:
                break

            # On the last iteration, force completion to EOS
            use_final = iteration == num_iterations - 1
            sampling_params = (
                final_sampling_params if use_final else step_sampling_params
            )

            # Build conversations and apply chat template
            convs = [
                build_conv(
                    prompt, particles[i].current_text, experiment_config.system_prompt
                )
                for i in active_indices
            ]
            templated_convs = tokenizer.apply_chat_template(
                convs,
                add_generation_prompt=(
                    len(particles[active_indices[0]].current_text) == 0
                ),
                continue_final_message=(
                    len(particles[active_indices[0]].current_text) > 0
                ),
                tokenize=False,
            )

            # Generate one step for each active particle
            responses = llm.generate(
                templated_convs, sampling_params=sampling_params, use_tqdm=False
            )
            for idx, resp in zip(active_indices, responses, strict=True):
                out = resp.outputs[0]
                gen_text = out.text
                stop_reason = out.stop_reason
                if stop_reason is None:
                    stop_reason = "EOS"
                particles[idx].current_text = (
                    particles[idx].current_text or ""
                ) + gen_text
                particles[idx].last_stop_reason = stop_reason
                # Heuristic completion: boxed indicates final answer, or EOS when forcing final step
                if "boxed{" in particles[idx].current_text or stop_reason == "EOS":
                    particles[idx].completed = True

            # PRM scoring for all particles (active and completed) to compute resampling weights
            prm_prompts: List[str] = [prompt for _ in range(n_particles)]
            prm_completions: List[List[str]] = [[p.current_text] for p in particles]
            all_scores_per_particle = prm.score(prm_prompts, prm_completions)

            # Update each particle's best_scores with its current full-trace score
            agg_scores: List[float] = []
            for p, s in zip(particles, all_scores_per_particle, strict=True):
                # s is List[List[float]], one candidate per inner list; we used one completion → s[0]
                p.best_scores = s[0]
                agg_scores.append(
                    aggregate_scores(
                        p.best_scores, experiment_config.search_config.agg_strategy
                    )
                )

            # Resample particles for next iteration according to softmax weights
            weights = _softmax(agg_scores)
            rng = np.random.default_rng(experiment_config.seed + iteration)
            ancestor_indices = rng.choice(
                np.arange(n_particles), size=n_particles, replace=True, p=weights
            )

            # Rebuild the particle population by cloning ancestors
            particles = [
                Particle(
                    prompt=particles[a].prompt,
                    current_text=particles[a].current_text,
                    completed=particles[a].completed,
                    last_stop_reason=particles[a].last_stop_reason,
                    best_scores=list(particles[a].best_scores),
                )
                for a in ancestor_indices
            ]

            # If all are completed, stop early
            if all(p.completed for p in particles):
                break

        # Convert final particles to Beam objects for uniform downstream handling
        for i, p in enumerate(particles):
            all_outputs.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text=p.current_text,
                    next_texts=None,
                    lookahead_texts=None,
                    stop_reasons=None,
                    best_scores=p.best_scores,
                    all_scores=[p.best_scores],
                    previous_text=None,
                    pruned=False,
                    history=[],
                    completed=p.completed,
                    completion_tokens=-1,
                )
            )

    return all_outputs


def particles(examples, experiment_config: ExperimentConfig, llm: LLM, prm: PRM):
    problems = examples["problem"]
    particle_results = _particles(problems, experiment_config, llm, prm)

    grouped_results = defaultdict(list)
    for res in particle_results:
        grouped_results[res.prompt].append(res)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        scores = [b.best_scores for b in beams]
        agg = [
            aggregate_scores(s, experiment_config.search_config.agg_strategy)
            for s in scores
        ]
        pred = completions[int(np.argmax(agg))]

        results["completions"].append(completions)
        results["scores"].append(scores)
        results["pred"].append(pred)
        results["completion_tokens"].append(-1)

    return results
