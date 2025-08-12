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
class PGParticle:
    prompt: str
    current_text: str
    completed: bool
    last_stop_reason: str | None
    best_scores: List[float]
    partial_agg_scores: List[float]


def _softmax(x: List[float], temperature: float = 1.0) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    x_arr = np.array(x, dtype=np.float64)
    if temperature <= 0:
        temperature = 1.0
    x_arr = (x_arr - np.max(x_arr)) / temperature
    exps = np.exp(x_arr)
    s = np.sum(exps)
    if not np.isfinite(s) or s <= 0.0:
        return np.ones_like(exps) / len(exps)
    return exps / s


def _run_pg_for_prompt(
    prompt: str, experiment_config: ExperimentConfig, llm: LLM, prm: PRM
) -> List[Beam]:
    n_particles = experiment_config.search_config.n
    num_iterations = experiment_config.beam_search_config.num_iterations
    num_ref_particles = 1

    step_sampling_params = SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=experiment_config.search_config.max_tokens,
        top_p=experiment_config.search_config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        seed=experiment_config.seed,
        n=1,
    )
    final_sampling_params = SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=experiment_config.search_config.max_tokens,
        top_p=experiment_config.search_config.top_p,
        seed=experiment_config.seed,
        n=1,
    )

    tokenizer = llm.get_tokenizer()
    if experiment_config.custom_chat_template is not None:
        tokenizer.chat_template = experiment_config.custom_chat_template

    ref_particles: List[PGParticle] = []
    final_population: List[PGParticle] = []

    for iteration in tqdm(
        range(num_iterations), desc="Particle Gibbs iterations", leave=False
    ):
        num_free_particles = n_particles - len(ref_particles)
        particles: List[PGParticle] = [
            PGParticle(
                prompt=prompt,
                current_text="",
                completed=False,
                last_stop_reason=None,
                best_scores=[],
                partial_agg_scores=[],
            )
            for _ in range(num_free_particles)
        ] + [
            PGParticle(
                prompt=p.prompt,
                current_text=p.current_text,
                completed=p.completed,
                last_stop_reason=p.last_stop_reason,
                best_scores=list(p.best_scores),
                partial_agg_scores=list(p.partial_agg_scores),
            )
            for p in ref_particles
        ]

        current_step = 0
        while not all(p.completed for p in particles):
            active_indices = [i for i, p in enumerate(particles) if not p.completed]
            if len(active_indices) == 0:
                break

            use_final = iteration == num_iterations - 1
            sampling_params = (
                final_sampling_params if use_final else step_sampling_params
            )

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

            responses = llm.generate(
                templated_convs, sampling_params=sampling_params, use_tqdm=False
            )
            for idx, resp in zip(active_indices, responses, strict=True):
                out = resp.outputs[0]
                gen_text = out.text
                stop_reason = out.stop_reason or "EOS"
                particles[idx].current_text = (
                    particles[idx].current_text or ""
                ) + gen_text
                particles[idx].last_stop_reason = stop_reason
                if "boxed{" in particles[idx].current_text or stop_reason == "EOS":
                    particles[idx].completed = True

            # Score all particles after this step
            prm_prompts: List[str] = [prompt for _ in range(len(particles))]
            prm_completions: List[List[str]] = [[p.current_text] for p in particles]
            all_scores_per_particle = prm.score(prm_prompts, prm_completions)

            agg_scores: List[float] = []
            for p, s in zip(particles, all_scores_per_particle, strict=True):
                s0 = s[0]
                if isinstance(s0, list):
                    p.best_scores = s0
                else:
                    p.best_scores = [float(s0)]
                agg_score = aggregate_scores(
                    p.best_scores, experiment_config.search_config.agg_strategy
                )
                p.partial_agg_scores.append(agg_score)
                agg_scores.append(agg_score)

            # Resample free particles using weights at the current step
            resample_tau = experiment_config.particles_config.resampling_temperature
            weights = _softmax(agg_scores, resample_tau)
            rng = np.random.default_rng(
                experiment_config.seed + iteration + current_step
            )
            candidate_indices = np.arange(len(particles))
            if not experiment_config.particles_config.allow_completed_ancestors:
                active_mask = np.array([not p.completed for p in particles], dtype=bool)
                if active_mask.any():
                    candidate_indices = candidate_indices[active_mask]
                    weights = weights[active_mask]
                    weights = weights / weights.sum()

            selected_free = rng.choice(
                candidate_indices, size=num_free_particles, replace=True, p=weights
            )

            # Duplicate selected as new free particles
            resampled_particles: List[PGParticle] = [
                PGParticle(
                    prompt=particles[i].prompt,
                    current_text=particles[i].current_text,
                    completed=particles[i].completed,
                    last_stop_reason=particles[i].last_stop_reason,
                    best_scores=list(particles[i].best_scores),
                    partial_agg_scores=list(particles[i].partial_agg_scores),
                )
                for i in selected_free
            ]

            # Rebuild population: free + unchanged references
            particles = (
                resampled_particles + particles[-len(ref_particles) :]
                if len(ref_particles) > 0
                else resampled_particles
            )

            current_step += 1

            if (
                all(p.completed for p in particles)
                and current_step >= experiment_config.particles_config.min_iterations
            ):
                break

        # Select references for the next iteration
        final_agg_scores = [
            aggregate_scores(
                p.best_scores, experiment_config.search_config.agg_strategy
            )
            for p in particles
        ]
        final_weights = _softmax(
            final_agg_scores, experiment_config.particles_config.resampling_temperature
        )
        rng = np.random.default_rng(experiment_config.seed + 1000 + iteration)
        chosen_indices = rng.choice(
            np.arange(len(particles)),
            size=num_ref_particles,
            replace=True,
            p=final_weights,
        )
        ref_particles = [particles[i] for i in chosen_indices]

        # Keep track of last population for outputs
        final_population = particles

    # Convert to Beam results
    beams: List[Beam] = []
    for i, p in enumerate(final_population):
        beams.append(
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
    return beams


def _gibbs(
    batch_of_prompts: List[str], experiment_config: ExperimentConfig, llm: LLM, prm: PRM
) -> List[Beam]:
    all_outputs: List[Beam] = []
    for prompt in batch_of_prompts:
        all_outputs.extend(_run_pg_for_prompt(prompt, experiment_config, llm, prm))
    return all_outputs


def gibbs(examples, experiment_config: ExperimentConfig, llm: LLM, prm: PRM):
    problems = examples["problem"]
    gibbs_results = _gibbs(problems, experiment_config, llm, prm)

    grouped_results = defaultdict(list)
    for res in gibbs_results:
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
        pred = completions[int(np.argmax(agg))] if len(completions) > 0 else ""

        results["completions"].append(completions)
        results["scores"].append(scores)
        results["pred"].append(pred)
        results["completion_tokens"].append(-1)

    return results
