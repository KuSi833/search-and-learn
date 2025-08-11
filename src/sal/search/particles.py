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

import copy
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


def _softmax(x: List[float], temperature: float = 1.0) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    x_arr = np.array(x, dtype=np.float64)
    # stabilise
    if temperature <= 0:
        temperature = 1.0
    x_arr = (x_arr - np.max(x_arr)) / temperature
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

            # Build conversations and apply chat template per particle depending on whether this is the first step
            convs = []
            is_first_step_flags = []
            for i in active_indices:
                conv = build_conv(
                    prompt, particles[i].current_text, experiment_config.system_prompt
                )
                convs.append(conv)
                is_first_step_flags.append(len(particles[i].current_text) == 0)

            # Partition into first-step and continuation to set chat template flags correctly
            first_idxs = [
                j for j, is_first in enumerate(is_first_step_flags) if is_first
            ]
            cont_idxs = [
                j for j, is_first in enumerate(is_first_step_flags) if not is_first
            ]

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

            # Generate one step for each active particle with unique seeds to promote diversity
            for idx, conv in zip(active_indices, templated_convs, strict=True):
                per_particle_params = copy.deepcopy(sampling_params)
                # Temperature jitter for diversity
                if experiment_config.particles_config.temperature_jitter_std > 0.0:
                    jitter = float(
                        np.random.default_rng(
                            experiment_config.seed + iteration * 7919 + idx
                        ).normal(
                            0.0,
                            experiment_config.particles_config.temperature_jitter_std,
                        )
                    )
                    base_temp = float(sampling_params.temperature or 0.0)
                    per_particle_params.temperature = max(0.0, base_temp + jitter)
                per_particle_params.seed = (
                    int(experiment_config.seed) + int(iteration) * 1000003 + int(idx)
                )
                resp_list = llm.generate(
                    [conv], sampling_params=per_particle_params, use_tqdm=False
                )
                out = resp_list[0].outputs[0]
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
                s0 = s[0]
                if isinstance(s0, list):
                    p.best_scores = s0
                else:
                    p.best_scores = [float(s0)]
                agg_scores.append(
                    aggregate_scores(
                        p.best_scores, experiment_config.search_config.agg_strategy
                    )
                )

            # Measure diversity before resampling
            diversity_before = len({p.current_text for p in particles})

            # Resample particles for next iteration according to softmax weights
            resample_tau = experiment_config.particles_config.resampling_temperature
            # Optional score noise to fight collapse
            if experiment_config.particles_config.score_noise_std > 0.0:
                noise = np.random.default_rng(
                    experiment_config.seed + iteration * 104729
                ).normal(
                    0.0,
                    experiment_config.particles_config.score_noise_std,
                    size=len(agg_scores),
                )
                noisy_scores = (np.array(agg_scores, dtype=np.float64) + noise).tolist()
            else:
                noisy_scores = agg_scores
            weights = _softmax(noisy_scores, resample_tau)
            rng = np.random.default_rng(experiment_config.seed + iteration)
            candidate_indices = np.arange(n_particles)
            if not experiment_config.particles_config.allow_completed_ancestors:
                # Mask completed indices out by renormalising over active ones
                active_mask = np.array([not p.completed for p in particles], dtype=bool)
                if active_mask.any():
                    candidate_indices = candidate_indices[active_mask]
                    weights = weights[active_mask]
                    weights = weights / weights.sum()
            # Keep a copy of the final weights used for sampling for telemetry
            chosen_weights = np.copy(weights)
            if (
                experiment_config.particles_config.resampling_method == "systematic"
                and weights.size > 0
            ):
                # Systematic resampling (low-variance)
                # Ensure weights sum to 1
                w = weights / weights.sum()
                cdf = np.cumsum(w)
                start = rng.random() / n_particles
                points = start + (np.arange(n_particles) / n_particles)
                ancestor_indices = np.searchsorted(cdf, points)
                # Map back into candidate_indices domain
                ancestor_indices = candidate_indices[ancestor_indices]
            else:
                ancestor_indices = rng.choice(
                    candidate_indices, size=n_particles, replace=True, p=weights
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

            # Optional debug telemetry
            if experiment_config.particles_config.debug_enable and (
                (iteration % max(1, experiment_config.particles_config.debug_log_every))
                == 0
            ):
                num_completed = int(sum(1 for p in particles if p.completed))
                diversity_after = len({p.current_text for p in particles})
                scores_arr = np.array(agg_scores, dtype=np.float64)
                weights_arr = np.array(chosen_weights, dtype=np.float64)
                ess = (
                    1.0 / float(np.sum(np.square(weights_arr)))
                    if weights_arr.size > 0
                    else 0.0
                )
                entropy = (
                    -float(
                        np.sum(weights_arr * np.log(np.clip(weights_arr, 1e-12, 1.0)))
                    )
                    if weights_arr.size > 0
                    else 0.0
                )
                unique_ancestors = len(set(int(a) for a in ancestor_indices.tolist()))
                completed_ancestor_frac = (
                    (
                        float(
                            sum(
                                1
                                for a in ancestor_indices
                                if particles[int(a)].completed
                            )
                        )
                        / len(ancestor_indices)
                    )
                    if len(ancestor_indices) > 0
                    else 0.0
                )

                # Compute formatted values for logging
                score_mean = float(scores_arr.mean()) if scores_arr.size else 0.0
                score_std = float(scores_arr.std()) if scores_arr.size else 0.0

                logger.info(
                    f"[particles] it={iteration} completed={num_completed}/{n_particles} div_pre={diversity_before} div_post={diversity_after} "
                    f"score_mean={score_mean:.4f} score_std={score_std:.4f} w_entropy={entropy:.3f} ess={ess:.2f} "
                    f"uniq_anc={unique_ancestors} comp_anc_frac={completed_ancestor_frac:.2f}"
                )

            # If all are completed, stop early (respect minimum iterations)
            if (
                all(p.completed for p in particles)
                and iteration + 1 >= experiment_config.particles_config.min_iterations
            ):
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
