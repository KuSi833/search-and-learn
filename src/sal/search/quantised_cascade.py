import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import ExperimentConfig
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps

logger = logging.getLogger()


def _qcts(
    batch_of_prompts: list[str],
    experiment_config: ExperimentConfig,
    target_llm: LLM,
    draft_llm: LLM,
    prm: PRM,
):
    sampling_params = SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=2048,
        top_p=experiment_config.search_config.top_p,
        stop=[
            "\n\n"
        ],  # we consider that a step in the problem is indicated by a double newline
        include_stop_str_in_output=True,
        n=1,
        seed=experiment_config.seed,
    )

    # Thresholds for quantised cascade
    high_threshold = experiment_config.qcconfig.high_threshold
    low_threshold = experiment_config.qcconfig.low_threshold  # Prune beam

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
                )
            )

    for i in range(experiment_config.beam_search_config.num_iterations):
        # generation with draft model (4-bit)
        gen_beams = [b for b in beams if not b.pruned]
        if len(gen_beams) == 0:
            break

        if i == experiment_config.beam_search_config.num_iterations - 1:
            # last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=experiment_config.search_config.temperature,
                max_tokens=2048,
                top_p=experiment_config.search_config.top_p,
                n=1,
                seed=experiment_config.seed,
            )

        # Step 1: Generate N beams using draft model (4-bit)
        convs = [
            build_conv(b.prompt, b.current_text, experiment_config.system_prompt)
            for b in gen_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = draft_llm.get_tokenizer()
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
        draft_gen_results = generate_k_steps(
            templated_convs,
            lookahead,
            draft_llm,
            sampling_params,
            experiment_config.beam_search_config.beam_width,
        )

        # Process draft generations
        draft_prompts, draft_completions = [], []
        for beam, gen_result in zip(gen_beams, draft_gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            if len(beam.next_texts) != experiment_config.beam_search_config.beam_width:
                beam.pruned = True
                logger.warning(
                    f"beam {beam.index} has {len(beam.next_texts)} completions"
                )
            draft_prompts.append(beam.prompt)
            draft_completions.append(
                [beam.current_text + t for t in beam.lookahead_texts]
            )

        # Step 2: Score all beams with PRM
        draft_scores = prm.score(draft_prompts, draft_completions)

        # Step 3: Apply quantised cascade logic
        upgrade_indices = []  # Track which beams need target model upgrade

        for beam_idx, (beam, scores) in enumerate(
            zip(gen_beams, draft_scores, strict=True)
        ):
            agg_scores = [
                aggregate_scores(s, experiment_config.search_config.agg_strategy)
                for s in scores
            ]
            best_score_ind = np.argmax(agg_scores)
            best_score = agg_scores[best_score_ind]

            if best_score > high_threshold:
                # Keep draft quality - good enough
                beam.all_scores = scores
                beam.previous_text = beam.current_text
                beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
                beam.history.append(beam.next_texts[best_score_ind])
                beam.best_scores = scores[best_score_ind]
            elif best_score > low_threshold:
                # Middle range - upgrade to target model
                upgrade_indices.append(beam_idx)
            else:
                # Low score - prune beam
                beam.pruned = True

        # Step 4: Re-execute selected beams with target model (8-bit)
        if upgrade_indices:
            upgrade_beams = [gen_beams[idx] for idx in upgrade_indices]
            upgrade_convs = [
                build_conv(b.prompt, b.current_text, experiment_config.system_prompt)
                for b in upgrade_beams
            ]

            target_tokenizer = target_llm.get_tokenizer()
            if experiment_config.custom_chat_template is not None:
                target_tokenizer.chat_template = experiment_config.custom_chat_template
            target_templated_convs = target_tokenizer.apply_chat_template(
                upgrade_convs,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )

            target_gen_results = generate_k_steps(
                target_templated_convs,
                lookahead,
                target_llm,
                sampling_params,
                experiment_config.beam_search_config.beam_width,
            )

            # Process target model generations
            target_prompts, target_completions = [], []
            for beam, gen_result in zip(upgrade_beams, target_gen_results, strict=True):
                beam.next_texts = gen_result.next_texts
                beam.stop_reasons = gen_result.stop_reasons
                beam.lookahead_texts = gen_result.lookahead_texts
                target_prompts.append(beam.prompt)
                target_completions.append(
                    [beam.current_text + t for t in beam.lookahead_texts]
                )

            # Score target model generations
            target_scores = prm.score(target_prompts, target_completions)

            # Update upgrade beams with target model results
            for beam, scores in zip(upgrade_beams, target_scores, strict=True):
                agg_scores = [
                    aggregate_scores(s, experiment_config.search_config.agg_strategy)
                    for s in scores
                ]
                best_score_ind = np.argmax(agg_scores)
                beam.all_scores = scores
                beam.previous_text = beam.current_text
                beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
                beam.history.append(beam.next_texts[best_score_ind])
                beam.best_scores = scores[best_score_ind]

        # Check for completion and prune completed beams
        for beam in gen_beams:
            if not beam.pruned:
                if (
                    beam.next_texts
                    and len(beam.next_texts) > 0
                    and (
                        beam.next_texts[0] == ""
                        or (beam.stop_reasons and beam.stop_reasons[0] == "EOS")
                    )
                ):
                    beam.pruned = True
                elif "boxed{" in beam.current_text:
                    beam.pruned = True

    # Prepare output with mixed-precision results
    output: list[Beam] = []
    for beam in beams:
        if beam.next_texts:
            for i in range(
                min(
                    len(beam.next_texts),
                    experiment_config.beam_search_config.beam_width,
                )
            ):
                output.append(
                    Beam(
                        prompt=beam.prompt,
                        index=beam.index,
                        current_text=beam.previous_text + beam.next_texts[i]
                        if beam.previous_text and beam.next_texts
                        else beam.current_text,
                        next_texts=None,
                        lookahead_texts=None,
                        stop_reasons=None,
                        best_scores=beam.all_scores[i]
                        if beam.all_scores and i < len(beam.all_scores)
                        else beam.best_scores,
                        all_scores=beam.all_scores,
                        previous_text=beam.current_text,
                        pruned=beam.pruned,
                        history=beam.history,
                    )
                )
        else:
            # Fallback for beams without next_texts
            output.append(beam)

    return output


def qcts(
    examples,
    experiment_config: ExperimentConfig,
    target_llm: LLM,
    draft_llm: LLM,
    prm: PRM,
):
    problems = examples["problem"]
    beam_results = _qcts(problems, experiment_config, target_llm, draft_llm, prm)

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

    return results
