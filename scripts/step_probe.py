#!/usr/bin/env python3

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Final, List, Tuple

import click
import pyrootutils
from vllm import LLM, SamplingParams  # type: ignore

from sal.config import (
    BaseConfig,
    ExperimentConfig,
    GeneratorConfig,
    PRMConfig,
    SearchConfig,
    VerifierGuidedBeamConfig,
    WeightedBoNConfig,
)
from sal.const import PROBE_DATA_INPUT_ROOT, PROBE_OUTPUT_ROOT
from sal.models.reward_models import load_prm
from sal.search.utils import build_conv, generate_k_steps
from sal.utils.experiment import get_model_base_path
from sal.utils.logging import setup_logging
from sal.utils.score import aggregate_scores

setup_logging()

root = pyrootutils.find_root(indicator="pyproject.toml")

logger = logging.getLogger(__name__)


def _load_probe_record(run_id: str, index: int) -> Dict[str, Any]:
    path = PROBE_DATA_INPUT_ROOT / run_id / f"{index}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Probe datum not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
        if not line:
            raise RuntimeError(f"Empty probe datum: {path}")
        return json.loads(line)


def _apply_chat_template(
    llm: LLM,
    convs: List[List[Dict[str, str]]],
    experiment_config: ExperimentConfig,
    add_generation_prompt: bool,
    continue_final_message: bool,
) -> List[str]:
    tokenizer = llm.get_tokenizer()
    if experiment_config.custom_chat_template is not None:
        tokenizer.chat_template = experiment_config.custom_chat_template
    return tokenizer.apply_chat_template(
        convs,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        tokenize=False,
    )


def _bon_candidates(
    llm: LLM,
    experiment_config: ExperimentConfig,
    problem: str,
    prefix_text: str,
    n: int,
    step_delimiter: str,
) -> Tuple[List[str], List[str]]:
    conv = build_conv(problem, prefix_text, experiment_config.system_prompt)
    templated = _apply_chat_template(
        llm,
        [conv],
        experiment_config,
        add_generation_prompt=False,
        continue_final_message=True,
    )[0]
    templated_batch = [templated for _ in range(n)]

    sampling_params = SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=experiment_config.search_config.max_tokens,
        top_p=experiment_config.search_config.top_p,
        stop=[step_delimiter],
        include_stop_str_in_output=True,
        n=1,
        seed=experiment_config.seed,
    )
    responses = llm.generate(
        templated_batch, sampling_params=sampling_params, use_tqdm=False
    )
    texts, stops = [], []
    for r in responses:
        out = r.outputs[0]
        texts.append(out.text)
        stops.append(out.stop_reason if out.stop_reason is not None else "EOS")
    return texts, stops


def _beam_candidates(
    llm: LLM,
    experiment_config: ExperimentConfig,
    problem: str,
    prefix_text: str,
    beam_width: int,
    step_delimiter: str,
) -> Tuple[List[str], List[str]]:
    conv = build_conv(problem, prefix_text, experiment_config.system_prompt)
    templated = _apply_chat_template(
        llm,
        [conv],
        experiment_config,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    sampling_params = SamplingParams(
        temperature=experiment_config.search_config.temperature,
        max_tokens=experiment_config.search_config.max_tokens,
        top_p=experiment_config.search_config.top_p,
        stop=[step_delimiter],
        include_stop_str_in_output=True,
        n=1,
        seed=experiment_config.seed,
    )

    beams = generate_k_steps(
        templated,
        lookahead_steps=0,
        llm=llm,
        sampling_params=sampling_params,
        beam_width=beam_width,
    )
    # Single input → one Beam object with next_texts of length beam_width
    next_texts = beams[0].next_texts or []
    stop_reasons = beams[0].stop_reasons or [None] * len(next_texts)
    stop_reasons = [s if s is not None else "EOS" for s in stop_reasons]
    return next_texts, stop_reasons


def _score_with_prm(
    prm,
    problem: str,
    prefix_text: str,
    candidates: List[str],
) -> List[List[float]]:
    prompts = [problem]
    completions = [[prefix_text + c for c in candidates]]
    scores = prm.score(prompts, completions)
    # scores shape: [1][num_candidates][score_vector]
    return scores[0]


def _summarise_aggregates(score_vecs: List[List[float]]) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for vec in score_vecs:
        results.append(
            {
                "last": float(aggregate_scores(vec, "last")),
                "mean": float(aggregate_scores(vec, "mean")),
                "min": float(aggregate_scores(vec, "min")),
                "prod": float(aggregate_scores(vec, "prod")),
            }
        )
    return results


@click.command(
    help="Probe next-step candidates from a shared prefix for selected strategies"
)
@click.option("--run-id", required=True, type=str)
@click.option("--index", required=True, type=int)
def main(run_id: str, index: int) -> None:
    # Load probe datum
    datum = _load_probe_record(run_id, index)
    problem: str = datum.get("problem", "")
    prefix_text: str = datum.get("prefix_text", "")
    step_delimiter: str = datum.get("step_delimiter", "\n\n")
    fail_step: int = int(datum.get("fail_step", 1))

    PROBE_DATA_INPUT_ROOT.mkdir(exist_ok=True)
    PROBE_OUTPUT_ROOT.mkdir(exist_ok=True)

    MODEL_BASE_PATH = get_model_base_path()

    INSTRUCT_MODEL: Final[GeneratorConfig] = GeneratorConfig(
        base_path=MODEL_BASE_PATH,
        name="Qwen/Qwen2.5-Math-7B-Instruct",
        parameter_count="7B",
    )

    PRM_MODEL: Final[PRMConfig] = PRMConfig(
        base_path=MODEL_BASE_PATH,
        name="Qwen/Qwen2.5-Math-PRM-7B",
    )

    BASE_CONFIG: Final[BaseConfig] = BaseConfig(
        generator_config=INSTRUCT_MODEL,
        prm_config=PRM_MODEL,
    )

    PROBE_SEARCH: Final[SearchConfig] = SearchConfig(
        temperature=0.7,
        top_p=0.8,
        max_tokens=2048,
        agg_strategy="prod",
    )

    WBON_CFG: Final[WeightedBoNConfig] = WeightedBoNConfig(n=8)
    VBEAM_CFG: Final[VerifierGuidedBeamConfig] = VerifierGuidedBeamConfig(
        beam_width=4, lookahead=0
    )

    PROBE_EXPERIMENT_CONFIG: Final[ExperimentConfig] = ExperimentConfig(
        search_config=PROBE_SEARCH,
        wbon_config=WBON_CFG,
        verifier_beam_config=VBEAM_CFG,
    )

    # Load models
    llm = LLM(
        model=BASE_CONFIG.generator_config.get_model_path(),
        gpu_memory_utilization=BASE_CONFIG.generator_config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=BASE_CONFIG.seed,
        tensor_parallel_size=1,
        max_model_len=BASE_CONFIG.generator_config.max_model_len,
        enforce_eager=True,
    )
    prm = load_prm(BASE_CONFIG.prm_config)

    # Strategies
    results: Dict[str, Any] = {
        "run_id": run_id,
        "index": index,
        "fail_step": fail_step,
        "prefix_text": prefix_text,
        "problem": problem,
        "strategies": [],
        "config": {
            "search": asdict(PROBE_EXPERIMENT_CONFIG.search_config),
            "wbon": asdict(PROBE_EXPERIMENT_CONFIG.wbon_config)
            if PROBE_EXPERIMENT_CONFIG.wbon_config
            else None,
            "verifier_beam": asdict(PROBE_EXPERIMENT_CONFIG.verifier_beam_config)
            if PROBE_EXPERIMENT_CONFIG.verifier_beam_config
            else None,
        },
    }

    # Weighted Best-of-N (proposal only)
    if PROBE_EXPERIMENT_CONFIG.wbon_config is not None:
        bon_n = int(PROBE_EXPERIMENT_CONFIG.wbon_config.n)
        bon_texts, bon_stops = _bon_candidates(
            llm,
            PROBE_EXPERIMENT_CONFIG,
            problem,
            prefix_text,
            n=bon_n,
            step_delimiter=step_delimiter,
        )
        bon_scores = _score_with_prm(prm, problem, prefix_text, bon_texts)
        bon_aggs = _summarise_aggregates(bon_scores)
        results["strategies"].append(
            {
                "name": "weighted_bon",
                "params": {"n": bon_n},
                "candidates": [
                    {
                        "text": t,
                        "stop_reason": s,
                        "prm_scores": sc,
                        "aggregates": agg,
                    }
                    for t, s, sc, agg in zip(
                        bon_texts, bon_stops, bon_scores, bon_aggs, strict=True
                    )
                ],
            }
        )

    # Verifier-guided beam (one-iteration, proposal only)
    if PROBE_EXPERIMENT_CONFIG.verifier_beam_config is not None:
        beam_width = int(PROBE_EXPERIMENT_CONFIG.verifier_beam_config.beam_width)
        beam_texts, beam_stops = _beam_candidates(
            llm,
            PROBE_EXPERIMENT_CONFIG,
            problem,
            prefix_text,
            beam_width=beam_width,
            step_delimiter=step_delimiter,
        )
        beam_scores = _score_with_prm(prm, problem, prefix_text, beam_texts)
        beam_aggs = _summarise_aggregates(beam_scores)
        results["strategies"].append(
            {
                "name": "verifier_guided_beam",
                "params": {"beam_width": beam_width},
                "candidates": [
                    {
                        "text": t,
                        "stop_reason": s,
                        "prm_scores": sc,
                        "aggregates": agg,
                    }
                    for t, s, sc, agg in zip(
                        beam_texts, beam_stops, beam_scores, beam_aggs, strict=True
                    )
                ],
            }
        )

    # Write output
    out_dir = PROBE_OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{index}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False))
        f.write("\n")
    click.secho(f"✔ Wrote probe results to {out_path}", fg="green")


if __name__ == "__main__":
    main()
