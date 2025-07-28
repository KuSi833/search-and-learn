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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pynvml
import pyrootutils
import torch
import wandb
from torch.profiler import record_function
from vllm import LLM

from sal.config import Config, OutputConfig
from sal.evaluation.evaluate import evaluate
from sal.models.reward_models import load_prm
from sal.profiler import Profiler
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_inference_output
from sal.utils.env import get_env_or_throw
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)

root = pyrootutils.find_root(indicator="pyproject.toml")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


def get_gpu_memory_gb():
    return torch.cuda.memory_allocated() / 1e9


def get_total_gpu_memory_gb():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return (info.total - info.free) / 1e9


def get_peak_gpu_memory_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def _set_up_output_dir(output_config: OutputConfig, run_id: str) -> Path:
    if output_config.output_dir is None:
        output_dir = root / f"outputs/{run_id}"
    else:
        output_dir = Path(output_config.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(config: Config):
    wandb.login(key=get_env_or_throw("WANDB_API_KEY"))
    with wandb.init(
        project=config.wandb_config.project,
        config=asdict(config),
        tags=list(config.wandb_config.tags),
    ) as run:
        output_dir = _set_up_output_dir(config.output_config, run.id)
        inference_output_path = output_dir / config.output_config.inference_output_file
        evaluation_score_path = output_dir / config.output_config.evaluation_score_file

        profiler = Profiler(config=config.profiler_config, output_dir=output_dir)

        profiler.start_profiling()
        torch.cuda.reset_peak_memory_stats()
        baseline = get_total_gpu_memory_gb()

        approach_fn = APPROACHES[config.approach]

        with record_function("load_llm"):
            llm = LLM(
                model=config.generator_config.get_model_path(),
                gpu_memory_utilization=config.gpu_memory_utilization,
                enable_prefix_caching=True,
                seed=config.search_config.seed,
                tensor_parallel_size=1,
                max_model_len=config.generator_config.max_model_len,
                enforce_eager=True,
            )
            llm_memory = get_total_gpu_memory_gb() - baseline

        with record_function("load_prm"):
            prm = load_prm(config.prm_config)
            prm_memory = get_total_gpu_memory_gb() - baseline - llm_memory

        with record_function("load_dataset"):
            dataset = get_dataset(config.dataset_config)

        # Reset peak tracking for inference
        torch.cuda.reset_peak_memory_stats()
        pre_inference = get_gpu_memory_gb()

        with record_function("inference"):
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search",
                load_from_cache_file=False,
            )

            inference_overhead = get_peak_gpu_memory_gb() - pre_inference

        with record_function("scoring"):
            dataset = score(dataset, config)
            save_inference_output(dataset, inference_output_path)

        run.log(
            {
                "memory/llm_gb": llm_memory,
                "memory/prm_gb": prm_memory,
                "memory/inference_overhead_gb": inference_overhead,
            }
        )
        logger.info(
            f"Memory - LLM: {llm_memory:.2f}GB, PRM: {prm_memory:.2f}GB, Inference: {inference_overhead:.2f}GB"
        )

        profiler.finish_profiling()

        evaluate(
            benchmark=config.evaluation_config.benchmark,
            dataset_path=inference_output_path,
            dataset_col=config.evaluation_config.dataset_col,
            output_file=evaluation_score_path,
        )
        logger.info("Done 🔥!")
