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

import torch
import wandb
from vllm import LLM

from sal.config import Config
from sal.evaluation.evaluate import evaluate
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.env import get_env_or_throw
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


def get_gpu_memory_gb():
    return torch.cuda.memory_allocated() / 1e9


def main(config: Config):
    wandb.login(key=get_env_or_throw("WANDB_API_KEY"))
    with wandb.init(
        project=config.wandb_config.project,
        config=asdict(config),
        tags=list(config.wandb_config.tags),
    ) as run:

        def trace_handler(prof):
            prof.export_chrome_trace("./trace/memory_trace.json")
            print("Trace exported successfully")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=trace_handler,
        ) as prof:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            baseline = get_gpu_memory_gb()

            # Load models
            approach_fn = APPROACHES[config.approach]

            llm = LLM(
                model=config.generator_config.get_model_path(),
                gpu_memory_utilization=config.gpu_memory_utilization,
                enable_prefix_caching=True,
                seed=config.search_config.seed,
                tensor_parallel_size=1,
                enforce_eager=True,
            )
            llm_memory = get_gpu_memory_gb() - baseline

            prm = load_prm(config.prm_config)
            prm_memory = get_gpu_memory_gb() - baseline - llm_memory

            dataset = get_dataset(config.dataset_config)

            # Reset peak tracking for inference
            torch.cuda.reset_peak_memory_stats()
            pre_inference = get_gpu_memory_gb()

            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=config.search_config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc="Running search",
                load_from_cache_file=False,
            )

            inference_overhead = torch.cuda.max_memory_allocated() / 1e9 - pre_inference

            dataset = score(dataset, config)

            # Log everything once
            run.log(
                {
                    "memory/llm_gb": llm_memory,
                    "memory/prm_gb": prm_memory,
                    "memory/inference_overhead_gb": inference_overhead,
                    "memory/peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "memory/final_gb": get_gpu_memory_gb(),
                }
            )

            logger.info(
                f"Memory - LLM: {llm_memory:.2f}GB, PRM: {prm_memory:.2f}GB, Inference: {inference_overhead:.2f}GB"
            )

            dataset_path = save_dataset(dataset, config, run.id)
            output_file: Path = dataset_path.parent / "score.jsonl"
            evaluate(
                benchmark=config.evaluation_config.benchmark,
                dataset_path=dataset_path,
                dataset_col=config.evaluation_config.dataset_col,
                output_file=output_file,
            )
            logger.info("Done 🔥!")
