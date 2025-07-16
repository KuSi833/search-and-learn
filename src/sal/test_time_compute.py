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

import wandb
import torch
from torch.profiler import record_function
from pathlib import Path
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.score import score
from sal.utils.env import get_dotenv_or_throw
from sal.evaluation.evaluate import evaluate
from dataclasses import asdict
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


def main(config: Config):
    wandb.login(key=get_dotenv_or_throw("WANDB_API_KEY"))

    with wandb.init(
        project=config.wandb_config.project,
        config=asdict(config),
        tags=list(config.wandb_config.tags),
    ) as run:
        approach_fn = APPROACHES[config.approach]

        num_gpus = torch.cuda.device_count()
        llm = LLM(
            model=config.generator_config.get_model_path(),
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.search_config.seed,
            tensor_parallel_size=num_gpus,
        )

        prm = load_prm(config.prm_config)

        dataset = get_dataset(config.dataset_config)

        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm, "prm": prm},
            desc="Running search",
            load_from_cache_file=False,
        )
        logger.info("Starting to stop profile")

        dataset = score(dataset, config)

        dataset_path = save_dataset(dataset, config, run.id)
        output_file: Path = dataset_path.parent / "score.jsonl"

        evaluate(
            benchmark=config.evaluation_config.benchmark,
            dataset_path=dataset_path,
            dataset_col=config.evaluation_config.dataset_col,
            output_file=output_file,
        )
        logger.info("Done 🔥!")
