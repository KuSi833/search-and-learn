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

import pyrootutils
import torch
import wandb
from torch.profiler import record_function
from vllm import LLM

from sal.config import Config
from sal.evaluation.evaluate import evaluate
from sal.models.reward_models import load_prm
from sal.profiler import Profiler
from sal.search import beam_search, best_of_n, dvts, qcts
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
    "qcts": qcts,
}


class TestTimeComputeRunner:
    def __init__(self, config: Config):
        self.config = config

        wandb.login(key=get_env_or_throw("WANDB_API_KEY"))
        self.wandb_run = wandb.init(
            project=config.wandb_config.project,
            config=asdict(config),
            tags=list(config.wandb_config.tags),
        )

        self.output_dir = self._set_up_output_dir(self.wandb_run.id)
        self.inference_output_path = (
            self.output_dir / config.output_config.inference_output_file
        )
        self.evaluation_score_path = (
            self.output_dir / config.output_config.evaluation_score_file
        )

        self.profiler = Profiler(config.profiler_config, self.output_dir)

    def run(self):
        self.profiler.start_memory_profiling()
        with self.profiler.get_pytorch_profiler() as prof:
            self._run_inference(prof)
        self.profiler.finish_memory_profiling()

        self._evaluate_score()
        self._finish()

    def _set_up_output_dir(self, run_id: str) -> Path:
        output_dir = root / self.config.output_config.output_dir_base / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _evaluate_score(self):
        logger.info("Evaluating...")
        evaluate(
            benchmark=self.config.evaluation_config.benchmark,
            dataset_path=self.inference_output_path,
            dataset_col=self.config.evaluation_config.dataset_col,
            output_file=self.evaluation_score_path,
        )

    def _run_inference(self, prof):
        torch.cuda.reset_peak_memory_stats()
        baseline = self.profiler.get_total_gpu_memory_gb()

        prof.step()
        with record_function("load_llm"):
            logger.info("Loading LLM...")
            llm = LLM(
                model=self.config.generator_config.get_model_path(),
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                enable_prefix_caching=True,
                seed=self.config.search_config.seed,
                tensor_parallel_size=1,
                max_model_len=self.config.generator_config.max_model_len,
                enforce_eager=True,
            )
            if self.config.draft_config is not None:
                draft_llm = LLM(
                    model=self.config.generator_config.get_model_path(),
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                    enable_prefix_caching=True,
                    seed=self.config.search_config.seed,
                    tensor_parallel_size=1,
                    max_model_len=self.config.generator_config.max_model_len,
                    enforce_eager=True,
                )

            llm_memory = self.profiler.get_total_gpu_memory_gb() - baseline

        prof.step()
        with record_function("load_prm"):
            logger.info("Loading PRM...")
            prm = load_prm(self.config.prm_config)
            prm_memory = self.profiler.get_total_gpu_memory_gb() - baseline - llm_memory

        prof.step()
        with record_function("load_dataset"):
            logger.info("Loading dataset...")
            dataset = get_dataset(self.config.dataset_config)

        # Reset peak tracking for inference
        torch.cuda.reset_peak_memory_stats()
        pre_inference = self.profiler.get_gpu_memory_gb()

        prof.step()
        with record_function("inference"):
            logger.info("Running inference...")
            approach_fn = APPROACHES[self.config.approach]

            fn_kwargs = {"config": self.config, "prm": prm}
            if self.config.draft_config is not None:
                fn_kwargs.update({"target_llm": llm, "draft_llm": draft_llm})
            else:
                fn_kwargs["llm"] = llm
            dataset = dataset.map(
                approach_fn,
                batched=True,
                batch_size=self.config.search_config.search_batch_size,
                fn_kwargs=fn_kwargs,
                desc="Running search",
                load_from_cache_file=False,
            )

            inference_overhead = self.profiler.get_peak_gpu_memory_gb() - pre_inference

        prof.step()
        with record_function("scoring"):
            logger.info("Scoring...")
            dataset = score(dataset, self.config)
            save_inference_output(dataset, self.inference_output_path)

        self.wandb_run.log(
            {
                "memory/llm_gb": llm_memory,
                "memory/prm_gb": prm_memory,
                "memory/inference_overhead_gb": inference_overhead,
            }
        )
        logger.info(
            f"Memory - LLM: {llm_memory:.2f}GB, PRM: {prm_memory:.2f}GB, Inference: {inference_overhead:.2f}GB"
        )

    def _finish(self):
        logger.info("Done 🔥!")
        self.wandb_run.finish()


def run(config: Config):
    experiment = TestTimeComputeRunner(config)
    experiment.run()
