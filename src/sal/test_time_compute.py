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
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import pyrootutils
import torch
import wandb
from torch.profiler import record_function
from vllm import LLM  # type: ignore

from sal.config import (
    BaseConfig,
    DatasetConfig,
    ExperimentConfig,
    GeneratorConfig,
    PRMConfig,
)
from sal.evaluation.evaluate import evaluate
from sal.models.reward_models import load_prm
from sal.profiler import Profiler
from sal.search import (
    beam_search,
    best_of_n,
    cgai,
    diagnostic_tts,
    dvts,
    gibbs,
    particles,
    q2,
    qcts,
)
from sal.utils.data import get_dataset, save_inference_output
from sal.utils.env import get_env_or_throw
from sal.utils.logging import setup_logging
from sal.utils.score import score

setup_logging()

root = pyrootutils.find_root(indicator="pyproject.toml")

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root path. This should be used by experiment files."""
    return root


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "cgai": cgai,
    "qcts": qcts,
    "q2": q2,
    "diagnostic_tts": diagnostic_tts,
    "particles": particles,
    "gibbs": gibbs,
}


class ExperimentRunner:
    def __init__(self, base_config: BaseConfig):
        self.base_config = base_config

        wandb.login(key=get_env_or_throw("WANDB_API_KEY"))
        self.profiler = Profiler(base_config.profiler_config)

    def run_experiments(self, experiment_configs: List[ExperimentConfig]):
        self.llm = self._load_llm(self.base_config.generator_config)
        self.maybe_draft_llm = self._maybe_load_draft_llm(self.base_config.draft_config)
        self.prm = self._load_prm(self.base_config.prm_config)
        self.dataset = self._load_dataset(self.base_config.dataset_config)
        # Add a stable index column for logging/selection reporting
        self.dataset = self.dataset.map(
            lambda x, idx: {"idx": idx},
            with_indices=True,
            desc="Indexing dataset",
            load_from_cache_file=False,
        )

        for experiment_config in experiment_configs:
            self._run_single_experiment(experiment_config)

    def _run_single_experiment(
        self,
        experiment_config: ExperimentConfig,
    ):
        with wandb.init(
            project=experiment_config.wandb_config.project,
            tags=list(experiment_config.wandb_config.tags),
            config=asdict(self.base_config) | asdict(experiment_config),
        ) as run:
            output_dir = self._set_up_output_dir(run.id)
            # TODO: could probably do this better with the profiler output dir
            self.profiler.set_output_dir(output_dir)
            inference_output_path = (
                output_dir / self.base_config.output_config.inference_output_file
            )
            evaluation_score_path = (
                output_dir / self.base_config.output_config.evaluation_score_file
            )

            self.profiler.start_memory_profiling()
            with self.profiler.get_pytorch_profiler() as pytorch_profiler:
                self._run_inference(
                    experiment_config,
                    pytorch_profiler,
                    inference_output_path,
                    output_dir,
                )
            self.profiler.finish_memory_profiling(run)

            self._evaluate_score(inference_output_path, evaluation_score_path)
            logger.info("Done 🔥!")

    def _load_llm(self, generator_config: GeneratorConfig):
        logger.info("Loading LLM...")

        torch.cuda.reset_peak_memory_stats()
        baseline = self.profiler.get_total_gpu_memory_gb()

        with record_function("load_llm"):
            llm = LLM(
                model=generator_config.get_model_path(),
                gpu_memory_utilization=generator_config.gpu_memory_utilization,
                enable_prefix_caching=True,
                seed=self.base_config.seed,
                tensor_parallel_size=1,
                max_model_len=generator_config.max_model_len,
                enforce_eager=self.base_config.enforce_eager,
            )
            self.profiler.memory_metrics.baseline_gb = baseline
            self.profiler.memory_metrics.llm_memory_gb = (
                self.profiler.get_total_gpu_memory_gb() - baseline
            )
        return llm

    def _maybe_load_draft_llm(
        self, draft_config: Optional[GeneratorConfig]
    ) -> Optional[LLM]:
        if draft_config is None:
            return None

        logger.info("Loading draft LLM...")
        with record_function("load_draft_llm"):
            draft_llm = LLM(
                model=draft_config.get_model_path(),
                gpu_memory_utilization=draft_config.gpu_memory_utilization,
                enable_prefix_caching=True,
                seed=self.base_config.seed,
                tensor_parallel_size=1,
                max_model_len=draft_config.max_model_len,
                enforce_eager=self.base_config.enforce_eager,
            )

            baseline = (
                self.profiler.memory_metrics.baseline_gb
                + self.profiler.memory_metrics.llm_memory_gb
            )
            self.profiler.memory_metrics.draft_llm_memory_gb = (
                self.profiler.get_total_gpu_memory_gb() - baseline
            )
        return draft_llm

    def _load_prm(self, prm_config: PRMConfig):
        with record_function("load_prm"):
            logger.info("Loading PRM...")
            prm = load_prm(prm_config)

            baseline = (
                self.profiler.memory_metrics.baseline_gb
                + self.profiler.memory_metrics.llm_memory_gb
            )
            if self.profiler.memory_metrics.draft_llm_memory_gb is not None:
                baseline += self.profiler.memory_metrics.draft_llm_memory_gb

            self.profiler.memory_metrics.prm_memory_gb = (
                self.profiler.get_total_gpu_memory_gb() - baseline
            )

        return prm

    def _load_dataset(self, dataset_config: DatasetConfig):
        with record_function("load_dataset"):
            logger.info("Loading dataset...")
            return get_dataset(dataset_config)

    def _set_up_output_dir(self, run_id: str) -> Path:
        output_dir = root / self.base_config.output_config.output_dir_base / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _run_inference(
        self,
        experiment_config: ExperimentConfig,
        pytorch_profiler,
        inference_output_path: Path,
        output_dir: Path,
    ):
        torch.cuda.reset_peak_memory_stats()
        pre_inference = self.profiler.get_gpu_memory_gb()

        pytorch_profiler.step()
        with record_function("inference"):
            logger.info("Running inference...")
            approach_fn = APPROACHES[experiment_config.approach]

            fn_kwargs = {
                "experiment_config": experiment_config,
                "prm": self.prm,
            }
            if experiment_config.approach == "diagnostic_tts":
                fn_kwargs["output_dir"] = output_dir
            if self.maybe_draft_llm is not None:
                fn_kwargs.update(
                    {"target_llm": self.llm, "draft_llm": self.maybe_draft_llm}
                )
            else:
                fn_kwargs["llm"] = self.llm

            start_time = time.time()
            dataset = self.dataset.map(
                approach_fn,
                batched=True,
                batch_size=experiment_config.search_config.search_batch_size,
                fn_kwargs=fn_kwargs,
                desc="Running search",
                load_from_cache_file=False,
            )
            end_time = time.time()
            self.profiler.inference_runtime = end_time - start_time

            inference_overhead = self.profiler.get_peak_gpu_memory_gb() - pre_inference
            self.profiler.memory_metrics.inference_overhead_gb = inference_overhead

        pytorch_profiler.step()
        with record_function("scoring"):
            logger.info("Scoring...")
            dataset = score(
                dataset, experiment_config, self.base_config.output_config.num_proc
            )
            save_inference_output(dataset, inference_output_path)

    def _evaluate_score(self, inference_output_path: Path, evaluation_score_path: Path):
        logger.info("Evaluating...")
        evaluate(
            benchmark=self.base_config.evaluation_config.benchmark,
            dataset_path=inference_output_path,
            dataset_col=self.base_config.evaluation_config.dataset_col,
            output_file=evaluation_score_path,
        )


def run(base_config: BaseConfig, experiment_configs: List[ExperimentConfig]):
    experiment_runner = ExperimentRunner(base_config)
    experiment_runner.run_experiments(experiment_configs)
