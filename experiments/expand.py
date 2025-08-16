import copy
from typing import List

from dotenv import load_dotenv

from sal.config import (
    BaseConfig,
    BestOfNConfig,
    DatasetConfig,
    ExperimentConfig,
    GeneratorConfig,
    PRMConfig,
    SamplingConfig,
    WandbConfig,
)
from sal.test_time_compute_new import run

# MODELS
TINY_GENERATOR = GeneratorConfig(
    name="Qwen/Qwen2-Math-1.5B-Instruct",
    parameter_count="1.5B",
)
TINY_PRM_CONFIG = PRMConfig(name="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B")

WANDB_CONFIG = WandbConfig(tags=set(["debug"]))

BASE_CONFIG = BaseConfig(
    prm_config=TINY_PRM_CONFIG,
    generator_config=TINY_GENERATOR,
    dataset_config=DatasetConfig(num_samples=1),
)

BEST_OF_N_CONFIG = ExperimentConfig(
    approach="best_of_n",
    bon=BestOfNConfig(
        sampling=SamplingConfig(
            n=4,
            temperature=0.7,
            top_p=0.8,
            max_tokens=2048,
            agg_strategy="prod",
        ),
        debug=False,
    ),
    search_batch_size=50,
    wandb_config=WANDB_CONFIG,
)

if __name__ == "__main__":
    load_dotenv()

    experiment_configs: List[ExperimentConfig] = []

    for n in [1]:
        experiment_copy = copy.deepcopy(BEST_OF_N_CONFIG)

        experiment_configs.append(experiment_copy)

    run(BASE_CONFIG, experiment_configs)
