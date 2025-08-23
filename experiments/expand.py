import copy
from typing import List

from dotenv import load_dotenv

from sal.config import (
    BaseConfig,
    DatasetConfig,
    ExperimentConfig,
    GeneratorConfig,
    PRMConfig,
    SearchConfig,
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

BEAM_SEARCH_CONFIG = ExperimentConfig(
    filter_duplicates=True,
    approach="beam_search",
    search_config=SearchConfig(
        n=4,
        search_batch_size=1,
    ),
    wandb_config=WANDB_CONFIG,
)

BASE_BEST_OF_N_CONFIG = ExperimentConfig(
    filter_duplicates=True,
    sort_completed=True,
    approach="best_of_n",
    search_config=SearchConfig(
        n=4,
        search_batch_size=25,
    ),
    wandb_config=WANDB_CONFIG,
)

if __name__ == "__main__":
    load_dotenv()

    experiment_configs: List[ExperimentConfig] = []

    for n in [1]:
        experiment_copy = copy.deepcopy(BASE_BEST_OF_N_CONFIG)
        experiment_copy.search_config.n = n

        experiment_configs.append(experiment_copy)

    run(BASE_CONFIG, experiment_configs)
