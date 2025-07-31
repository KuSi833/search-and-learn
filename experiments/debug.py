from dataclasses import replace

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
from sal.test_time_compute import run

# MODELS
TINY_GENERATOR = GeneratorConfig(
    name="Qwen/Qwen2-Math-1.5B-Instruct",
    parameter_count="1.5B",
)
TINY_PRM_CONFIG = PRMConfig(path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B")

WANDB_CONFIG = WandbConfig(tags=set(["debug"]))

BASE_CONFIG = BaseConfig(
    prm_config=TINY_PRM_CONFIG,
    dataset_config=DatasetConfig(num_samples=1),
    seed=0,
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

BEST_OF_N_CONFIG = ExperimentConfig(
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

    experiment_config = replace(BEAM_SEARCH_CONFIG, generator_config=TINY_GENERATOR)
    run(BASE_CONFIG, [experiment_config])
