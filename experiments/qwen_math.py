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
from sal.test_time_compute import run

MODEL_BASE_PATH = "/vol/bitbucket/km1124/search-and-learn/models"

INSTRUCT_MODEL = GeneratorConfig(
    name="Qwen/Qwen2.5-Math-7B-Instruct",
    parameter_count="7B",
)
BASE_MODEL = GeneratorConfig(
    name="Qwen/Qwen2.5-Math-7B",
    parameter_count="7B",
)
Q8_MODEL = GeneratorConfig(
    base_path=MODEL_BASE_PATH,
    name="quant_factory/Qwen2.5-Math-7B.Q8_0.gguf",
    parameter_count="7B",
    quantisation="Q8_0",
)
Q4_MODEL = GeneratorConfig(
    base_path=MODEL_BASE_PATH,
    name="quant_factory/Qwen2.5-Math-7B.Q4_0.gguf",
    parameter_count="7B",
    quantisation="Q4_0",
)
SMALLEST_MODEL = GeneratorConfig(
    name="Qwen/Qwen2-Math-1.5B-Instruct",
    parameter_count="1.5B",
)


# PRM_CONFIG = PRMConfig(path="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")
PRM_CONFIG = PRMConfig(path="Qwen/Qwen2.5-Math-PRM-7B")
# PRM_CONFIG = PRMConfig(path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B")

WANDB_CONFIG = WandbConfig(tags=set([]))

BASE_CONFIG = BaseConfig(
    prm_config=PRM_CONFIG,
    generator_config=Q8_MODEL,
    dataset_config=DatasetConfig(num_samples=100),
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

    experiment_configs: List[ExperimentConfig] = []

    experiment_configs.append(BEST_OF_N_CONFIG)
    experiment_configs.append(BEAM_SEARCH_CONFIG)

    run(BASE_CONFIG, experiment_configs)
