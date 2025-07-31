from dataclasses import asdict, replace

import wandb
from dotenv import load_dotenv

from sal.config import (
    DatasetConfig,
    ExperimentConfig,
    GeneratorConfig,
    PRMConfig,
    SearchConfig,
)
from sal.utils.env import get_env_or_throw

load_dotenv()

MODEL_BASE_PATH = "/vol/bitbucket/km1124/search-and-learn/models/"

Q8_MODEL = GeneratorConfig(
    base_path=MODEL_BASE_PATH,
    name="quant_factory/Qwen2.5-Math-7B.Q8_0.gguf",
    parameter_count="7B",
    quantisation="Q8_0",
)

PRM_CONFIG = PRMConfig(path="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")

BEAM_SEARCH_CONFIG = ExperimentConfig(
    prm_config=PRM_CONFIG,
    filter_duplicates=True,
    approach="beam_search",
    search_config=SearchConfig(
        n=4,
        search_batch_size=1,
        seed=0,
    ),
    dataset_config=DatasetConfig(
        num_samples=100,
    ),
)

BEST_OF_N_CONFIG = ExperimentConfig(
    prm_config=PRM_CONFIG,
    filter_duplicates=True,
    sort_completed=True,
    approach="best_of_n",
    search_config=SearchConfig(
        n=4,
        search_batch_size=25,
        seed=0,
    ),
    dataset_config=DatasetConfig(
        num_samples=100,
    ),
)


def main(config: ExperimentConfig):
    wandb.login(key=get_env_or_throw("WANDB_API_KEY"))

    with wandb.init(
        project=config.wandb_config.project,
        config=asdict(config),
        tags=list(config.wandb_config.tags),
    ) as run:
        print(run.name)


if __name__ == "__main__":
    config = replace(BEST_OF_N_CONFIG, generator_config=Q8_MODEL)
    main(config)
