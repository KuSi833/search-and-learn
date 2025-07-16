from sal.test_time_compute import main
from sal.config import Config, SearchConfig, DatasetConfig, GeneratorConfig, PRMConfig
from dataclasses import replace

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

PRM_CONFIG = PRMConfig(path="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")
# DATASET_CONFIG = DatasetConfig(num_samples=100)
DATASET_CONFIG = DatasetConfig(num_samples=1)

BEAM_SEARCH_CONFIG = Config(
    prm_config=PRM_CONFIG,
    filter_duplicates=True,
    approach="beam_search",
    search_config=SearchConfig(
        n=4,
        search_batch_size=1,
        seed=0,
    ),
    dataset_config=DATASET_CONFIG,
)

BEST_OF_N_CONFIG = Config(
    prm_config=PRM_CONFIG,
    filter_duplicates=True,
    sort_completed=True,
    approach="best_of_n",
    search_config=SearchConfig(
        n=4,
        search_batch_size=25,
        seed=0,
    ),
    dataset_config=DATASET_CONFIG,
)

if __name__ == "__main__":
    # config = replace(BEST_OF_N_CONFIG, generator_config=Q8_MODEL)
    # config = replace(BEAM_SEARCH_CONFIG, generator_config=Q8_MODEL)
    config = replace(BEST_OF_N_CONFIG, generator_config=Q4_MODEL)
    # config = replace(BEAM_SEARCH_CONFIG, generator_config=Q4_MODEL)
    # config = replace(BEST_OF_N_CONFIG, generator_config=BASE_MODEL)
    # config = replace(BEAM_SEARCH_CONFIG, generator_config=BASE_MODEL)
    # config = replace(BEST_OF_N_CONFIG, generator_config=INSTRUCT_MODEL)
    # config = replace(BEAM_SEARCH_CONFIG, generator_config=INSTRUCT_MODEL)
    main(config)
