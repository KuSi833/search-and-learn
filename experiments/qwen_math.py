import copy
from typing import List

from dotenv import load_dotenv
from networkx import expected_degree_graph

from sal.config import (
    BaseConfig,
    BeamSearchConfig,
    DatasetConfig,
    ExperimentConfig,
    GeneratorConfig,
    PRMConfig,
    SearchConfig,
    WandbConfig,
)
from sal.test_time_compute import run
from sal.utils.experiment import get_math500_indices, get_model_base_path

if __name__ == "__main__":
    load_dotenv()

    model_base_path = get_model_base_path()

    INSTRUCT_MODEL = GeneratorConfig(
        # name="Qwen/Qwen2.5-Math-7B-Instruct",
        base_path=model_base_path,
        name="Qwen/Qwen2.5-Math-7B-Instruct",
        parameter_count="7B",
    )
    BASE_MODEL = GeneratorConfig(
        name="Qwen/Qwen2.5-Math-7B",
        parameter_count="7B",
    )
    Q8_MODEL = GeneratorConfig(
        base_path=model_base_path,
        name="quant_factory/Qwen2.5-Math-7B.Q8_0.gguf",
        parameter_count="7B",
        quantisation="Q8_0",
    )
    Q4_MODEL = GeneratorConfig(
        base_path=model_base_path,
        name="quant_factory/Qwen2.5-Math-7B.Q4_0.gguf",
        parameter_count="7B",
        quantisation="Q4_0",
    )
    SMALLEST_MODEL = GeneratorConfig(
        name="Qwen/Qwen2-Math-1.5B-Instruct",
        parameter_count="1.5B",
    )

    # PRM_CONFIG = PRMConfig(path="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")
    PRM_CONFIG = PRMConfig(
        base_path=model_base_path,
        name="Qwen/Qwen2.5-Math-PRM-7B",
    )
    # PRM_CONFIG = PRMConfig(path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B")

    # WANDB_CONFIG = WandbConfig(tags=set(["baseline"]))
    WANDB_CONFIG = WandbConfig(tags=set([]))

    # DATASET_CONFIG = DatasetConfig(num_samples=100)
    # DATASET_CONFIG = DatasetConfig(dataset_indicies=get_math500_indices(subset="hard"))
    # DATASET_CONFIG = DatasetConfig(
    #     dataset_indicies=get_math500_indices(subset="bon_hard_2")
    # )
    DATASET_CONFIG = DatasetConfig(num_samples=500)  # FULL DATASET
    # DATASET_CONFIG = DatasetConfig(
    #     dataset_name="HuggingFaceH4/aime_2024"
    # )  # FULL DATASET

    BASE_CONFIG = BaseConfig(
        prm_config=PRM_CONFIG,
        # generator_config=Q8_MODEL,
        generator_config=INSTRUCT_MODEL,
        dataset_config=DATASET_CONFIG,
    )

    BEAM_SEARCH_CONFIG = ExperimentConfig(
        filter_duplicates=True,
        approach="beam_search",
        search_config=SearchConfig(
            n=4,
            temperature=0.7,  # Their exact setting (you had 0.8)
            top_p=0.8,  # Their exact setting (you had 1.0)
            prm_batch_size=4,
            search_batch_size=1,
            max_tokens=2048,
            agg_strategy="prod",
        ),
        beam_search_config=BeamSearchConfig(
            beam_width=4,
        ),
        wandb_config=WANDB_CONFIG,
    )

    # BEST_OF_N_CONFIG = ExperimentConfig(
    #     filter_duplicates=True,
    #     sort_completed=True,
    #     approach="best_of_n",
    #     search_config=SearchConfig(
    #         n=4,
    #         search_batch_size=50,
    #     ),
    #     wandb_config=WANDB_CONFIG,
    # )

    # DVTS_CONFIG = ExperimentConfig(
    #     approach="dvts",
    #     custom_chat_template=None,
    #     search_config=SearchConfig(
    #         n=4,
    #         # search_batch_size=10,
    #         search_batch_size=50,
    #     ),
    #     wandb_config=WANDB_CONFIG,
    # )

    DVTS_CONFIG = ExperimentConfig(
        approach="dvts",
        custom_chat_template=None,
        search_config=SearchConfig(
            n=4,
            temperature=0.7,  # Their exact setting (you had 0.8)
            top_p=0.8,  # Their exact setting (you had 1.0)
            prm_batch_size=4,
            search_batch_size=50,
            max_tokens=2048,
            agg_strategy="prod",
        ),
        wandb_config=WANDB_CONFIG,
    )

    BEST_OF_N_CONFIG = ExperimentConfig(
        filter_duplicates=True,
        sort_completed=True,
        approach="best_of_n",
        search_config=SearchConfig(
            n=4,
            temperature=0.7,  # Their exact setting (you had 0.8)
            top_p=0.8,  # Their exact setting (you had 1.0)
            prm_batch_size=4,
            search_batch_size=50,
            max_tokens=2048,
            agg_strategy="prod",
        ),
        wandb_config=WANDB_CONFIG,
    )

    experiment_configs: List[ExperimentConfig] = []

    for n in [8]:
        # for n in [8, 16, 32, 64]:
        config_variant = copy.deepcopy(BEST_OF_N_CONFIG)
        config_variant.search_config.n = n
        experiment_configs.append(config_variant)

    # for beam_width in [16, 8, 4]:
    #     config_variant = copy.deepcopy(BEAM_SEARCH_CONFIG)
    #     config_variant.beam_search_config.beam_width = beam_width
    #     experiment_configs.append(config_variant)

    run(BASE_CONFIG, experiment_configs)
