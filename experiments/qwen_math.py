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

    WANDB_CONFIG = WandbConfig(tags=set(["baseline"]))

    # Run full MATH-500 (no index restriction)
    DATASET_CONFIG = DatasetConfig(num_samples=500)
    # DATASET_CONFIG = DatasetConfig(num_samples=10)
    # DATASET_CONFIG = DatasetConfig(
    #     dataset_name="HuggingFaceH4/aime_2024"
    # )  # FULL DATASET

    BASE_CONFIG = BaseConfig(
        prm_config=PRM_CONFIG,
        # generator_config=Q8_MODEL,
        generator_config=INSTRUCT_MODEL,
        dataset_config=DATASET_CONFIG,
    )

    # Best-of-N baseline
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

    # Optional: DVTS or other strategies can be re-added here with their own configs

    experiment_configs: List[ExperimentConfig] = []

    for n in [4]:
        config_variant = copy.deepcopy(BEST_OF_N_CONFIG)
        config_variant.bon.sampling.n = n
        experiment_configs.append(config_variant)

    # Also include the base n=4
    experiment_configs.append(BEST_OF_N_CONFIG)

    run(BASE_CONFIG, experiment_configs)
