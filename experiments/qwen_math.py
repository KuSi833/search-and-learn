import copy
from typing import List

from dotenv import load_dotenv

from sal.config import (
    BaseConfig,
    BeamSearchConfig,
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

    WANDB_CONFIG = WandbConfig(tags=set(["beam_sweep"]))

    PRM_CONFIG = PRMConfig(
        base_path=model_base_path,
        name="Qwen/Qwen2.5-Math-PRM-7B",
    )

    # DATASET_CONFIG = DatasetConfig(num_samples=500)
    # DATASET_CONFIG = DatasetConfig(num_samples=10)
    DATASET_CONFIG = DatasetConfig(
        # dataset_name="HuggingFaceH4/aime_2024",
        dataset_name="HuggingFaceH4/MATH-500",
        dataset_indicies=get_math500_indices(subset="hard_bon"),
    )

    BASE_CONFIG = BaseConfig(
        prm_config=PRM_CONFIG,
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

    BEAM_SEARCH_CONFIG = ExperimentConfig(
        approach="beam_search",
        search_batch_size=1,
        beam=BeamSearchConfig(
            sampling=SamplingConfig(
                n=8,
                temperature=0.7,
                top_p=0.8,
                max_tokens=2048,
                agg_strategy="prod",
            ),
            beam_width=4,
            num_iterations=40,
            filter_duplicates=False,
            sort_completed=False,
            debug=False,
        ),
        wandb_config=WANDB_CONFIG,
    )

    experiment_configs: List[ExperimentConfig] = []

    experiment_configs.append(BEST_OF_N_CONFIG)

    # for n in [4, 8]:
    #     for beam_width in [4, 8, 16]:
    #         config_variant = copy.deepcopy(BEAM_SEARCH_CONFIG)
    #         config_variant.beam.sampling.n = n
    #         config_variant.beam.beam_width = beam_width
    #         experiment_configs.append(config_variant)

    run(BASE_CONFIG, experiment_configs)
