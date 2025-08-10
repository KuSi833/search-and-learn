import copy
from typing import List

from dotenv import load_dotenv

from sal.config import (
    BaseConfig,
    DatasetConfig,
    ExperimentConfig,
    GeneratorConfig,
    PRMConfig,
    QCConfig,
    SearchConfig,
    WandbConfig,
)
from sal.test_time_compute import run
from sal.utils.experiment import get_model_base_path

if __name__ == "__main__":
    load_dotenv()

    model_base_path = get_model_base_path()

    INSTRUCT_MODEL = GeneratorConfig(
        base_path=model_base_path,
        name="Qwen/Qwen2.5-Math-7B-Instruct",
        parameter_count="7B",
        gpu_memory_utilization=0.3,
    )
    # BASE_MODEL = GeneratorConfig(
    #     name="Qwen/Qwen2.5-Math-7B",
    #     parameter_count="7B",
    # )
    Q8_MODEL = GeneratorConfig(
        base_path=model_base_path,
        name="quant_factory/Qwen2.5-Math-7B.Q8_0.gguf",
        parameter_count="7B",
        quantisation="Q8_0",
        gpu_memory_utilization=0.2,
    )
    Q4_MODEL = GeneratorConfig(
        base_path=model_base_path,
        # name="quant_factory/Qwen2.5-Math-7B.Q4_0.gguf",
        name="quant_factory/Qwen2.5-Math-7B.Q4_K_M.gguf",
        parameter_count="7B",
        quantisation="Q4_0",
        gpu_memory_utilization=0.2,
    )

    # PRM_CONFIG = PRMConfig(path="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")
    PRM_CONFIG = PRMConfig(
        base_path=model_base_path,
        name="Qwen/Qwen2.5-Math-PRM-7B",
    )

    DATASET_CONFIG = DatasetConfig(
        num_samples=100,
        # num_samples=25,
        # num_samples=1,
    )

    WANDB_CONFIG = WandbConfig(tags=set(["q2 sweep"]))

    BASE_CONFIG = BaseConfig(
        prm_config=PRM_CONFIG,
        # generator_config=Q8_MODEL,
        generator_config=INSTRUCT_MODEL,
        draft_config=Q8_MODEL,
        # draft_config=Q8_MODEL,
        dataset_config=DATASET_CONFIG,
    )

    QC_CONFIG = ExperimentConfig(
        filter_duplicates=True,
        approach="qcts",
        wandb_config=WANDB_CONFIG,
        search_config=SearchConfig(
            n=4,
            temperature=0.7,  # Their exact setting (you had 0.8)
            top_p=0.8,  # Their exact setting (you had 1.0)
            prm_batch_size=4,
            search_batch_size=50,
            max_tokens=2048,
            agg_strategy="prod",
        ),
        qcconfig=QCConfig(
            low_threshold=0.4,
            high_threshold=0.9,
        ),
    )

    Q2_CONFIG = ExperimentConfig(
        filter_duplicates=True,
        approach="q2",
        wandb_config=WANDB_CONFIG,
        search_config=SearchConfig(
            n=4,
            temperature=0.7,  # Their exact setting (you had 0.8)
            top_p=0.8,  # Their exact setting (you had 1.0)
            prm_batch_size=4,
            search_batch_size=50,
            max_tokens=2048,
            agg_strategy="prod",
        ),
        qcconfig=QCConfig(
            target_upgrade_rate=0.35,
        ),
    )

    experiment_configs: List[ExperimentConfig] = []

    experiment_configs.append(Q2_CONFIG)
    experiment_configs.append(QC_CONFIG)

    run(BASE_CONFIG, experiment_configs)
