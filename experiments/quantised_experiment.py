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

from .util import get_model_base_path

if __name__ == "__main__":
    load_dotenv()

    model_base_path = get_model_base_path()

    # INSTRUCT_MODEL = GeneratorConfig(
    #     name="Qwen/Qwen2.5-Math-7B-Instruct",
    #     parameter_count="7B",
    # )
    # BASE_MODEL = GeneratorConfig(
    #     name="Qwen/Qwen2.5-Math-7B",
    #     parameter_count="7B",
    # )
    Q8_MODEL = GeneratorConfig(
        base_path=model_base_path,
        name="quant_factory/Qwen2.5-Math-7B.Q8_0.gguf",
        parameter_count="7B",
        quantisation="Q8_0",
        gpu_memory_utilization=0.3,
    )
    Q4_MODEL = GeneratorConfig(
        base_path=model_base_path,
        name="quant_factory/Qwen2.5-Math-7B.Q4_0.gguf",
        parameter_count="7B",
        quantisation="Q4_0",
        gpu_memory_utilization=0.3,
    )

    # PRM_CONFIG = PRMConfig(path="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")
    PRM_CONFIG = PRMConfig(path="Qwen/Qwen2.5-Math-PRM-7B")

    DATASET_CONFIG = DatasetConfig(
        num_samples=100,
        # num_samples=1,
    )

    WANDB_CONFIG = WandbConfig(tags=set(["qc_v1"]))

    BASE_CONFIG = BaseConfig(
        prm_config=PRM_CONFIG,
        generator_config=Q8_MODEL,
        draft_config=Q4_MODEL,
        dataset_config=DATASET_CONFIG,
    )

    QC_CONFIG = ExperimentConfig(
        filter_duplicates=True,
        approach="qcts",
        wandb_config=WANDB_CONFIG,
        search_config=SearchConfig(
            n=4,
            search_batch_size=1,
        ),
    )
    run(BASE_CONFIG, [QC_CONFIG])
