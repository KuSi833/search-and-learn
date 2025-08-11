from dotenv import load_dotenv

from sal.config import (
    BaseConfig,
    DatasetConfig,
    ExperimentConfig,
    GeneratorConfig,
    ParticlesConfig,
    PRMConfig,
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
    )

    PRM_CONFIG = PRMConfig(
        base_path=model_base_path,
        name="Qwen/Qwen2.5-Math-PRM-7B",
    )

    WANDB_CONFIG = WandbConfig(tags=set(["particles"]))
    DATASET_CONFIG = DatasetConfig(num_samples=100)

    BASE_CONFIG = BaseConfig(
        prm_config=PRM_CONFIG,
        generator_config=INSTRUCT_MODEL,
        dataset_config=DATASET_CONFIG,
    )

    # Simple sweep over resampling temperatures
    temps = [0.7, 1.0, 1.3, 1.6, 2.0]
    exp_list = []
    for tau in temps:
        cfg = ExperimentConfig(
            approach="particles",
            custom_chat_template=None,
            search_config=SearchConfig(
                n=4,
                temperature=0.7,
                top_p=0.8,
                prm_batch_size=4,
                search_batch_size=25,
                max_tokens=2048,
                agg_strategy="prod",
            ),
            particles_config=ParticlesConfig(
                resampling_temperature=tau,
                min_iterations=0,
                allow_completed_ancestors=True,
            ),
            wandb_config=WANDB_CONFIG,
        )
        exp_list.append(cfg)

    run(BASE_CONFIG, exp_list)
