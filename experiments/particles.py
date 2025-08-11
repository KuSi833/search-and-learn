import copy

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

    base_experiment_config = ExperimentConfig(
        approach="particles",
        search_config=SearchConfig(
            temperature=0.7,
            top_p=0.8,
            prm_batch_size=4,
            search_batch_size=25,
            max_tokens=2048,
            agg_strategy="prod",
        ),
        particles_config=ParticlesConfig(
            min_iterations=0,
            allow_completed_ancestors=True,
            debug_enable=True,
        ),
        wandb_config=WANDB_CONFIG,
    )

    n_values = [8, 16]
    taus = [1, 2.0]
    exp_list = []
    for n in n_values:
        for tau in taus:
            cfg = copy.deepcopy(base_experiment_config)
            cfg.search_config.n = n
            cfg.particles_config.resampling_temperature = tau
            exp_list.append(cfg)

    run(BASE_CONFIG, exp_list)
