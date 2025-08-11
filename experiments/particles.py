import copy
from typing import List

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

    # WANDB_CONFIG = WandbConfig(tags=set(["particles", "diagnostic"]))
    WANDB_CONFIG = WandbConfig(tags=set(["particles", "agg_strat_sweep"]))
    DATASET_CONFIG = DatasetConfig(num_samples=100)
    # DATASET_CONFIG = DatasetConfig(num_samples=25)

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
            search_batch_size=10,
            max_tokens=2048,
            agg_strategy="mean",
        ),
        particles_config=ParticlesConfig(
            min_iterations=3,
            allow_completed_ancestors=False,
            resampling_temperature=1.5,
            temperature_jitter_std=0.2,
            score_noise_std=0.02,
            resampling_method="systematic",
            debug_enable=True,
        ),
        wandb_config=WANDB_CONFIG,
    )

    exp_list: List[ExperimentConfig] = [base_experiment_config]

    for agg_strategy in ["sum", "last", "prod"]:
        cfg = copy.deepcopy(base_experiment_config)
        cfg.search_config.agg_strategy = agg_strategy
        # Keep diversity-friendly particle defaults; tag the run for clarity
        cfg.wandb_config.tags.add(f"agg={agg_strategy}")
        cfg.wandb_config.tags.add("anti_takeover")
        cfg.wandb_config.tags.add("tau_1p5")
        exp_list.append(cfg)

    run(BASE_CONFIG, exp_list)
