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
from sal.utils.experiment import (
    get_math500_indices,
    get_model_base_path,
)

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

    WANDB_CONFIG = WandbConfig(tags=set(["particles", "ps1"]))
    # DATASET_CONFIG = DatasetConfig(num_samples=100)
    # DATASET_CONFIG = DatasetConfig(num_samples=500)
    DATASET_CONFIG = DatasetConfig(
        benchmark_indicies=get_math500_indices(subset="hard"),
        # dataset_indicies=get_math500_indices(subset="crash_debug"),
    )
    # DATASET_CONFIG = DatasetConfig(
    #     num_samples=100,
    #     dataset_start=40,
    #     dataset_end=45,
    # )

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
            search_batch_size=1,
            max_tokens=2048,
            agg_strategy="prod",
        ),
        particles_config=ParticlesConfig(
            min_iterations=0,
            allow_completed_ancestors=False,
            debug_enable=False,
        ),
        wandb_config=WANDB_CONFIG,
    )

    def with_tags(cfg: ExperimentConfig, tags: List[str]) -> ExperimentConfig:
        cfg2 = copy.deepcopy(cfg)
        cfg2.wandb_config.tags = set(
            list(cfg2.wandb_config.tags) + tags + ["particles", "sweep"]
        )  # type: ignore[arg-type]
        return cfg2

    exp_list: List[ExperimentConfig] = []

    # 0) Baseline
    exp_list.append(with_tags(base_experiment_config, ["baseline"]))

    # 1) Vary number of particles n
    for n in [8, 16]:
        cfg = copy.deepcopy(base_experiment_config)
        cfg.search_config.n = n
        exp_list.append(with_tags(cfg, [f"n{n}"]))

    # 2) Aggregation strategy variants
    for agg in ["last", "sum", "mean"]:
        cfg = copy.deepcopy(base_experiment_config)
        cfg.search_config.agg_strategy = agg  # type: ignore[assignment]
        cfg.search_config.n = 8
        exp_list.append(with_tags(cfg, [f"agg_{agg}", "n8"]))

    # 3) Resampling method and temperature
    for method in ["multinomial", "systematic"]:
        for tau in [0.5, 1.0, 2.0]:
            cfg = copy.deepcopy(base_experiment_config)
            cfg.search_config.n = 8
            cfg.particles_config.resampling_method = method  # type: ignore[assignment]
            cfg.particles_config.resampling_temperature = float(tau)
            exp_list.append(with_tags(cfg, [f"resamp_{method}", f"tau{tau}", "n8"]))

    # 4) Iteration budget (deeper rollouts)
    for iters in [20, 40, 60]:
        cfg = copy.deepcopy(base_experiment_config)
        cfg.beam_search_config.num_iterations = iters
        cfg.search_config.n = 8
        exp_list.append(with_tags(cfg, [f"iters{iters}", "n8"]))

    # 5) Allow completed ancestors vs not, and enforce minimum iterations
    for allow in [False, True]:
        for min_it in [0, 2, 4]:
            cfg = copy.deepcopy(base_experiment_config)
            cfg.particles_config.allow_completed_ancestors = allow
            cfg.particles_config.min_iterations = min_it
            cfg.search_config.n = 8
            exp_list.append(
                with_tags(
                    cfg,
                    [
                        "allow_done" if allow else "no_done",
                        f"minit{min_it}",
                        "n8",
                    ],
                )
            )

    # 6) Exploration parameters (temperature, top_p)
    for temp, top_p in [(0.5, 0.95), (0.7, 0.95), (0.6, 0.9)]:
        cfg = copy.deepcopy(base_experiment_config)
        cfg.search_config.temperature = float(temp)
        cfg.search_config.top_p = float(top_p)
        cfg.search_config.n = 8
        exp_list.append(with_tags(cfg, [f"T{temp}", f"p{top_p}", "n8"]))

    # 7) Diversity and anti-collapse tweaks (small jitter/noise)
    for jit, sn in [(0.0, 0.05), (0.03, 0.0), (0.05, 0.03)]:
        cfg = copy.deepcopy(base_experiment_config)
        cfg.search_config.n = 8
        cfg.particles_config.temperature_jitter_std = float(jit)
        cfg.particles_config.score_noise_std = float(sn)
        cfg.particles_config.resampling_method = "systematic"  # type: ignore[assignment]
        cfg.particles_config.resampling_temperature = 0.8
        exp_list.append(with_tags(cfg, [f"jit{jit}", f"sn{sn}", "sys", "tau0.8", "n8"]))

    # 8) Stronger search: combine several beneficial settings
    strong_cfg = copy.deepcopy(base_experiment_config)
    strong_cfg.search_config.n = 16
    strong_cfg.search_config.temperature = 0.6
    strong_cfg.search_config.top_p = 0.95
    strong_cfg.search_config.agg_strategy = "last"  # type: ignore[assignment]
    strong_cfg.beam_search_config.num_iterations = 60
    strong_cfg.particles_config.resampling_method = "systematic"  # type: ignore[assignment]
    strong_cfg.particles_config.resampling_temperature = 0.7
    strong_cfg.particles_config.temperature_jitter_std = 0.03
    strong_cfg.particles_config.score_noise_std = 0.0
    strong_cfg.particles_config.min_iterations = 2
    strong_cfg.particles_config.allow_completed_ancestors = False
    exp_list.append(
        with_tags(
            strong_cfg,
            [
                "strong",
                "n16",
                "T0.6",
                "p0.95",
                "agg_last",
                "iters60",
                "sys",
                "tau0.7",
                "jit0.03",
                "minit2",
                "no_done",
            ],
        )
    )

    run(BASE_CONFIG, exp_list)
