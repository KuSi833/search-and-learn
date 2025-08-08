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
        name="Qwen/Qwen2.5-Math-7B-Instruct",
        parameter_count="7B",
        gpu_memory_utilization=0.4,
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
        gpu_memory_utilization=0.3,
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
    PRM_CONFIG = PRMConfig(path="Qwen/Qwen2.5-Math-PRM-7B")

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
        draft_config=Q4_MODEL,
        # draft_config=Q8_MODEL,
        dataset_config=DATASET_CONFIG,
    )

    QC_CONFIG = ExperimentConfig(
        filter_duplicates=True,
        approach="qcts",
        wandb_config=WANDB_CONFIG,
        search_config=SearchConfig(
            n=4,
            # search_batch_size=1,
            # search_batch_size=10,
            search_batch_size=25,
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
            # search_batch_size=1,
            # search_batch_size=10,
            search_batch_size=25,
        ),
        qcconfig=QCConfig(
            target_upgrade_rate=0.35,
        ),
    )

    experiment_configs: List[ExperimentConfig] = []

    # 1) Q2 single-knob target upgrade rate sweeps (fast to compare compute/accuracy)
    # for tur in [0.20, 0.30, 0.35, 0.45]:
    for tur in [0.45]:
        experiment_copy = copy.deepcopy(Q2_CONFIG)
        experiment_copy.qcconfig.target_upgrade_rate = tur
        experiment_copy.qcconfig.use_dynamic_thresholds = True
        experiment_copy.qcconfig.enable_margin_stability = False  # single-knob mode
        experiment_copy.wandb_config.tags.add(f"q2_tur={tur}")
        experiment_configs.append(experiment_copy)

    # 2) Q2 dynamic quantiles with/without margin+stability (no target_upgrade_rate)
    dyn_variants = [
        {"high_q": 0.75, "low_q": 0.15, "enable_ms": True},
        {"high_q": 0.85, "low_q": 0.25, "enable_ms": False},
    ]
    for dv in dyn_variants:
        experiment_copy = copy.deepcopy(Q2_CONFIG)
        experiment_copy.qcconfig.target_upgrade_rate = None
        experiment_copy.qcconfig.use_dynamic_thresholds = True
        experiment_copy.qcconfig.high_q = dv["high_q"]
        experiment_copy.qcconfig.low_q = dv["low_q"]
        experiment_copy.qcconfig.enable_margin_stability = dv["enable_ms"]
        experiment_copy.wandb_config.tags.add(
            f"q2_dyn_hq={dv['high_q']}_lq={dv['low_q']}_ms={dv['enable_ms']}"
        )
        experiment_configs.append(experiment_copy)

    # 3) Q2 single-knob with lookahead enabled to test deeper PRM context
    for tur in [0.35, 0.30]:
        experiment_copy = copy.deepcopy(Q2_CONFIG)
        experiment_copy.qcconfig.target_upgrade_rate = tur
        experiment_copy.qcconfig.use_dynamic_thresholds = True
        experiment_copy.qcconfig.enable_margin_stability = False
        experiment_copy.beam_search_config.lookahead = 1
        experiment_copy.wandb_config.tags.add(f"q2_tur={tur}_la=1")
        experiment_configs.append(experiment_copy)

    # 4) Q2 single-knob with gentler pruning threshold
    experiment_copy = copy.deepcopy(Q2_CONFIG)
    experiment_copy.qcconfig.target_upgrade_rate = 0.35
    experiment_copy.qcconfig.use_dynamic_thresholds = True
    experiment_copy.qcconfig.enable_margin_stability = False
    experiment_copy.qcconfig.low_threshold = 0.10
    experiment_copy.wandb_config.tags.add("q2_tur=0.35_low=0.10")
    experiment_configs.append(experiment_copy)

    # 5) Baseline qcts variant with alternative thresholds for comparison
    for high_t, low_t in [(0.85, 0.35)]:
        experiment_copy = copy.deepcopy(QC_CONFIG)
        experiment_copy.qcconfig.high_threshold = high_t
        experiment_copy.qcconfig.low_threshold = low_t
        experiment_copy.wandb_config.tags.add(f"qcts_h={high_t}_l={low_t}")
        experiment_configs.append(experiment_copy)

    run(BASE_CONFIG, experiment_configs)
