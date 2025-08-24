import copy
from typing import List

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
from sal.test_time_compute import get_project_root, run
from sal.utils.constants import Benchmarks
from sal.utils.experiment import get_model_base_path
from sal.utils.runs import fusion_base_runs_best

if __name__ == "__main__":
    load_dotenv()

    project_root = get_project_root()
    model_base_path = get_model_base_path()

    INSTRUCT_MODEL = GeneratorConfig(
        base_path=model_base_path,
        name="Qwen/Qwen2.5-Math-7B-Instruct",
        parameter_count="7B",
    )
    BASE_MODEL = GeneratorConfig(
        name="Qwen/Qwen2.5-Math-7B",
        parameter_count="7B",
    )

    PRM_CONFIG = PRMConfig(
        base_path=model_base_path,
        name="Qwen/Qwen2.5-Math-PRM-7B",
    )

    WANDB_CONFIG = WandbConfig(tags=set(["fusion-rerun"]))

    BEST_OF_N_CONFIG = ExperimentConfig(
        filter_duplicates=True,
        sort_completed=True,
        approach="best_of_n",
        search_config=SearchConfig(
            n=8,
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

    COVERAGE = 10

    for subset_run_id in fusion_base_runs_best():
        dataset_config = DatasetConfig.from_subset_file(
            run_id=subset_run_id,
            coverage=COVERAGE,
            benchmark=Benchmarks.MATH500.value,
            project_root=project_root,
        )
        base_config = BaseConfig(
            prm_config=PRM_CONFIG,
            generator_config=INSTRUCT_MODEL,
            dataset_config=dataset_config,
        )
        base_config_var = copy.deepcopy(base_config)
        base_config_var.dataset_config = dataset_config

        experiment_configs.append(BEST_OF_N_CONFIG)

        run(base_config, experiment_configs)
