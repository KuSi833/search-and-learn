from enum import Enum
from pathlib import Path
from typing import Dict


class Benchmark(Enum):
    MATH500 = "math500"
    AIME24 = "aime24"


DATASETS: Dict[str, Dict[str, str]] = {
    Benchmark.MATH500.value: {
        "hf_name": "HuggingFaceH4/MATH-500",
        "split": "test",
    },
    Benchmark.AIME24.value: {
        "hf_name": "HuggingFaceH4/aime_2024",  # Replace with actual dataset name
        "split": "train",
    },
}

BENCHMARK_MAPPINGS_ROOT = Path("./data/benchmark_mappings")
BENCHMARK_SUBSETS_ROOT = Path("./data/benchmark_mappings")
