from enum import Enum
from typing import Dict


class Benchmark(Enum):
    MATH500 = "math500"
    AIME24 = "aime24"


DATASETS: Dict[str, Dict[str, str]] = {
    Benchmark.MATH500.value: {
        "hf_name": "HuggingFaceH4/MATH-500",
        "file": "data/benchmark_mappings/math500.json",
        "split": "test",
    },
    Benchmark.AIME24.value: {
        "hf_name": "HuggingFaceH4/aime_2024",  # Replace with actual dataset name
        "file": "data/benchmark_mappings/aime24.json",
        "split": "train",
    },
}
