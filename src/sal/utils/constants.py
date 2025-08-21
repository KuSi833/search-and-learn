from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class Benchmark:
    key: str
    hf_name: str
    split: str


class Benchmarks(Enum):
    """Namespace for benchmark constants."""

    MATH500 = Benchmark("math500", "HuggingFaceH4/MATH-500", "test")
    AIME24 = Benchmark("aime24", "HuggingFaceH4/aime_2024", "train")

    @classmethod
    def from_key(cls, key: str) -> Benchmark:
        """Return the benchmark instance by its key."""
        for benchmark_enum in cls:
            if benchmark_enum.value.key == key:
                return benchmark_enum.value
        raise ValueError(f"Unknown benchmark key: {key}")


# Default relative paths (can be overridden when project root is known)
BENCHMARK_MAPPINGS_ROOT = Path("./data/benchmark_mappings")
BENCHMARK_SUBSETS_ROOT = Path("./data/benchmark_subsets")
