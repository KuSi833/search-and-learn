#!/usr/bin/env python3
import json

import click
from datasets import load_dataset
from rich.console import Console

from sal.utils.constants import BENCHMARK_MAPPINGS_ROOT, Benchmark, Benchmarks

console = Console()


def generate_mapping(benchmark: Benchmark):
    """Generate and save mapping for a dataset."""
    print(f"Loading {benchmark.key} dataset...")
    ds = load_dataset(benchmark.hf_name, split=benchmark.split)

    match benchmark:
        case Benchmarks.MATH500.value:
            unique_id_key = "unique_id"
        case Benchmarks.AIME24.value:
            unique_id_key = "id"

    mapping = {str(idx): row[unique_id_key] for idx, row in enumerate(ds)}

    output_file = BENCHMARK_MAPPINGS_ROOT / benchmark.hf_name / "mapping.json"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=2)

    console.print(f"Saved mapping to [yellow]{output_file}")


@click.command()
@click.argument(
    "benchmark",
    dest="benchmark_key",
    type=click.Choice([b.value.key for b in Benchmarks]),
)
def main(benchmark_key: str):
    """Generate and save mapping for a dataset."""
    benchmark = Benchmarks.from_key(benchmark_key)
    generate_mapping(benchmark)


if __name__ == "__main__":
    main()
