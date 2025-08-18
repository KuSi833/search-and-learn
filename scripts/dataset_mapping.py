#!/usr/bin/env python3
import json

import click
from datasets import load_dataset
from rich.console import Console

from sal.utils.constants import BENCHMARK_MAPPINGS_ROOT, DATASETS

console = Console()


def generate_mapping(dataset_name: str):
    """Generate and save mapping for a dataset."""
    config = DATASETS[dataset_name]

    print(f"Loading {dataset_name} dataset...")
    ds = load_dataset(config["hf_name"], split=config["split"])

    match dataset_name:
        case "math500":
            unique_id_key = "unique_id"
        case "aime24":
            unique_id_key = "id"

    mapping = {str(idx): row[unique_id_key] for idx, row in enumerate(ds)}

    output_file = BENCHMARK_MAPPINGS_ROOT / config["hf_name"] / "mapping.json"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w") as f:
        json.dump(mapping, f, indent=2)

    console.print(f"Saved mapping to [yellow]{output_file}")


@click.command()
@click.argument("dataset", type=click.Choice(["math500", "aime24"]))
def main(dataset: str):
    """Generate and save mapping for a dataset."""
    generate_mapping(dataset)


if __name__ == "__main__":
    main()
