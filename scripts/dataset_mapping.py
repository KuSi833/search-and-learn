#!/usr/bin/env python3
import json
import os
from typing import Dict

import click
from datasets import load_dataset

from sal.utils.constants import DATASETS

# Dataset configurations


def generate_mapping(dataset_name: str):
    """Generate and save mapping for a dataset."""
    config = DATASETS[dataset_name]

    print(f"Loading {dataset_name} dataset...")
    ds = load_dataset(config["hf_name"], split="test")
    mapping = {str(idx): row["unique_id"] for idx, row in enumerate(ds)}

    os.makedirs(os.path.dirname(config["file"]), exist_ok=True)
    with open(config["file"], "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Saved mapping to {config['file']}")


@click.command()
@click.argument("dataset", type=click.Choice(["math500", "aime24"]))
def main(dataset: str):
    """Generate and save mapping for a dataset."""
    generate_mapping(dataset)


if __name__ == "__main__":
    main()
