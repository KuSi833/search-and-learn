# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging

from datasets import Dataset, load_dataset

from sal.config import DatasetConfig
from sal.utils.constants import BENCHMARK_MAPPINGS_ROOT, DATASETS

logger = logging.getLogger()


class BenchmarkMapping:
    """Simple mapping cache for benchmark datasets."""

    def __init__(self, dataset_name: str):
        config = DATASETS[dataset_name]
        self.file = BENCHMARK_MAPPINGS_ROOT / config["hf_name"] / "mapping.json"
        with open(self.file, "r") as f:
            data = json.load(f)
        self._id_to_unique = data
        self._unique_to_id = {v: k for k, v in data.items()}

    def get_unique_id(self, index: str) -> str:
        """Get unique_id from index."""
        return self._id_to_unique[index]

    def get_index(self, unique_id: str) -> str:
        """Get index from unique_id."""
        return self._unique_to_id[unique_id]


def get_dataset(config: DatasetConfig) -> Dataset:
    """Load a dataset split and apply optional slicing and indexing.

    Ensures the returned object is a `datasets.Dataset` (not a `DatasetDict` or
    streaming dataset), so downstream calls to `select` and `len` are valid.
    """
    ds = load_dataset(config.dataset_name, split=config.dataset_split)

    if not isinstance(ds, Dataset):
        raise TypeError(
            "Expected `datasets.Dataset` when loading split=...; got a different type. "
            "Ensure `dataset_split` yields a materialised dataset (non-streaming)."
        )

    # Apply explicit index selection first if provided
    if len(config.dataset_indicies) > 0:
        return ds.select(list(config.dataset_indicies))

    # Apply start/end slicing if both are provided
    if config.dataset_start is not None and config.dataset_end is not None:
        ds = ds.select(range(config.dataset_start, config.dataset_end))

    # Limit total number of samples if requested
    if config.num_samples is not None:
        ds = ds.select(range(min(len(ds), config.num_samples)))

    return ds


def save_inference_output(dataset, result_file_path):
    dataset.to_json(result_file_path, lines=True)
    logger.info(f"Saved completions to {result_file_path}")


# Note: dataset-specific curated index helpers are defined close to experiment
# code (see `sal/utils/experiment.py`). This module remains focused on generic
# dataset loading and I/O utilities only.
