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

import logging
from typing import Dict, Literal, Optional

from datasets import Dataset, load_dataset

from sal.config import DatasetConfig

logger = logging.getLogger()


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


# ---------- Convenience loaders ----------

# Known subsets for MATH-500. Extend as needed.
Math500Subset = Literal["hard"]
_MATH500_SUBSETS: Dict[Math500Subset, set[int]] = {
    "hard": {40, 62},
}


def get_math500(subset: Optional[Math500Subset] = None) -> Dataset:
    """Return the MATH-500 test split optionally filtered to a named subset.

    Parameters
    - subset: an optional subset name (e.g., "hard"). When provided and known,
      the dataset will be indexed accordingly.
    """
    cfg = DatasetConfig(
        dataset_name="HuggingFaceH4/MATH-500",
        dataset_split="test",
    )

    if subset is not None:
        indices = _MATH500_SUBSETS[subset]
        cfg.dataset_indicies = set(indices)

    return get_dataset(cfg)
