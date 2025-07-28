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
from pathlib import Path

from datasets import Dataset, load_dataset

from sal.config import Config, DatasetConfig

logger = logging.getLogger()


def get_dataset(config: DatasetConfig) -> Dataset:
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    if config.num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))

    return dataset


def save_inference_output(dataset, result_file_path):
    dataset.to_json(result_file_path, lines=True)
    logger.info(f"Saved completions to {result_file_path}")
