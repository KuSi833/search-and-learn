from pathlib import Path
from typing import Dict, Literal, Set

# Curated MATH-500 subsets. Extend as needed.
Math500Subset = Literal["hard"]
_MATH500_SUBSETS: Dict[Math500Subset, Set[int]] = {
    "hard": {40, 62},
}


def get_math500_indices(subset: Math500Subset) -> Set[int]:
    """Return a stable, sorted list of indices for a named MATH-500 subset.

    Use this in experiments to populate `DatasetConfig.dataset_indicies`.
    """
    return _MATH500_SUBSETS[subset]


def get_model_base_path() -> Path:
    MODEL_BASE_PATH_LOCAL = Path("/data/km1124/search-and-learn/models")
    MODEL_BASE_PATH_GIT_BUCKET = Path("/vol/bitbucket/km1124/search-and-learn/models")

    if MODEL_BASE_PATH_LOCAL.exists():
        print(f"Found local models folder under {MODEL_BASE_PATH_LOCAL}")
        model_base_path = MODEL_BASE_PATH_LOCAL
    else:
        print("Couldn't find local models storage, using bitbucket instead.")
        model_base_path = MODEL_BASE_PATH_GIT_BUCKET

    return model_base_path
