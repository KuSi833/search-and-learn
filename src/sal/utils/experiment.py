from pathlib import Path
from typing import Dict, Literal, Set

# Curated MATH-500 subsets. Extend as needed.
Math500Subset = Literal["hard"]
_MATH500_SUBSETS: Dict[Math500Subset, Set[int]] = {
    "hard": {
        9,
        11,
        18,
        25,
        43,
        52,
        58,
        60,
        62,
        66,
        71,
        78,
        80,
        94,
        96,
        100,
        103,
        104,
        105,
        110,
        115,
        126,
        138,
        147,
        150,
        152,
        154,
        164,
        168,
        177,
        184,
        189,
        204,
        224,
        239,
        240,
        242,
        245,
        248,
        249,
        257,
        264,
        274,
        284,
        285,
        286,
        294,
        295,
        301,
        302,
        303,
        306,
        308,
        324,
        326,
        327,
        340,
        352,
        355,
        359,
        372,
        381,
        385,
        392,
        400,
        401,
        419,
        422,
        425,
        432,
        436,
        444,
        445,
        454,
        456,
        458,
        460,
        467,
        475,
        477,
        478,
        481,
        490,
        491,
        494,
        497,
    },
}


def get_math500_debug_indices() -> Set[int]:
    """Return a small subset of the hard MATH-500 indices for debugging.

    This is a subset of the hard indices that starts from around index 56
    where crashes have been observed, useful for reproducing and debugging issues.
    """
    # Take indices 56-65 from the hard set for debugging
    hard_indices = _MATH500_SUBSETS["hard"]
    hard_list = sorted(list(hard_indices))
    debug_indices = hard_list[56:66]  # 10 examples starting from where it crashed
    return set(debug_indices)


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
