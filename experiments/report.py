from datetime import timedelta
from typing import List, Tuple

import numpy as np

from .fusion.fusion_v2 import hyperparameter_scaling


def runtime_stats(times: List[timedelta]) -> Tuple[float, float]:
    """Returns (mean, std) in seconds"""
    times_seconds = [t.total_seconds() for t in times]
    mean = float(np.mean(times_seconds))
    std = float(np.std(times_seconds, ddof=1))
    return mean, std


def format_runtime(mean: float, std: float, precision: int = 1) -> str:
    """Format runtime with appropriate units"""
    if mean < 1:
        return f"{mean * 1000:.{precision}f} ± {std * 1000:.{precision}f} ms"
    elif mean < 60:
        return f"{mean:.{precision}f} ± {std:.{precision}f} s"
    else:
        return f"{mean / 60:.{precision}f} ± {std / 60:.{precision}f} min"


# Or using timedelta for readability:
def format_runtime_td(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    return str(td)


if __name__ == "__main__":
    # Data used for report experiments

    times = [
        timedelta(minutes=3, seconds=4),
        timedelta(minutes=3, seconds=3),
        timedelta(minutes=3, seconds=12),
    ]
    mean, std = runtime_stats(times)
    print(format_runtime(mean, std))  # "1.3 ± 0.2 s"
    # hyperparameter_scaling()
