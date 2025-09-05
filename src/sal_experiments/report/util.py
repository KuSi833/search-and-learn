from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np


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


def format_mean_std(values: List[float], precision: int = 1) -> str:
    """Format a list of values as 'mean ± std'."""
    if len(values) == 1:
        return f"{values[0]:.{precision}f}"

    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0
    return f"{mean:.{precision}f} ± {std:.{precision}f}"
