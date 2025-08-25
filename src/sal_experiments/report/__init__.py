from datetime import timedelta

from .util import format_runtime, runtime_stats


def main():
    times = [
        timedelta(minutes=3, seconds=4),
        timedelta(minutes=3, seconds=3),
        timedelta(minutes=3, seconds=12),
    ]
    mean, std = runtime_stats(times)
    print(format_runtime(mean, std))  # "1.3 ± 0.2 s"
