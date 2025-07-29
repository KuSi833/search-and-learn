from contextlib import nullcontext
from pathlib import Path

import pynvml
import torch

from .config import ProfilerConfig


class NoOpProfiler(nullcontext):
    def step(self):
        pass

    def export_chrome_trace(self):
        pass


def get_profiler(enabled):
    if enabled:
        return torch.profiler.profile(
            activities=[
                # torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )
    return NoOpProfiler()


class Profiler:
    def __init__(self, config: ProfilerConfig, output_dir: Path):
        self.config = config
        self.memory_snapshot_path = output_dir / "memory_snapshot.pickle"

    def start_profiling(self):
        if self.config.profile_memory:
            torch.cuda.memory._record_memory_history(max_entries=10000)

    def finish_profiling(self):
        if self.config.profile_memory:
            torch.cuda.memory._dump_snapshot(str(self.memory_snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)

    @staticmethod
    def get_gpu_memory_gb():
        return torch.cuda.memory_allocated() / 1e9

    @staticmethod
    def get_total_gpu_memory_gb():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (info.total - info.free) / 1e9

    @staticmethod
    def get_peak_gpu_memory_gb():
        return torch.cuda.max_memory_allocated() / 1e9
