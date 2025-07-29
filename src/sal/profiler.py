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


class Profiler:
    def __init__(self, config: ProfilerConfig, output_dir: Path):
        self.config = config
        self.memory_snapshot_path = output_dir / self.config.memory_snapshot_file

    def start_memory_profiling(self):
        if self.config.profile_memory:
            torch.cuda.memory._record_memory_history(max_entries=10000)

    def finish_memory_profiling(self):
        if self.config.profile_memory:
            torch.cuda.memory._dump_snapshot(str(self.memory_snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)

    def get_pytorch_profiler(self):
        if self.config.profile_operations:

            def trace_handler(prof):
                prof.export_chrome_trace(
                    self.config.operations_trace_file, device="cuda:0"
                )

            return torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=trace_handler,
            )
        return NoOpProfiler()

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
