import torch
from contextlib import nullcontext


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
