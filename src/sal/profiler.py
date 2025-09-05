import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import pynvml
import torch
from git import Optional
from wandb.wandb_run import Run

from .config import ProfilerConfig


class NoOpProfiler(nullcontext):
    def step(self):
        pass

    def export_chrome_trace(self):
        pass


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class MemoryMetrics:
    """Type-safe container for memory profiling results."""

    baseline_gb: float = 0.0
    llm_memory_gb: float = 0.0
    draft_llm_memory_gb: Optional[float] = None
    prm_memory_gb: float = 0.0
    inference_overhead_gb: float = 0.0

    def to_wandb_dict(self) -> dict[str, float]:
        """Convert to wandb logging format."""
        log_dict = {
            "memory/baseline_gb": self.baseline_gb,
            "memory/llm_gb": self.llm_memory_gb,
            "memory/prm_gb": self.prm_memory_gb,
            "memory/inference_overhead_gb": self.inference_overhead_gb,
        }
        if self.draft_llm_memory_gb is not None:
            log_dict["memory/draft_llm_gb"] = self.draft_llm_memory_gb
        return log_dict

    def get_total_model_memory(self) -> float:
        """Calculate total memory used by all models."""
        total = self.llm_memory_gb + self.prm_memory_gb
        if self.draft_llm_memory_gb is not None:
            total += self.draft_llm_memory_gb
        return total


class Profiler:
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.memory_metrics = MemoryMetrics()
        self.inference_runtime: float = 0.0

    def set_output_dir(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def start_memory_profiling(self):
        if self.config.profile_memory:
            torch.cuda.memory._record_memory_history(
                max_entries=self.config.memory_max_entries
            )

    def finish_memory_profiling(self, run: Run):
        if self.config.profile_memory:
            torch.cuda.memory._dump_snapshot(str(self._get_memory_snapshot_path()))
            torch.cuda.memory._record_memory_history(enabled=None)

        run.log(self.memory_metrics.to_wandb_dict())
        run.log({"runtime/inference_runtime": self.inference_runtime})

        memory_info = [
            f"LLM: {self.memory_metrics.llm_memory_gb:.2f}GB",
            f"PRM: {self.memory_metrics.prm_memory_gb:.2f}GB",
        ]

        if self.memory_metrics.draft_llm_memory_gb is not None:
            memory_info.append(
                f"Draft LLM: {self.memory_metrics.draft_llm_memory_gb:.2f}GB"
            )

        memory_info.append(
            f"Inference: {self.memory_metrics.inference_overhead_gb:.2f}GB"
        )

        logger.info(f"Memory - {', '.join(memory_info)}")
        logger.info(f"Inference Runtime: {self.inference_runtime}")

    def get_pytorch_profiler(self):
        if self.config.profile_operations:
            return torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(self._get_trace_dir())
                ),
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

    def _get_memory_snapshot_path(self) -> Path:
        return self.output_dir / self.config.memory_snapshot_file

    def _get_trace_dir(self) -> Path:
        return self.output_dir / self.config.operations_trace_dir
