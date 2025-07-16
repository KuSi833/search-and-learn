#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpgpuB
#SBATCH --job-name=qtts
#SBATCH --output=./logs/slurm-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1  # Request 1 GPU

source /vol/cuda/12.0.0/setup.sh
#AMD7-A100-T

source .venv/bin/activate

export UV_PYTHON_INSTALL_DIR="/vol/bitbucket/km1124/.cache/python"
export HF_HOME="/vol/bitbucket/km1124/.cache/huggingface"
export UV_CACHE_DIR="/vol/bitbucket/km1124/.cache/uv"

# Set VLLM profiling and logging configuration
export VLLM_TORCH_PROFILER_DIR="./trace/"
export VLLM_LOGGING_LEVEL="DEBUG"

python experiments/qwen_math.py

