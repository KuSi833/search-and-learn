#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=AMD7-A100-T
#SBATCH --job-name=inference-scaling
#SBATCH --output=./logs/slurm-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1  # Request 1 GPU

source /vol/cuda/12.0.0/setup.sh

source .venv/bin/activate

export UV_PYTHON_INSTALL_DIR="/vol/bitbucket/km1124/.cache/python"
export HF_HOME="/vol/bitbucket/km1124/.cache/huggingface"
export UV_CACHE_DIR="/vol/bitbucket/km1124/.cache/uv"

# export CONFIG="recipes/Llama-3.2-1B-Instruct/best_of_n.yaml"
# export CONFIG="recipes/Llama-3.2-1B-Instruct/beam_search.yaml"
export CONFIG="recipes/qwen-math-16/beam_search.yaml"

python scripts/test_time_compute.py $CONFIG

