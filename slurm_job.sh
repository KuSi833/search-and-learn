#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=AMD7-A100-T
#SBATCH --job-name=qtts
#SBATCH --output=./logs/slurm-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# Exit on any error
set -e

echo "Launching SLURM jobscript"
source /vol/cuda/12.0.0/setup.sh
source .venv/bin/activate

echo "Fetching commit with hash $COMMIT_HASH"
git reset --hard HEAD

git remote set-url origin https://$GITHUB_TOKEN@github.com/KuSi833/search-and-learn.git
# Fetch with error handling
echo "Fetching latest changes..."
if ! git fetch; then
    echo "ERROR: Failed to fetch from remote repository"
    echo "This might be due to SSH key issues or network problems"
    exit 1
fi

# Checkout specific commit with error handling
echo "Checking out commit $COMMIT_HASH..."
if ! git checkout $COMMIT_HASH; then
    echo "ERROR: Failed to checkout commit $COMMIT_HASH"
    echo "Commit may not exist or may not be fetched"
    exit 1
fi

echo "Successfully checked out commit $COMMIT_HASH"

export UV_PYTHON_INSTALL_DIR="/vol/bitbucket/km1124/.cache/python"
export HF_HOME="/vol/bitbucket/km1124/.cache/huggingface"
export UV_CACHE_DIR="/vol/bitbucket/km1124/.cache/uv"

# Set VLLM profiling and logging configuration
export VLLM_TORCH_PROFILER_DIR="./trace"
# export VLLM_LOGGING_LEVEL="DEBUG"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export NCCL_P2P_DISABLE=1

echo "Running main file"
# python experiments/qwen_math.py
python experiments/expand.py
