from pathlib import Path

from vllm import LLM

from sal.config import PRMConfig
from sal.models.reward_models import load_prm

model_base_path = Path("/data/km1124/search-and-learn/models")

PRM_CONFIG = PRMConfig(
    base_path=model_base_path,
    name="Qwen/Qwen2.5-Math-PRM-7B",
)

print("Starting to load model...", flush=True)
prm = load_prm(PRM_CONFIG)
print("Model loaded successfully!")
