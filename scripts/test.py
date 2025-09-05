from pathlib import Path

from vllm import LLM

from sal.config import GeneratorConfig, PRMConfig
from sal.models.reward_models import load_prm
from sal.utils.logging import setup_logging

setup_logging()

model_base_path = Path("/data/km1124/search-and-learn/models")

# PRM_CONFIG = PRMConfig(
#     base_path=model_base_path,
#     name="Qwen/Qwen2.5-Math-PRM-7B",
# )
# prm = load_prm(PRM_CONFIG)

SMALLEST_MODEL = GeneratorConfig(
    base_path=model_base_path,
    name="Qwen/Qwen2-Math-1.5B-Instruct",
    parameter_count="1.5B",
)
llm = LLM(model=SMALLEST_MODEL.name)

print("Starting to load model...", flush=True)
print("Model loaded successfully!")
