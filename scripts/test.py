from vllm import LLM

print("Starting to load model...")
model_path = "/vol/bitbucket/km1124/models/Qwen2.5-Math-7B-Instruct"
# model_path = "/data/km1124/models/Qwen2.5-Math-7B-Instruct"
llm = LLM(model=model_path)
print("Model loaded successfully!")
