from vllm import LLM

print("Starting to load model...", flush=True)
model_path = "/vol/bitbucket/km1124/search-and-learn/models/Qwen2.5-Math-7B-Instruct/"
# model_path = "/vol/bitbucket/km1124/search-and-learn/models/Qwen2.5-Math-1.5B-Instruct/"
# model_path = "/data/km1124/models/Qwen2.5-Math-7B-Instruct"
llm = LLM(model=model_path)
print("Model loaded successfully!")
