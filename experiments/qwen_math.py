from sal.test_time_compute import main
from sal.config import Config, SearchConfig, DatasetConfig

QWEN_MODELS_PATH = "/vol/bitbucket/km1124/search-and-learn/models/quant_factory"
QWEN_Q8_PATH = QWEN_MODELS_PATH + "Qwen2.5-Math-7B.Q8_0.gguf"

base_config = Config(
    model_path=QWEN_Q8_PATH,
    filter_duplicates=True,
    approach="beam_search",
    search_config=SearchConfig(
        n=4,
        search_batch_size=1,
        seed=0,
    ),
    dataset_config=DatasetConfig(
        num_samples=100,
    ),
)


if __name__ == "__main__":
    main(base_config)
