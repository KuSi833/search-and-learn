from sal.test_time_compute import main
from sal.config import Config, SearchConfig, DatasetConfig

QWEN_MODELS_PATH = "/vol/bitbucket/km1124/search-and-learn/models/quant_factory/"
QWEN_Q8_PATH = QWEN_MODELS_PATH + "Qwen2.5-Math-7B.Q8_0.gguf"

beam_search_config = Config(
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

best_of_n_config = Config(
    model_path="/vol/bitbucket/km1124/search-and-learn/models/qwen2.5-math-7b-gguf/qwen2.5-math-7b-q8_0.gguf",
    filter_duplicates=True,
    sort_completed=True,
    approach="best_of_n",
    search_config=SearchConfig(
        n=4,
        search_batch_size=25,
        seed=0,
    ),
    dataset_config=DatasetConfig(
        num_samples=100,
    ),
)

if __name__ == "__main__":
    main(best_of_n_config)
