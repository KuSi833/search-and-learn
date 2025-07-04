from sal.test_time_compute import main
from sal.config import Config, SearchConfig, DatasetConfig

base_config = Config(
    model_path="/vol/bitbucket/km1124/search-and-learn/models/qwen2.5-math-7b-gguf-q4/qwen2.5-math-7b-q4_k_s.gguf",
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
