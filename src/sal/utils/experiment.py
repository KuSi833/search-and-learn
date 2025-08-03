from pathlib import Path


def get_model_base_path() -> Path:
    MODEL_BASE_PATH_LOCAL = Path("/data/models")
    MODEL_BASE_PATH_GIT_BUCKET = Path("/vol/bitbucket/km1124/search-and-learn/models")

    model_base_path = (
        MODEL_BASE_PATH_LOCAL
        if MODEL_BASE_PATH_LOCAL.exists()
        else MODEL_BASE_PATH_GIT_BUCKET
    )
    return model_base_path
