from pathlib import Path


def get_model_base_path() -> Path:
    MODEL_BASE_PATH_LOCAL = Path("/data/km1124/models")
    MODEL_BASE_PATH_GIT_BUCKET = Path("/vol/bitbucket/km1124/search-and-learn/models")

    if MODEL_BASE_PATH_LOCAL.exists():
        print(f"Found local models folder under {MODEL_BASE_PATH_LOCAL}")
        model_base_path = MODEL_BASE_PATH_LOCAL
    else:
        print(f"Couldn't find local models storage, using bitbucket instead.")
        model_base_path = MODEL_BASE_PATH_GIT_BUCKET

    return model_base_path
