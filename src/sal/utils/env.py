import os


def get_dotenv_or_throw(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise RuntimeError(f"Environment variable '{key}' is not set")
    return value
