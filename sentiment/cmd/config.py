import os
from pathlib import Path


_DATA_DIR_ENV = "DATA_DIR"
_AIRLINE_COMMENTS_FILENAME = "tweets.csv"


def airline_comments_path() -> Path:
    data_dir = Path(os.environ[_DATA_DIR_ENV])
    return Path(data_dir, _AIRLINE_COMMENTS_FILENAME)
