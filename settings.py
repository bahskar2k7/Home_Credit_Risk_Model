import typing as t
import datetime as dt
from pathlib import Path

PROJECT_ROOT: Path = Path()


def get_timestamp_str() -> str:
    """
    Return timestamp as string in format YYYYMMDD_HHMM
    :return:
    """
    return dt.datetime.now().strftime('%Y%m%d_%H%M%S')


class Paths(t.NamedTuple):
    MODEL_OUT: Path = PROJECT_ROOT.joinpath("results")

    DATA_OUT: Path = PROJECT_ROOT.joinpath("output")
    DATA_OUT_TRAIN_FILE: Path = PROJECT_ROOT.joinpath("output", "train", "train_clean.parquet")
    DATA_OUT_TEST_FILE: Path = PROJECT_ROOT.joinpath("output", "test", "test_clean.parquet")


PATH = Paths()
