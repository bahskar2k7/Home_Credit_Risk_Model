import typing as t
from pathlib import Path

import pandas as pd

import utils

logger = utils.logger
TRAIN_PATH = Path().absolute().parent.joinpath("data", "output", "train", "train_clean.parquet")


def load_data(filepath: t.Optional[Path] = None):
    if not filepath:
        filepath = TRAIN_PATH
    df = pd.read_parquet(str(filepath))
    logger.info("Loaded train data. Shape: %s", df.shape)
    return df
