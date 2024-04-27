# suppress enjoying warnings from seaborn
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os

# for fiding file names
from pathlib import Path
from glob import glob
import gc

# data processing libraries
import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np

from utils.std_logger import logger
from utils.decorator import time_this

# project CONSTANTS
# ROOT = Path("/kaggle/input/home-credit-credit-risk-model-stability")
DRIVE = "I:"
USERNAME = os.environ['USERNAME']
ROOT = Path(DRIVE).joinpath(USERNAME, "data", "Home Credit - Credit Risk Model Stability")
TRAIN_DIR = ROOT / "parquet_files" / "train"
TEST_DIR = ROOT / "parquet_files" / "test"
SEED = 2024

# # current dir is data/
OUTPUT = Path(".") / "output"
OUTPUT_TRAIN = OUTPUT / "train"
OUTPUT_TEST = OUTPUT / "test"

OUTPUT.mkdir(exist_ok=True)
OUTPUT_TEST.mkdir(exist_ok=True)
OUTPUT_TRAIN.mkdir(exist_ok=True)

logger.info("Root path: %s", ROOT)
logger.info("Output path: %s", OUTPUT)


class Pipeline:
    """
    Helper class taken from notebook:
    https://www.kaggle.com/code/daviddirethucus/home-credit-risk-lightgbm
    """

    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  # !!?
                df = df.with_columns(pl.col(col).dt.total_days())  # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().sum()
                if isnull >= 0.8:  # changed to 0.8
                    df = df.drop(col)

        # for col in df.columns:
        #     if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
        #         freq = df[col].n_unique()
        #         # if (freq == 1) | (freq > 200):
        #         #     df = df.drop(col)
        #         if (freq == 1) | (freq > 200):
        #             df = df.drop(col)
        return df


class Aggregator:
    """
    Helper class taken from notebook:
    https://www.kaggle.com/code/daviddirethucus/home-credit-risk-lightgbm
    """

    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]  # max & replace col name
        return expr_max

    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)

        return exprs


def read_file(path, depth=None):
    """
    Helper function taken from notebook:
    https://www.kaggle.com/code/daviddirethucus/home-credit-risk-lightgbm
    """
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df


def read_files(regex_path, depth=None):
    """
    Helper function taken from notebook:
    https://www.kaggle.com/code/daviddirethucus/home-credit-risk-lightgbm
    """
    chunks = []

    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)

    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df


def feature_eng(df_base, depth_0, depth_1, depth_2):
    """
    Helper function taken from notebook:
    https://www.kaggle.com/code/daviddirethucus/home-credit-risk-lightgbm
    """
    df_base = (
        df_base
            .with_columns(
            month_decision=pl.col("date_decision").dt.month(),
            weekday_decision=pl.col("date_decision").dt.weekday(),
        )
    )
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base


def to_pandas(df_data, cat_cols=None):
    """
    Helper function taken from notebook:
    https://www.kaggle.com/code/daviddirethucus/home-credit-risk-lightgbm
    """
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


def reduce_mem_usage(df, df_name, verbose=True):
    """
    This function changes pandas numerical dtypes (reduces bit size if possible)
    to reduce memory usage

    :param df: pandas DataFrame
    :param df_name: str, name of DataFrame
    :param verbose: bool, if True prints out message of how much memory usage was reduced

    :return:  pandas DataFrame
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float64)
                else:
                    df[col] = df[col].astype(np.float64)
    # calculate memory after reduction
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        # reduced memory usage in percent
        diff_pst = 100 * (start_mem - end_mem) / start_mem
        msg = f'{df_name} mem. usage decreased to {end_mem:5.2f} Mb ({diff_pst:.1f}% reduction)'
        print(msg)
    return df


# def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
#     numeric_columns = [x for x in df.columns if 'int' in df[x].dtype.name or 'float' in df[x].dtype.name]
#     str_columns = [x for x in df.columns if 'category' in df[x].dtype.name or 'string' in df[x].dtype.name]
#     object_columns = [x for x in df.columns if 'object' in df[x].dtype.name]
#     bool_columns = [x for x in df.columns if 'bool' in df[x].dtype.name]
#     assert len(str_columns) + len(numeric_columns) + len(object_columns) + len(bool_columns) == df.shape[1]
#
#     df[numeric_columns].fillna(0.0, inplace=True)
#     df[str_columns] = df[str_columns].astype(str).fillna("-1").astype('category')
#     df[bool_columns].fillna(False, inplace=True)
#     return df


def fill_missing_values(df: pl.DataFrame):
    # https://docs.pola.rs/user-guide/expressions/column-selections/#by-dtype
    df = df.with_columns(cs.string().fill_null("-1"))
    df = df.with_columns(cs.categorical().fill_null("-1"))
    df = df.with_columns(cs.integer().fill_null(0))
    df = df.with_columns(cs.float().fill_null(0.0))
    df = df.with_columns(cs.boolean().fill_null(False))
    return df


@time_this
def build_data():
    data_train = {
        "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
        "depth_0": [
            read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
            read_files(TRAIN_DIR / "train_static_0_*.parquet"),
        ],
        "depth_1": [
            read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
            read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
            read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
            read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
            read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
            read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
            read_file(TRAIN_DIR / "train_other_1.parquet", 1),
            read_file(TRAIN_DIR / "train_person_1.parquet", 1),
            read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
            read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
            read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),
        ]
    }

    data_test = {
        "df_base": read_file(TEST_DIR / "test_base.parquet"),
        "depth_0": [
            read_file(TEST_DIR / "test_static_cb_0.parquet"),
            read_files(TEST_DIR / "test_static_0_*.parquet"),
        ],
        "depth_1": [
            read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
            read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
            read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
            read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
            read_files(TEST_DIR / "test_credit_bureau_a_1_*.parquet", 1),
            read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
            read_file(TEST_DIR / "test_other_1.parquet", 1),
            read_file(TEST_DIR / "test_person_1.parquet", 1),
            read_file(TEST_DIR / "test_deposit_1.parquet", 1),
            read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
            read_files(TEST_DIR / "test_credit_bureau_a_2_*.parquet", 2),
        ]
    }

    # get column names of original/raw features
    FEATS_ORIG = []

    # get column names
    for _key in data_train.keys():
        if isinstance(data_train[_key], list):
            for _df in data_train[_key]:
                FEATS_ORIG += _df.columns

    # leave only unique values
    FEATS_ORIG = list(set(FEATS_ORIG))
    # drop case_id column
    FEATS_ORIG.remove("case_id")

    df_train = feature_eng(**data_train)
    print("train data shape:\t", df_train.shape)
    # clean memory
    del data_train
    gc.collect()

    df_test = feature_eng(**data_test)
    print("test data shape:\t", df_test.shape)
    # clean memory
    del data_test
    gc.collect()

    # get column names of new created features
    FEATS_NEW = set(df_train.columns)
    # remove speific columns
    FEATS_NEW = FEATS_NEW.difference({'case_id', 'WEEK_NUM', 'target'})
    FEATS_NEW = FEATS_NEW.difference(set(FEATS_ORIG))
    FEATS_NEW = list(FEATS_NEW)

    print(f"Before feature engineering: {len(FEATS_ORIG)} featues.")
    print(f"{len(FEATS_NEW)} new features created.")

    # Drop the insignificant features
    df_train = df_train.pipe(Pipeline.filter_cols)
    df_test = df_test.select([col for col in df_train.columns if col != "target"])

    print("train data shape: ", df_train.shape)
    print("test data shape: ", df_test.shape)
    gc.collect()

    # get column names of the remaining features
    FEATS_REMAIN = set(df_train.columns)
    # remove speific columns
    FEATS_REMAIN = FEATS_REMAIN.difference({'case_id', 'WEEK_NUM', 'target'})
    # remaining in original features
    _1 = [_ for _ in FEATS_REMAIN if _ in FEATS_ORIG]
    # remaining of the new features
    _2 = [_ for _ in FEATS_REMAIN if _ in FEATS_NEW]

    print("After removing insignificant features:")
    print(f"{len(_1)} original features left")
    print(f"{len(_2)} new features left")
    gc.collect()

    # fill missing values using polars
    df_train = fill_missing_values(df_train)
    df_test = fill_missing_values(df_test)

    # convert back to pandas
    df_train, cat_cols = to_pandas(df_train)
    df_test, cat_cols = to_pandas(df_test, cat_cols)
    gc.collect()

    # reduce memory usage if available
    df_train = reduce_mem_usage(df_train, "df_train")
    df_test = reduce_mem_usage(df_test, "df_test")
    gc.collect()

    logger.info(f"DF train shape: {df_train.shape}")
    logger.info(f"DF test shape: {df_test.shape}")

    logger.info("Write train to: %s", OUTPUT_TRAIN)
    df_train.to_parquet(str(OUTPUT_TRAIN.joinpath("train_clean.parquet")), index=False)

    logger.info("Write test to: %s", OUTPUT_TEST)
    df_test.to_parquet(str(OUTPUT_TEST.joinpath("test_clean.parquet")), index=False)


if __name__ == "__main__":
    build_data()
