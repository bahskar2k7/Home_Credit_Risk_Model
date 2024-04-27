import typing as t

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 2024


def split_data(df: pd.DataFrame, train_size: t.Optional[float] = 0.7) -> t.Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_ids_train, case_ids_test = train_test_split(df['case_id'], train_size=train_size, random_state=SEED)

    x_train = df.loc[df['case_id'].isin(case_ids_train), :]
    x_train = x_train.drop("target", axis=1)

    x_test = df.loc[df['case_id'].isin(case_ids_test), :]
    x_test = x_test.drop("target", axis=1)

    y_train = df.loc[df['case_id'].isin(case_ids_train), 'target']
    y_test = df.loc[df['case_id'].isin(case_ids_test), 'target']

    return x_train, y_train, x_test, y_test
