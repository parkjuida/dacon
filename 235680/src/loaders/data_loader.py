import os
from typing import Union

import numpy as np
import pandas as pd

from src.loaders.path_loader import get_train_data_path, get_test_data_path, get_data_path
from src.preprocessors.add_columns import add_ghi, add_sin_cos_day, add_sin_cos_hour
from src.preprocessors.preprocessors import split_train_valid_test, apply_standard_scale, do_common_preprocess
from src.settings import TRAIN_VALID_TEST_RATIO


def load_train_data() -> pd.DataFrame:
    return pd.read_csv(f"{get_train_data_path()}{os.sep}train.csv")


def load_test_data() -> pd.DataFrame:
    test_dataset = pd.DataFrame()

    for i in range(0, 81):
        test_dataset = pd.concat((test_dataset, pd.read_csv(f"{get_test_data_path()}{os.sep}{i}.csv")))

    return test_dataset


def load_submission_data() -> pd.DataFrame:
    return pd.read_csv(f"{get_data_path()}{os.sep}sample_submission.csv")


def load_basic_preprocessed_train():
    df = load_train_data()
    df = do_common_preprocess(df)

    train_df, valid_df, test_df = split_train_valid_test(df, TRAIN_VALID_TEST_RATIO)

    return apply_standard_scale(train_df, valid_df, test_df)


def load_basic_preprocessed_predict():
    df = load_train_data()
    df = do_common_preprocess(df)

    submission_df = load_test_data()
    submission_df = do_common_preprocess(submission_df)

    train_df, _, _ = split_train_valid_test(df, TRAIN_VALID_TEST_RATIO)
    _, submission_df, _ = apply_standard_scale(train_df, submission_df, submission_df)
    return train_df, submission_df


def load_test_features(df: Union[pd.DataFrame, np.array], input_steps):
    if type(df) == pd.DataFrame:
        df = df.values
    _one_week = 48 * 7
    filter_indices = [slice(i * _one_week - input_steps, i * _one_week) for i in range(1, 82)]

    return np.array([df[s] for s in filter_indices])