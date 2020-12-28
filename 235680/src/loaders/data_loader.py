import pandas as pd
import os

from src.loaders.path_loader import get_train_data_path, get_test_data_path, get_data_path
from src.preprocessors.add_columns import add_sin_cos
from src.preprocessors.preprocessors import split_train_valid_test, apply_standard_scale
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
    df = add_sin_cos(df, "Hour")

    train_df, valid_df, test_df = split_train_valid_test(df, TRAIN_VALID_TEST_RATIO)

    return apply_standard_scale(train_df, valid_df, test_df)


def load_basic_preprocessed_predict():
    df = load_train_data()
    df = add_sin_cos(df, "Hour")

    submission_df = load_test_data()
    submission_df = add_sin_cos(submission_df, "Hour")

    train_df, _, _ = split_train_valid_test(df, TRAIN_VALID_TEST_RATIO)
    _, submission_df, _ = apply_standard_scale(train_df, submission_df, submission_df)
    return train_df, submission_df
