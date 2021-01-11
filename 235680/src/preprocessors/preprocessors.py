from typing import List, Tuple

import pandas as pd
import numpy as np

from src.preprocessors.add_columns import add_sin_cos_day, add_sin_cos_hour, add_ghi, add_min_max_scaled


def split_train_valid_test(
        df: pd.DataFrame, ratio: List[float]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if type(ratio) != List:
        print("ratio should be 3 length list, example: [0.6, 0.3, 0.1]")
    if len(ratio) != 3:
        print("ratio length should be 3, example: [0.6, 0.3, 0.1]")
    if sum(ratio) != 1:
        print("sum of ration should be 1, example: [0.7, 0.2, 0.1]")

    n = len(df)
    train_valid_boundary = int(n * ratio[0])
    valid_test_boundary = int(n * (ratio[0] + ratio[1]))

    train_slice = slice(0, train_valid_boundary)
    valid_slice = slice(train_valid_boundary, valid_test_boundary)
    test_slice = slice(valid_test_boundary, n)

    train_df = df[train_slice]
    valid_df = df[valid_slice]
    test_df = df[test_slice]

    print(f"shape of train, valid, test: {train_df.shape}, {valid_df.shape}, {test_df.shape}")

    return train_df, valid_df, test_df


def apply_standard_scale(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _mean = train_df.mean()
    _std = train_df.std()

    scaled_train_df = (train_df - _mean) / _std
    scaled_valid_df = (valid_df - _mean) / _std
    scaled_test_df = (test_df - _mean) / _std

    return scaled_train_df, scaled_valid_df, scaled_test_df


def apply_minmax_scale(
 train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _min = train_df.min()
    _max = train_df.max()

    scaled_train_df = (train_df - _min) / (_max - _min)
    scaled_valid_df = (valid_df - _min) / (_max - _min)
    scaled_test_df = (test_df - _min) / (_max - _min)

    return scaled_train_df, scaled_valid_df, scaled_test_df


def inverse_standard_scale(
        train_df: pd.DataFrame,
        scaled_pred_y: np.array
):
    _mean = train_df.mean()
    _std = train_df.std()

    return scaled_pred_y * _std + _mean


def do_common_preprocess(
        df
):
    df = add_sin_cos_day(df)
    df = add_sin_cos_hour(df)
    df = add_ghi(df)

    df["TARGET_ROLLING_MEAN_3_shift_1"] = df["TARGET"].rolling(3).mean().shift(-1).fillna(0)
    df["TARGET_ROLLING_MEAN_5_shift_2"] = df["TARGET"].rolling(5).mean().shift(-2).fillna(0)
    df["TARGET_ROLLING_MEAN_11_shift_5"] = df["TARGET"].rolling(11).mean().shift(-5).fillna(0)
    df["TARGET_ROLLING_MEAN_23_shift_11"] = df["TARGET"].rolling(23).mean().shift(-11).fillna(0)
    df["TARGET_ROLLING_MEAN_47_shift_23"] = df["TARGET"].rolling(47).mean().shift(-23).fillna(0)
    # df["GHI_ANGLE_COS"] = df["GHI"] * df["Hour_cos"]
    # df["GHI_ANGLE_SIN"] = df["GHI"] * df["Hour_sin"]
    #
    # df = add_min_max_scaled(df, "DHI")
    # df = add_min_max_scaled(df, "DNI")
    # df = add_min_max_scaled(df, "GHI")
    # df = add_min_max_scaled(df, "WS")
    # df = add_min_max_scaled(df, "RH")
    # df = add_min_max_scaled(df, "T")
    #
    # df = add_min_max_scaled(df, "TARGET_ROLLING_MEAN_3_shift_1")
    # df = add_min_max_scaled(df, "TARGET_ROLLING_MEAN_5_shift_2")
    # df = add_min_max_scaled(df, "TARGET_ROLLING_MEAN_11_shift_5")
    # df = add_min_max_scaled(df, "TARGET_ROLLING_MEAN_23_shift_11")
    # df = add_min_max_scaled(df, "TARGET_ROLLING_MEAN_47_shift_23")
    # df = add_min_max_scaled(df, "GHI_ANGLE_COS")
    # df = add_min_max_scaled(df, "GHI_ANGLE_SIN")

    df.drop(["Day", "Hour", "Minute"], axis=1, inplace=True)

    return df
