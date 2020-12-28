from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf

from src.loaders.data_loader import load_basic_preprocessed_predict
from src.preprocessors.preprocessors import inverse_standard_scale


def load_test_features(df: Union[pd.DataFrame, np.array], input_steps):
    if type(df) == pd.DataFrame:
        df = df.values
    _one_week = 48 * 7
    filter_indices = [slice(i * _one_week - input_steps, i * _one_week) for i in range(1, 82)]

    return np.array([df[s] for s in filter_indices])


def make_submission_file(
        model: tf.keras.Model,
        submission_df: pd.DataFrame,
        input_steps: int,
        tau: float
):
    train_df, predict_df = load_basic_preprocessed_predict()
    # train_df: not scaled, predict_df: scaled
    predict_df = load_test_features(predict_df, input_steps)
    answer = model.predict(predict_df)
    answer_df = pd.DataFrame(answer.reshape(-1, predict_df.shape[-1]), columns=train_df.columns)
    actual_df = inverse_standard_scale(train_df, answer_df)

    submission_df[f"q_{tau}"] = actual_df["TARGET"]

    return submission_df
