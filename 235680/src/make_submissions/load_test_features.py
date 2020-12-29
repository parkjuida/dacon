from datetime import datetime

import os
import pandas as pd
import tensorflow as tf

from src.loaders.data_loader import load_basic_preprocessed_predict, load_test_features
from src.loaders.path_loader import get_data_path
from src.preprocessors.preprocessors import inverse_standard_scale


def make_submission_df(
        model: tf.keras.Model,
        submission_df: pd.DataFrame,
        input_steps: int,
        tau: float,
        cutter=None,
):
    train_df, predict_df = load_basic_preprocessed_predict()
    # train_df: not scaled, predict_df: scaled
    predict_df = load_test_features(predict_df, input_steps)

    if cutter is not None:
        columns = train_df.columns
        num_column = len(columns)
        train_df = train_df[cutter]
        predict_df = pd.DataFrame(
            predict_df.reshape(-1, num_column),
            columns=columns)[cutter].values.reshape(-1, input_steps, len(train_df.columns))

    answer = model.predict(predict_df)
    answer_df = pd.DataFrame(answer.reshape(-1, predict_df.shape[-1]), columns=train_df.columns)
    actual_df = inverse_standard_scale(train_df, answer_df)

    submission_df[f"q_{tau}"] = actual_df["TARGET"]

    return submission_df


def to_submission_csv(submission_df, name):
    base_path = get_data_path()
    time = str(datetime.now()).replace(':', '_')
    if name:
        submission_df.to_csv(f"{base_path}{os.sep}submission{os.sep}{name}_{time}.csv", index=False)
    else:
        submission_df.to_csv(f"{base_path}{os.sep}submission{os.sep}{time}.csv", index=False)
