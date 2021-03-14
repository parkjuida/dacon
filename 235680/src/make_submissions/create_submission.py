import numpy as np
import pandas as pd

from src.loaders.data_loader import generate_test_data
from src.preprocessors.feature_engineering import feature_engineering_lightgbm
from src.preprocessors.preprocessors import apply_minmax_scale


def create_submission_using_lightgbm_model(selector, bst_1, bst_2, days):
    result = []
    for test_data in generate_test_data():
        test_data = feature_engineering_lightgbm(test_data)
        test_np = test_data.values.reshape(7, 48, -1)[(7 - days):, :, :].transpose(1, 0, 2).reshape(48, -1)
        columns = [f"{day}_{column}" for day in range(days) for column in test_data.columns]

        test_df = pd.DataFrame(test_np, columns=columns)
        td = test_df[selector]

        result.append(bst_1.predict(td))
        result.append(bst_2.predict(td))

    return np.array(result).reshape(-1)


def create_submission_using_cnn_model(selector, cnn, train, days):
    result = []
    for test_data in generate_test_data():
        test_data = feature_engineering_lightgbm(test_data)
        _, submission_df, _ = apply_minmax_scale(train, test_data, test_data)

        submission_data = submission_df[selector]

        predict_np = submission_data.values.reshape(7, 48, -1)[(7 - days):, :, :].reshape(-1, days, 48,
                                                                                          submission_data.shape[-1])
        result.append(cnn.predict(predict_np))

    return np.array(result).reshape(-1)


def evaluate_with_submission(q, selector, bst_1, bst_2, days, target_column="TARGET"):
    y_1_error = 0
    y_2_error = 0
    for test_data in generate_test_data():
        test_data = feature_engineering_lightgbm(test_data)
        test_np = test_data.values.reshape(7, 48, -1)[:days, :, :].transpose(1, 0, 2).reshape(48, -1)
        test_y_1 = test_data.values.reshape(7, 48, -1)[days:days + 1, :, :].transpose(1, 0, 2).reshape(48, -1)
        test_y_2 = test_data.values.reshape(7, 48, -1)[days + 1:days + 2, :, :].transpose(1, 0, 2).reshape(48, -1)
        columns = [f"{day}_{column}" for day in range(days) for column in test_data.columns]

        test_x_df = pd.DataFrame(test_np, columns=columns)
        test_y_1_df = pd.DataFrame(test_y_1, columns=test_data.columns)
        test_y_2_df = pd.DataFrame(test_y_2, columns=test_data.columns)

        test_x = test_x_df[selector]
        test_y_1 = test_y_1_df[target_column]
        test_y_2 = test_y_2_df[target_column]

        pred_y_1 = bst_1.predict(test_x).reshape(-1)
        pred_y_2 = bst_2.predict(test_x).reshape(-1)

        y_1_error += np.sum(np.maximum(q * (test_y_1 - pred_y_1), (q - 1) * (test_y_1 - pred_y_1)))
        y_2_error += np.sum(np.maximum(q * (test_y_2 - pred_y_2), (q - 1) * (test_y_2 - pred_y_2)))

    return y_1_error, y_2_error


def evaluate_with_submission_cnn(q, train, cnn, selector, days):
    error = 0
    for test_data in generate_test_data():
        test_data = feature_engineering_lightgbm(test_data)
        test_np = test_data.values.reshape(7, 48, -1)[:days, :, :].reshape(days * 48, -1)
        test_y = test_data.values.reshape(7, 48, -1)[days:days + 2, :, :].reshape(2 * 48, -1)

        test_x_df = pd.DataFrame(test_np, columns=train.columns)
        test_y_df = pd.DataFrame(test_y, columns=train.columns)

        _, test_x, _ = apply_minmax_scale(train, test_x_df, test_x_df)
        test_x = test_x[selector]

        test_y = test_y_df["TARGET"].values
        pred_y = cnn.predict(test_x.values.reshape(-1, days, 48, test_x.shape[-1])).reshape(-1)

        error += np.sum(np.maximum(q * (test_y - pred_y), (q - 1) * (test_y - pred_y)))

    return error
