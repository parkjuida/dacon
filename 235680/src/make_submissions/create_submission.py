import numpy as np
import pandas as pd

from src.loaders.data_loader import generate_test_data
from src.preprocessors.feature_engineering import feature_engineering_lightgbm


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
