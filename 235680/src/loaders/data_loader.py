import pandas as pd
import os

from src.loaders.path_loader import get_train_data_path, get_test_data_path


def load_train_data() -> pd.DataFrame:
    return pd.read_csv(f"{get_train_data_path()}{os.sep}train.csv")


def load_test_data() -> pd.DataFrame:
    test_dataset = pd.DataFrame()

    for i in range(0, 81):
        test_dataset = pd.concat((test_dataset, pd.read_csv(f"{get_test_data_path()}{os.sep}{i}.csv")))

    num_columns = test_dataset.shape[1]
    return test_dataset
