import pandas as pd
import os

from src.path_loader import get_train_data_path


def load_train_data():
    return pd.read_csv(f"{get_train_data_path()}{os.sep}train.csv")
