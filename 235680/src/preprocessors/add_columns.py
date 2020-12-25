import pandas as pd
import numpy as np


def add_sin_cos(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df[f"{column_name}_sin"] = np.sin(df[column_name] * (2 * np.pi) / 24)
    df[f"{column_name}_cos"] = np.cos(df[column_name] * (2 * np.pi) / 24)

    return df