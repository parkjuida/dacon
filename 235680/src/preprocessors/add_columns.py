import numpy as np
import pandas as pd


def add_sin_cos_day(df: pd.DataFrame) -> pd.DataFrame:
    df[f"Day_sin"] = np.sin(df["Day"] * (2 * np.pi) / 365)
    df[f"Day_cos"] = np.cos(df["Day"] * (2 * np.pi) / 365)

    return df


def add_sin_cos_hour(df: pd.DataFrame) -> pd.DataFrame:
    df[f"Hour_sin"] = np.sin(df["Hour"] * (2 * np.pi) / 24)
    df[f"Hour_cos"] = np.cos(df["Hour"] * (2 * np.pi) / 24)

    return df


def add_ghi(df: pd.DataFrame) -> pd.DataFrame:
    df["GHI"] = df["DHI"] + df["DNI"] * -np.cos(df["Hour"] * (2 * np.pi) / 24)

    return df


def add_min_max_scaled(df: pd.DataFrame, column_name) -> pd.DataFrame:
    minimum = min(df[column_name])
    maximum = max(df[column_name])

    df[f"{column_name}_min_max_scaled"] = (df[column_name] - minimum) / (maximum - minimum)

    return df
