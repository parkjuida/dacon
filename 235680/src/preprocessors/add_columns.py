import pandas as pd
import numpy as np


def add_sin_cos_day(df: pd.DataFrame) -> pd.DataFrame:
    df[f"Day_sin"] = np.sin(df["Day"] * (2 * np.pi) / 365)
    df[f"Day_cos"] = np.cos(df["Day"] * (2 * np.pi) / 365)

    return df


def add_sin_cos_hour(df: pd.DataFrame) -> pd.DataFrame:
    df[f"Hour_sin"] = np.sin(df["Hour"] * (2 * np.pi) / 24)
    df[f"Hour_cos"] = np.cos(df["Hour"] * (2 * np.pi) / 24)

    return df


def add_ghi(df: pd.DataFrame) -> pd.DataFrame:
    df["GHI"] = df["DHI"] + df["DNI"]

    return df
