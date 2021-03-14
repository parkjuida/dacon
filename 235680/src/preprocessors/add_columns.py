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


def add_rolling_mean(df: pd.DataFrame, column, rows) -> pd.DataFrame:
    df[f"{column}_rolling_mean_{rows}"] = df[column].rolling(rows).mean()

    return df


def add_rolling_mean_bulk(df: pd.DataFrame, column) -> pd.DataFrame:
    df[f"{column}_rolling_mean_4"] = df[column].rolling(4).mean()
    df[f"{column}_rolling_mean_8"] = df[column].rolling(8).mean()
    df[f"{column}_rolling_mean_12"] = df[column].rolling(12).mean()
    df[f"{column}_rolling_mean_24"] = df[column].rolling(24).mean()
    df[f"{column}_rolling_mean_48"] = df[column].rolling(48).mean()
    df[f"{column}_rolling_mean_64"] = df[column].rolling(64).mean()
    df[f"{column}_rolling_mean_96"] = df[column].rolling(96).mean()

    return df


def add_rolling_std_bulk(df: pd.DataFrame, column) -> pd.DataFrame:
    df[f"{column}_rolling_std_4"] = df[column].rolling(4).std()
    df[f"{column}_rolling_std_8"] = df[column].rolling(8).std()
    df[f"{column}_rolling_std_12"] = df[column].rolling(12).std()
    df[f"{column}_rolling_std_24"] = df[column].rolling(24).std()
    df[f"{column}_rolling_std_48"] = df[column].rolling(48).std()
    df[f"{column}_rolling_std_64"] = df[column].rolling(64).std()
    df[f"{column}_rolling_std_96"] = df[column].rolling(96).std()

    return df


def add_min_max_scaled(df: pd.DataFrame, column_name) -> pd.DataFrame:
    minimum = min(df[column_name])
    maximum = max(df[column_name])

    df[f"{column_name}_min_max_scaled"] = (df[column_name] - minimum) / (maximum - minimum)

    return df


def add_rh_t(df):
    df["RH_T"] = df["RH"] * df["T"]

    return df

