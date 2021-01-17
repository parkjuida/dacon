from src.preprocessors.add_columns import add_ghi, add_sin_cos_day, add_sin_cos_hour, add_rolling_mean_bulk, add_rh_t


def feature_engineering_lightgbm(df):
    df = add_ghi(df)
    df = add_sin_cos_day(df)
    df = add_sin_cos_hour(df)
    df = add_rolling_mean_bulk(df, "TARGET")
    df = add_rolling_mean_bulk(df, "DHI")
    df = add_rolling_mean_bulk(df, "DNI")
    df = add_rolling_mean_bulk(df, "GHI")
    df = add_rh_t(df)
    df.dropna()

    return df


def feature_engineering_cnn(df):
    df = add_ghi(df)
    df = add_sin_cos_day(df)
    df = add_sin_cos_hour(df)
    df = add_rolling_mean_bulk(df, "TARGET")
    df = add_rolling_mean_bulk(df, "DHI")
    df = add_rolling_mean_bulk(df, "DNI")
    df = add_rolling_mean_bulk(df, "GHI")
    df = add_rh_t(df)
    df.dropna()

    return df
