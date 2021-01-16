from src.preprocessors.splitors import split_x_y_for_lightgbm_train
import lightgbm as lgb


def create_lightgbm_dataset(df, days, selector, target_column="TARGET"):
    x, y_1, y_2, columns = split_x_y_for_lightgbm_train(df, days, target_column)
    x = x[selector]
    one_day_dataset = lgb.Dataset(x, label=y_1, feature_name=selector)
    two_day_dataset = lgb.Dataset(x, label=y_2, feature_name=selector)

    return one_day_dataset, two_day_dataset
