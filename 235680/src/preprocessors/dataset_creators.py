from src.loaders.window_generator import WindowGenerator
from src.preprocessors.preprocessors import apply_minmax_scale
from src.preprocessors.splitors import split_x_y_for_lightgbm_train
import lightgbm as lgb


def create_lightgbm_dataset(df, days, selector, target_column="TARGET"):
    x, y_1, y_2, columns = split_x_y_for_lightgbm_train(df, days, target_column)
    x = x[selector]
    one_day_dataset = lgb.Dataset(x, label=y_1, feature_name=selector)
    two_day_dataset = lgb.Dataset(x, label=y_2, feature_name=selector)

    return one_day_dataset, two_day_dataset


def create_cnn_dataset(train, valid, test, days, selector):
    train_target = train["TARGET"]
    valid_target = valid["TARGET"]
    test_target = test["TARGET"]

    train_df, valid_df, test_df = apply_minmax_scale(train, valid, test)

    train_df = train_df[selector]
    valid_df = valid_df[selector]
    test_df = test_df[selector]

    train_df["ANSWER"] = train_target
    valid_df["ANSWER"] = valid_target
    test_df["ANSWER"] = test_target

    window_generator = WindowGenerator(
        train_df,
        valid_df,
        test_df,
        input_width=48 * days,
        label_width=96,
        shift=96,
        sequence_stride=1,
        label_columns=["ANSWER"]
    )

    return window_generator
