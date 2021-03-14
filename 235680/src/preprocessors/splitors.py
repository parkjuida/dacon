import pandas as pd
import numpy as np


def split_train_valid_test_for_cv(df, ratio):
    length = df.shape[0]
    boundary = {
        0.6: [(slice(0, int(length * 0.6)), slice(int(length * 0.6), int(length * 0.9)),
               slice(int(length * 0.9), int(length * 1.0))),
              # (slice(int(length * 0.1), int(length * 0.7)), slice(int(length * 0.7), int(length * 1)),
              #  slice(int(length * 0), int(length * 0.1))),
              # (slice(int(length * 0.3), int(length * 0.9)), slice(int(length * 0.9), int(length * 1)),
              #  slice(int(length * 0), int(length * 0.3))),
              (slice(int(length * 0.4), int(length * 1)), slice(int(length * 0), int(length * 0.3)),
               slice(int(length * 0.3), int(length * 0.4))),
              ],
        0.8: [
            (slice(0, int(length * 0.8)), slice(int(length * 0.8), int(length * 1.0)),
             slice(int(length * 0.9), int(length * 1.0))),
            # (slice(int(length * 0.1), int(length * 0.9)), slice(int(length * 0.9), int(length * 1)),
            #  slice(int(length * 0), int(length * 0.1))),
            (slice(int(length * 0.2), int(length * 1)), slice(int(length * 0), int(length * 0.2)),
             slice(int(length * 0.1), int(length * 0.2))),
        ],
        0.5: [
            (slice(0, int(length * 0.5)), slice(int(length * 0.5), int(length * 0.8)),
             slice(int(length * 0.8), int(length * 1.0))),
            (slice(int(length * 0.5), int(length * 1)), slice(int(length * 0), int(length * 0.3)),
             slice(int(length * 0.3), int(length * 0.5))),
        ],
        0.7: [
            (slice(0, int(length * 0.7)), slice(int(length * 0.7), int(length * 0.9)),
             slice(int(length * 0.9), int(length * 1.0)))
        ]
    }

    for train_slice, valid_slice, test_slice in boundary[ratio]:
        yield df[train_slice], df[valid_slice], df[test_slice]


def split_x_y_for_lightgbm_train(df, days, target_column="TARGET"):
    x = []
    y_1 = []
    y_2 = []
    feature_shape = df.shape[-1]
    columns = [f"{day}_{column}" for day in range(days) for column in df.columns]

    one_day = 48
    one_set_unit = one_day * days

    for i in range(0, df.shape[0] - (one_set_unit + one_day * 2) + 1, 48):
        tmp = df[i: i + one_set_unit + 2 * one_day].values
        tmp = tmp.reshape(days + 2, 48, -1).transpose(1, 0, 2)

        x.append(tmp[:, :days, :])
        y_1.append(tmp[:, days: days + 1, :])
        y_2.append(tmp[:, days + 1:days + 2, :])

    x = pd.DataFrame(np.array(x).reshape(-1, feature_shape * days), columns=columns)
    y_1 = pd.DataFrame(np.array(y_1).reshape(-1, feature_shape), columns=df.columns)
    y_2 = pd.DataFrame(np.array(y_2).reshape(-1, feature_shape), columns=df.columns)

    return x, y_1[target_column], y_2[target_column], columns

