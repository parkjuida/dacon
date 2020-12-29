import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class WindowGenerator:
    def __init__(
        self,
        train_df,
        valid_df,
        test_df,
        input_width,
        label_width,
        shift,
        label_columns=None
    ):
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return f"""
            Total window size: {self.total_window_size}
            Input indices: {self.input_indices}
            Label indices: {self.label_indices}
            Label column name(s): {self.label_columns}
        """

    def split_window(self, features):
        inputs = features[:, self.input_slice, :-1]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1
            )

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=32,
        )

        return ds.map(self.split_window)

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def valid(self):
        return self.make_dataset(self.valid_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def test_for_predict(self):
        return self.make_dataset(self.test_df, shuffle=False)

    @property
    def example(self):
        result = getattr(self, "_example", None)

        if result is None:
            result = next(iter(self.train))
            self._example = result

        return result

    def plot(self, model=None, plot_col="TARGET", max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(self.input_indices, inputs[6 * n, :, plot_col_index],
                     label="Inputs", marker=".", zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue
                
            plt.scatter(self.label_indices, labels[6 * n, :, label_col_index],
                        edgecolors="k", label="Labels", c="#2ca02c", s=64)
            
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[6 * n, :, label_col_index],
                            marker="X", edgecolors="k", label="Predictions", c="#ff7f0e", s=64)
                
            if n == 0:
                plt.legend()
                
        plt.xlabel("T")
