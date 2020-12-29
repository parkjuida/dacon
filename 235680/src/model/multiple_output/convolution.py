import tensorflow as tf


class Convolution1D(tf.keras.Model):
    name = "Convolution_1D"

    def __init__(self, conv_width, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
            tf.keras.layers.Conv1D(1024, activation="relu", kernel_size=conv_width),
            tf.keras.layers.Dense(output_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass


class Convolution2D3(tf.keras.Model):
    name = "convolution_2D_3"

    def __init__(self, days, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, days, 48, num_features))),
            tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 1, 3, 2))),
            tf.keras.layers.Conv2D(512, activation="relu", kernel_size=(days, num_features)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(output_steps * num_features),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass
