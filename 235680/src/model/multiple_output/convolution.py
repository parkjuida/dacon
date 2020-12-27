import tensorflow as tf


class Convolution(tf.keras.Model):
    def __init__(self, conv_width, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
            tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(conv_width)),
            tf.keras.layers.Dense(output_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass

