import tensorflow as tf


class Lstm(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass

    def __init__(self, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(output_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])
