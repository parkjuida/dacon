import tensorflow as tf


class MultipleDense(tf.keras.Model):
    def __init__(self):
        super(MultipleDense, self).__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Reshape([1, -1]),
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass
