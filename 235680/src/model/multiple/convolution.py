import tensorflow as tf


class SimpleConv(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._conv1 = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=(48, ),
            activation=tf.nn.relu,
        )
        self._dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self._output = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self._conv1(inputs)
        x = self._dense1(x)
        return self._output(x)

    def get_config(self):
        pass
