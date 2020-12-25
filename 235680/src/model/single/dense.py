import tensorflow as tf


class TwoLayerDense(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self):
        super(TwoLayerDense, self).__init__()
        self._dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self._dense_2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)

    def call(self, inputs, training=None, mask=None):
        x = self._dense_1(inputs)
        return self._dense_2(x)
