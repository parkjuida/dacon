import tensorflow as tf


class RepeatBaseline(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        return tf.tile(inputs, tf.constant([1, 2, 1], tf.int32))

    def get_config(self):
        pass
