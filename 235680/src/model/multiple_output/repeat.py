import tensorflow as tf


class RepeatBaseline(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        return inputs

    def get_config(self):
        pass