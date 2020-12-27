import tensorflow as tf


class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, output_step):
        super().__init__()
        self.output_step = output_step

    def call(self, inputs, training=None, mask=None):
        return tf.tile(inputs[:, -1:, :], [1, self.output_step, 1])

    def get_config(self):
        pass
