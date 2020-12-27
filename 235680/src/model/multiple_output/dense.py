import tensorflow as tf


class Dense(tf.keras.Model):
    def __init__(self, output_steps, num_features, ):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(output_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass
