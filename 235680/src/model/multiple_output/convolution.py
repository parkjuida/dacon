import tensorflow as tf


class Convolution1(tf.keras.Model):
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


class Convolution2(tf.keras.Model):
    def __init__(self, conv_width, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, *conv_width, num_features))),
            tf.keras.layers.Conv2D(512, activation="relu", kernel_size=conv_width),
            tf.keras.layers.Dense(output_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass


class Convolution3(tf.keras.Model):
    def __init__(self, conv_width, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 7, 48, num_features))),
            tf.keras.layers.Conv2D(128, activation="relu", kernel_size=(3, 16)),
            tf.keras.layers.Conv2D(128, activation="relu", kernel_size=(3, 16)),
            tf.keras.layers.Conv2D(128, activation="relu", kernel_size=(3, 16)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass


class Convolution4(tf.keras.Model):
    def __init__(self, conv_width, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 7, 48, num_features))),
            tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 3, 1))),
            tf.keras.layers.Conv2D(1024, activation="relu", kernel_size=(48, 11)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(output_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass

