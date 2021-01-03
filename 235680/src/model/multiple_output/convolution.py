import tensorflow as tf


class Convolution1D(tf.keras.Model):
    name = "Convolution_1D"

    def __init__(self, conv_width, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
            tf.keras.layers.Conv1D(1024, activation="relu", kernel_size=conv_width),
            tf.keras.layers.Dense(output_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass


class Convolution1D2(tf.keras.Model):
    name = "Convolution_1D_2"

    def __init__(self, conv_width, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
            tf.keras.layers.Conv1D(2048, activation="relu", kernel_size=conv_width),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(output_steps * num_features,
                                  ),
            tf.keras.layers.Reshape([output_steps, num_features])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass


class Convolution2D3(tf.keras.Model):
    name = "convolution_2D_3"

    def __init__(self, days, output_steps, num_features):
        super().__init__()
        self._model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, days, 48, num_features))),
            tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 1, 3, 2))),
            tf.keras.layers.Conv2D(128, activation="relu", kernel_size=(days, num_features)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(output_steps * 1),
            tf.keras.layers.Reshape([output_steps, 1])
        ])

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs)

    def get_config(self):
        pass


class ConvolutionVarious(tf.keras.Model):
    name = "Convolution_various"

    def __init__(self, conv_width):
        super().__init__()
        self.conv1_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :])
        self.conv2_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 2:, :])
        self.conv3_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 3:, :])
        self.conv4_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 4:, :])
        self.conv5_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 6:, :])
        self.conv6_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 48:, :])

        self.conv1 = tf.keras.layers.Conv1D(32, kernel_size=conv_width, activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(64, kernel_size=conv_width // 2, activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(128, kernel_size=conv_width // 3, activation="relu")
        self.conv4 = tf.keras.layers.Conv1D(256, kernel_size=conv_width // 4, activation="relu")
        self.conv5 = tf.keras.layers.Conv1D(512, kernel_size=conv_width // 6, activation="relu")
        self.conv6 = tf.keras.layers.Conv1D(1024, kernel_size=conv_width // 48, activation="relu")

        self.conv1_1 = tf.keras.layers.Conv1D(8, kernel_size=1, activation="relu")
        self.conv2_1 = tf.keras.layers.Conv1D(8, kernel_size=1, activation="relu")
        self.conv3_1 = tf.keras.layers.Conv1D(16, kernel_size=1, activation="relu")
        self.conv4_1 = tf.keras.layers.Conv1D(16, kernel_size=1, activation="relu")
        self.conv5_1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="relu")
        self.conv6_1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="relu")

        self.dense1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense2 = tf.keras.layers.Dense(512, activation="relu")
        self.dense3 = tf.keras.layers.Dense(96)
        self.output_layer = tf.keras.layers.Reshape([96, 1])

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        conv1_model = self.conv1_1(self.conv1(self.conv1_input(inputs)))
        conv2_model = self.conv2_1(self.conv2(self.conv2_input(inputs)))
        conv3_model = self.conv3_1(self.conv3(self.conv3_input(inputs)))
        conv4_model = self.conv4_1(self.conv4(self.conv4_input(inputs)))
        conv5_model = self.conv5_1(self.conv5(self.conv5_input(inputs)))
        conv6_model = self.conv6_1(self.conv6(self.conv6_input(inputs)))

        net = tf.concat(axis=2, values=[conv1_model, conv2_model, conv3_model, conv4_model, conv5_model, conv6_model])
        # net = self.dense1(net)
        # net = self.dense2(net)
        net = self.dense3(net)
        return self.output_layer(net)


class ConvolutionVarious2(tf.keras.Model):
    name = "Convolution_various_2"

    def __init__(self, conv_width):
        super().__init__()
        self.conv1_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :])
        self.conv2_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 2:, :])
        self.conv3_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 3:, :])
        self.conv4_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 4:, :])
        self.conv5_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 6:, :])
        self.conv6_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 8:, :])
        self.conv7_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 12:, :])
        self.conv8_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 16:, :])
        self.conv9_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 24:, :])
        self.conv10_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 48:, :])

        self.conv1 = tf.keras.layers.Conv1D(64, kernel_size=conv_width, activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(64, kernel_size=conv_width // 2, activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(128, kernel_size=conv_width // 3, activation="relu")
        self.conv4 = tf.keras.layers.Conv1D(128, kernel_size=conv_width // 4, activation="relu")
        self.conv5 = tf.keras.layers.Conv1D(256, kernel_size=conv_width // 6, activation="relu")
        self.conv6 = tf.keras.layers.Conv1D(256, kernel_size=conv_width // 8, activation="relu")
        self.conv7 = tf.keras.layers.Conv1D(512, kernel_size=conv_width // 12, activation="relu")
        self.conv8 = tf.keras.layers.Conv1D(512, kernel_size=conv_width // 16, activation="relu")
        self.conv9 = tf.keras.layers.Conv1D(1024, kernel_size=conv_width // 24, activation="relu")
        self.conv10 = tf.keras.layers.Conv1D(1024, kernel_size=conv_width // 48, activation="relu")

        self.conv1_1 = tf.keras.layers.Conv1D(16, kernel_size=1, activation="relu")
        self.conv2_1 = tf.keras.layers.Conv1D(16, kernel_size=1, activation="relu")
        self.conv3_1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="relu")
        self.conv4_1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="relu")
        self.conv5_1 = tf.keras.layers.Conv1D(64, kernel_size=1, activation="relu")
        self.conv6_1 = tf.keras.layers.Conv1D(64, kernel_size=1, activation="relu")
        self.conv7_1 = tf.keras.layers.Conv1D(128, kernel_size=1, activation="relu")
        self.conv8_1 = tf.keras.layers.Conv1D(128, kernel_size=1, activation="relu")
        self.conv9_1 = tf.keras.layers.Conv1D(256, kernel_size=1, activation="relu")
        self.conv10_1 = tf.keras.layers.Conv1D(256, kernel_size=1, activation="relu")

        self.dense1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense2 = tf.keras.layers.Dense(512, activation="relu")
        self.dense3 = tf.keras.layers.Dense(96)
        self.output_layer = tf.keras.layers.Reshape([96, 1])

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        conv1_model = self.conv1_1(self.conv1(self.conv1_input(inputs)))
        conv2_model = self.conv2_1(self.conv2(self.conv2_input(inputs)))
        conv3_model = self.conv3_1(self.conv3(self.conv3_input(inputs)))
        conv4_model = self.conv4_1(self.conv4(self.conv4_input(inputs)))
        conv5_model = self.conv5_1(self.conv5(self.conv5_input(inputs)))
        conv6_model = self.conv6_1(self.conv6(self.conv6_input(inputs)))
        conv7_model = self.conv7_1(self.conv7(self.conv7_input(inputs)))
        conv8_model = self.conv8_1(self.conv8(self.conv8_input(inputs)))
        conv9_model = self.conv9_1(self.conv9(self.conv9_input(inputs)))
        conv10_model = self.conv10_1(self.conv10(self.conv10_input(inputs)))

        net = tf.concat(axis=2, values=[
            conv1_model,
            conv2_model,
            conv3_model,
            conv4_model,
            conv5_model,
            conv6_model,
            conv7_model,
            conv8_model,
            conv9_model,
            conv10_model,
        ])
        # net = self.dense1(net)
        # net = self.dense2(net)
        net = self.dense3(net)
        return self.output_layer(net)


class ConvolutionVariousSmall(tf.keras.Model):
    name = "Convolution_various_small"

    def __init__(self, conv_width):
        super().__init__()
        self.conv1_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :])
        self.conv2_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 2:, :])
        self.conv3_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 3:, :])
        self.conv4_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 4:, :])

        self.conv1 = tf.keras.layers.Conv1D(64, kernel_size=conv_width, activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(64, kernel_size=conv_width // 2, activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(128, kernel_size=conv_width // 3, activation="relu")
        self.conv4 = tf.keras.layers.Conv1D(128, kernel_size=conv_width // 4, activation="relu")

        self.conv1_1 = tf.keras.layers.Conv1D(16, kernel_size=1, activation="relu")
        self.conv2_1 = tf.keras.layers.Conv1D(16, kernel_size=1, activation="relu")
        self.conv3_1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="relu")
        self.conv4_1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="relu")

        self.dense3 = tf.keras.layers.Dense(96)
        self.output_layer = tf.keras.layers.Reshape([96, 1])

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        conv1_model = self.conv1_1(self.conv1(self.conv1_input(inputs)))
        conv2_model = self.conv2_1(self.conv2(self.conv2_input(inputs)))
        conv3_model = self.conv3_1(self.conv3(self.conv3_input(inputs)))
        conv4_model = self.conv4_1(self.conv4(self.conv4_input(inputs)))

        net = tf.concat(axis=2, values=[
            conv1_model,
            conv2_model,
            conv3_model,
            conv4_model,
        ])
        # net = self.dense1(net)
        # net = self.dense2(net)
        net = self.dense3(net)
        return self.output_layer(net)


class ConvolutionVariousSmallReturnOnce(tf.keras.Model):
    name = "Convolution_various_small_return_once"

    def __init__(self, conv_width):
        super().__init__()
        self.conv1_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :])
        self.conv2_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 2:, :])
        self.conv3_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 3:, :])
        self.conv4_input = tf.keras.layers.Lambda(lambda x: x[:, -conv_width // 4:, :])

        self.conv1 = tf.keras.layers.Conv1D(64, kernel_size=conv_width, activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(64, kernel_size=conv_width // 2, activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(128, kernel_size=conv_width // 3, activation="relu")
        self.conv4 = tf.keras.layers.Conv1D(128, kernel_size=conv_width // 4, activation="relu")

        self.conv1_1 = tf.keras.layers.Conv1D(16, kernel_size=1, activation="relu")
        self.conv2_1 = tf.keras.layers.Conv1D(16, kernel_size=1, activation="relu")
        self.conv3_1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="relu")
        self.conv4_1 = tf.keras.layers.Conv1D(32, kernel_size=1, activation="relu")


        self.dense2 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense3 = tf.keras.layers.Dense(512, activation="relu")

        self.dense_list = []
        self.output_list = []
        for i in range(1, 10):
            dense1 = tf.keras.layers.Dense(96)
            output_layer1 = tf.keras.layers.Reshape([96, 1])
            self.dense_list.append(dense1)
            self.output_list.append(output_layer1)

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        conv1_model = self.conv1_1(self.conv1(self.conv1_input(inputs)))
        conv2_model = self.conv2_1(self.conv2(self.conv2_input(inputs)))
        conv3_model = self.conv3_1(self.conv3(self.conv3_input(inputs)))
        conv4_model = self.conv4_1(self.conv4(self.conv4_input(inputs)))

        net = tf.concat(axis=2, values=[
            conv1_model,
            conv2_model,
            conv3_model,
            conv4_model,
        ])

        net = self.dense2(net)
        net = self.dense3(net)

        return tf.concat(axis=2, values=[
            output(dense(net)) for dense, output in zip(self.dense_list, self.output_list)
        ])

