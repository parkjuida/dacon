import tensorflow as tf


class Feedback(tf.keras.Model):
    def __init__(self, output_steps, num_features, units):
        super().__init__()
        self.output_steps = output_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)

        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None, mask=None):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)

        for n in range(1, self.output_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)

            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    def get_config(self):
        pass

