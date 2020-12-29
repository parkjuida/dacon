import tensorflow as tf
import numpy as np


def pinball_loss(tau):
    @tf.function
    def _pinball_loss(y_true, y_pred):
        return tf.reduce_mean(tf.maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred)))

    return _pinball_loss


def pinball_loss_numpy(y_true, y_pred, tau):
    return np.mean(np.maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred)))
