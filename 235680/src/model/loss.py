import tensorflow as tf


def pinball_loss(tau):
    @tf.function
    def _pinball_loss(y_true, y_pred):
        return tf.maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))

    return _pinball_loss


def pinball_loss_numpy(y_true, y_pred):
    return np.mean(np.maximum(0.1 * (y_true - y_pred), (0.1 - 1) * (y_true - y_pred)))
