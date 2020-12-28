import tensorflow as tf

from src.model.loss import pinball_loss
from src.settings import MAX_EPOCHS


def compile_and_fit_with_pinball_loss(model, window, tau, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min"
    )

    loss_function = pinball_loss(tau)

    model.compile(
        loss=loss_function,
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError(),
                 tf.metrics.MeanSquaredError(),
                 loss_function]
    )

    history = model.fit(
        window.train, epochs=MAX_EPOCHS, validation_data=window.valid, callbacks=[early_stopping]
    )

    return history
