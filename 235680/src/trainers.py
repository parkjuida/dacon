import os
from datetime import datetime

import tensorflow as tf

from src.loaders.path_loader import get_data_path
from src.model.loss import pinball_loss
from src.settings import MAX_EPOCHS


def compile_and_fit_with_pinball_loss(model, window, tau, patience=10):
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

    model_save_path = f"{get_data_path()}{os.sep}submission{os.sep}{model.name}{os.sep}{str(datetime.now()).replace(':', '_')}"
    os.makedirs(model_save_path, exist_ok=True)
    model.save_weights(model_save_path)

    return history


def compile_and_fit_with_pinball_loss_once(model, window, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min"
    )

    loss_functions = [pinball_loss(tau / 10) for tau in range(1, 10)]

    model.compile(
        loss=loss_functions,
        optimizer=tf.optimizers.Adam(),
        metrics=[tf.metrics.MeanAbsoluteError(),
                 tf.metrics.MeanSquaredError(),
                 ]
    )

    history = model.fit(
        window.train, epochs=MAX_EPOCHS, validation_data=window.valid, callbacks=[early_stopping]
    )

    model_save_path = f"{get_data_path()}{os.sep}submission{os.sep}{model.name}{os.sep}{str(datetime.now()).replace(':', '_')}"
    os.makedirs(model_save_path, exist_ok=True)
    model.save_weights(model_save_path)

    return history
