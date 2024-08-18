"""
Model class with batch training.
"""

import logging
from typing import Optional

import numpy as np

from .model import Model
from . import (
    Loss,
    Metrics,
    Optimizer,
    BinaryCrossEntropyLoss,
    EarlyStopping
)


# pylint: disable=duplicate-code
class BatchModel(Model):
    """
    Model class with batch training.
    """

    # pylint: disable=too-many-arguments, arguments-differ, too-many-locals
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        *args,
        loss: Loss = BinaryCrossEntropyLoss,
        optimizer: Optional[Optimizer] = None,
        early_stopping: Optional[EarlyStopping] = None,
        **kwargs
     ) -> None:
        """
        Trains the model.

        Args:
            x_train (np.ndarray): The input data for training.
            y_train (np.ndarray): The target data for training.
            x_val (np.ndarray): The input data for validation.
            y_val (np.ndarray): The target data for validation.
            epochs (int): The number of epochs to train the model.
            loss (Loss): The loss function to use.
            optimizer (Optimizer): The optimizer to use.
            early_stopping (EarlyStopping): The early stopping.
        """

        logging.info('Training the model using batch training.')

        self.loss = loss
        self._initialize_layers(x_train.shape[1], optimizer=optimizer)

        self.metrics = Metrics()

        for epoch in range(epochs):
            train_output = self.forward(x_train)
            train_loss = self.loss.forward(y_train, train_output)
            self.metrics.add_train(y_true=y_train, y_pred=train_output, loss=train_loss)

            gradient = loss.backward(y_train, train_output)
            self.backward(gradient)

            test_output = self.forward(x_test)
            test_loss = loss.forward(y_test, test_output)
            self.metrics.add_test(y_true=y_test, y_pred=test_output, loss=test_loss)

            if self._should_stop(epoch, early_stopping):
                break

            if epoch % 1000 == 0:
                self.metrics.log(epoch)

        logging.info('Finished training the model.')
