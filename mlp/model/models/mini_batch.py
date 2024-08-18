"""
Model class with mini-batch training.
"""

import logging
from typing import Optional
from itertools import cycle, batched

import numpy as np

from .model import Model
from . import (
    Loss,
    Metrics,
    BinaryCrossEntropyLoss,
    EarlyStopping
)


# pylint: disable=duplicate-code
class MiniBatchModel(Model):
    """
    Model class with mini-batch training.
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
        early_stopping: Optional[EarlyStopping] = None,
        batch_size: int = 32,
        **kwargs
     ) -> None:
        """
        Trains the model.

        Args:
            x_train (np.ndarray): The input data for training.
            y_train (np.ndarray): The target data for training.
            x_test (np.ndarray): The input data for validation.
            y_test (np.ndarray): The target data for validation.
            epochs (int): The number of epochs to train the model.
            loss (Loss): The loss function to use.
            early_stopping (EarlyStopping): The early stopping.
            batch_size (int): The batch size used during training.
        """

        if self.input_size is None or self.output_size is None:
            raise ValueError('The model must be initialized before training.')

        logging.info('Training the model using mini-batch training.')

        self.loss = loss
        self.metrics = Metrics()

        batch_zip = cycle(zip(x_train, y_train))
        batch_cycle = batched(batch_zip, batch_size)

        for (epoch, batch) in zip(range(epochs), batch_cycle):
            x_batch, y_batch = zip(*batch)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            train_output = self.forward(x_batch)
            train_loss = self.loss.forward(y_batch, train_output)
            self.metrics.add_train(y_true=y_batch, y_pred=train_output, loss=train_loss)

            gradient = loss.backward(y_batch, train_output)
            self.backward(gradient)

            test_output = self.forward(x_test)
            test_loss = loss.forward(y_test, test_output)
            self.metrics.add_test(y_true=y_test, y_pred=test_output, loss=test_loss)

            if self._should_stop(epoch, early_stopping):
                break

            if epoch % 1000 == 0:
                self.metrics.log(epoch)

        logging.info('Finished training the model.')
