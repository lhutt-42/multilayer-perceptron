"""
Model class with mini-batch training.
"""

import logging
from typing import Optional
from itertools import cycle, batched

import numpy as np

from .model import Model
from . import Loss, Metrics, Optimizer, BinaryCrossEntropyLoss


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
        loss: Loss = BinaryCrossEntropyLoss,
        optimizer: Optional[Optimizer] = None,
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
            optimizer (Optimizer): The optimizer to use.
            batch_size (int): The batch size used during training.
        """

        logging.info('Training the model using mini-batch training.')

        self.loss = loss
        self._initialize_layers(x_train.shape[1], optimizer=optimizer)

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

            if epoch % 1000 == 0:
                self.metrics.log(epoch)

        logging.info('Finished training the model.')
