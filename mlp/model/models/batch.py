"""
Model class with batch training.
"""

import numpy as np

from .model import Model
from . import Loss, LossMetrics


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
        x_val: np.ndarray,
        y_val: np.ndarray,
        loss: Loss,
        epochs: int,
        **kwargs
     ) -> None:
        """
        Trains the model.

        Args:
            x_train (np.ndarray): The input data for training.
            y_train (np.ndarray): The target data for training.
            x_val (np.ndarray): The input data for validation.
            y_val (np.ndarray): The target data for validation.
            loss (Loss): The loss function to use.
            epochs (int): The number of epochs to train the model.
        """

        self.loss = loss
        self._initialize_layers(x_train.shape[1])
        self.loss_metrics = LossMetrics()

        for epoch in range(epochs):
            train_output = self.forward(x_train)
            train_loss = self.loss.forward(y_train, train_output)
            self.loss_metrics.add_train_loss(train_loss)

            gradient = loss.backward(y_train, train_output)
            self.backward(gradient)

            val_output = self.forward(x_val)
            val_loss = loss.forward(y_val, val_output)
            self.loss_metrics.add_val_loss(val_loss)

            if epoch % 1000 == 0:
                self._log_epoch_loss(epoch)
