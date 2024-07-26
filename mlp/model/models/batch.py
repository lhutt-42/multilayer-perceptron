"""
Model class with batch training.
"""

import numpy as np

from .model import Model
from . import Loss


class BatchModel(Model):
    """
    Model class with batch training.
    """


    # pylint: disable=arguments-differ
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss: Loss,
        epochs: int,
        **kwargs
     ) -> None:
        """
        Trains the model.

        Args:
            x (np.ndarray): The input data.
            y (np.ndarray): The target data.
            loss (Loss): The loss function to use.
            epochs (int): The number of epochs to train the model.
        """

        self.loss = loss
        self._initialize_layers(x.shape[1])

        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(output, y)

            if epoch % 1000 == 0:
                loss_value = self.loss.forward(y, output)
                print(f'Epoch {epoch}, Loss: {loss_value}')
