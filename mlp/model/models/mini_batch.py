"""
Model class with mini-batch training.
"""

from itertools import islice, cycle, batched

import numpy as np

from .model import Model
from . import Loss


class MiniBatchModel(Model):
    """
    Model class with mini-batch training.
    """


    # pylint: disable=arguments-differ, too-many-arguments
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss: Loss,
        epochs: int,
        batch_size: int,
        **kwargs
     ) -> None:
        """
        Trains the model.

        Args:
            x (np.ndarray): The input data.
            y (np.ndarray): The target data.
            loss (Loss): The loss function to use.
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size used during training.
        """

        self.loss = loss
        self._initialize_layers(x.shape[1])

        batch_zip = zip(cycle(x), cycle(y))
        batch_cycle = batched(cycle(batch_zip), batch_size)

        for (epoch, batch) in zip(range(epochs), islice(batch_cycle, epochs)):
            x_batch, y_batch = zip(*batch)

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            output = self.forward(x_batch)
            self.backward(output, y_batch)

            if epoch % 1000 == 0:
                output = self.forward(x_batch)
                loss_value = self.loss.forward(y_batch, output)
                print(f'Epoch {epoch}, Loss: {loss_value}')
