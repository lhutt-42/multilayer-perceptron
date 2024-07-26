"""
Model class.
"""

import logging
from typing import List

import numpy as np

from . import Layer, Loss


class Model:
    """
    Model class.
    """

    loss: Loss | None = None


    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self.input_size: int | None = None


    def add(self, layers: Layer | List[Layer]) -> None:
        """
        Adds one or multiple layer(s) to the model.

        Args:
            layers (Layer | List[Layer]): The layer(s) to add.
        """

        if isinstance(layers, list):
            for layer in layers:
                self.layers.append(layer)
            return

        self.layers.append(layers)


    def _initialize_layers(self, input_size: int) -> None:
        """
        Initializes the layers of the model.

        Args:
            input_size (int): The size of the input.
        """

        self.input_size = input_size

        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.initialize(self.input_size)
            else:
                layer.initialize(self.layers[i - 1].layer_size)


    def _log_epoch_loss(self, epoch: int, loss_value: float) -> None:
        """
        Displays the epoch and loss value.

        Args:
            epoch (int): The epoch.
            loss_value (float): The loss value.
        """

        logging.info('Epoch %5s, Loss: %.4f', epoch, loss_value)


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the model.

        Args:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The output of the model.
        """

        for layer in self.layers:
            x = layer.forward(x)
        return x


    def backward(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Computes the backward pass of the model.

        Args:
            x (np.ndarray): The input data.
            y (np.ndarray): The target data.
        """

        if self.loss is None:
            raise ValueError('Loss function not set.')

        output_error = self.loss.backward(y, x)
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error)


    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        loss: Loss,
        **kwargs
     ) -> None:
        """
        Trains the model.

        Args:
            x (np.ndarray): The input data.
            y (np.ndarray): The target data.
            loss (Loss): The loss function to use.
        """

        raise NotImplementedError


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the target data.

        Args:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted target data.
        """

        return self.forward(x)
