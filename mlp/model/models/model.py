"""
Model class.
"""

import logging
from typing import List

import numpy as np

from . import Layer, Loss, LossMetrics


class Model:
    """
    Model class.
    """

    loss: Loss | None = None
    loss_metrics: LossMetrics = LossMetrics()


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


    def _log_epoch_loss(self, epoch: int) -> None:
        """
        Displays the epoch and loss value.

        Args:
            epoch (int): The epoch.
            loss_value (float): The loss value.
        """

        logging.info(
            'Epoch %5s, Train Loss: %.4f, Val Loss: %.4f',
            epoch,
            self.loss_metrics.train_loss[-1],
            self.loss_metrics.val_loss[-1]
        )


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


    def backward(self, gradient: np.ndarray) -> None:
        """
        Computes the backward pass of the model.

        Args:
            gradient (np.ndarray): The gradient.
        """

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


    # pylint: disable=too-many-arguments
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        loss: Loss,
        **kwargs
     ) -> LossMetrics:
        """
        Trains the model.

        Args:
            x_train (np.ndarray): The input data for training.
            y_train (np.ndarray): The target data for training.
            x_val (np.ndarray): The input data for validation.
            y_val (np.ndarray): The target data for validation.
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
