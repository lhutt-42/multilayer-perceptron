"""
Model class.
"""

from typing import List

import numpy as np

from . import Layer, Loss, Metrics


class Model:
    """
    Model class.
    """

    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self.input_size: int | None = None

        self.loss: Loss | None = None
        self.metrics: Metrics | None = None


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
        x_test: np.ndarray,
        y_test: np.ndarray,
        loss: Loss,
        **kwargs
     ) -> None:
        """
        Trains the model.

        Args:
            x_train (np.ndarray): The input data for training.
            y_train (np.ndarray): The target data for training.
            x_test (np.ndarray): The input data for validation.
            y_test (np.ndarray): The target data for validation.
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
