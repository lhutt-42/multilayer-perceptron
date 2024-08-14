"""
Early stopping class.
"""

from typing import List

import numpy as np

from . import Layer


# pylint: disable=too-many-instance-attributes
class EarlyStopping:
    """
    Early stopping class.
    """

    def __init__(self, patience: int = 5, delta: float = 0.0):
        """
        Initializes the early stopping.

        Args:
            patience (int): The number of epochs to wait before stopping.
            delta (float): The minimum change in the loss to qualify as an improvement.
        """

        self.patience = patience
        self.delta = delta

        self.counter: int = 0
        self.best_loss: float = np.inf

        self.weights: List[np.ndarray] | None = None
        self.biases: List[np.ndarray] | None = None


    def save_weights(self, layers: List[Layer]) -> None:
        """
        Saves the weights and biases of the layers.

        Args:
            layers (List[Layer]): The layers of the model
        """

        self.weights = [
            np.copy(layer.weights) for layer in layers
        ]
        self.biases = [
            np.copy(layer.biases) for layer in layers
        ]


    def restore_weights(self, layers: List[Layer]) -> None:
        """
        Restores the weights and biases of the layers.

        Args:
            layers (List[Layer]): The layers of the model
        """

        if self.weights is None or self.biases is None:
            return

        for (i, layer) in enumerate(layers):
            layer.weights = self.weights[i]
            layer.biases = self.biases[i]


    def should_stop(self, loss: float, layers: List[Layer]) -> bool:
        """
        Checks if the training should stop.

        Args:
            loss (float): The current loss.
            layers (List[Layer]): The layers of the model.

        Returns:
            bool: Whether the training should stop.
        """

        if loss > self.best_loss - self.delta:
            self.counter += 1
            return self.counter >= self.patience

        self.best_loss = loss
        self.counter = 0
        self.save_weights(layers)
        return False
