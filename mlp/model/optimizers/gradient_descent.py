"""
Gradient descent optimizer.
"""

from typing import Tuple

import numpy as np

from .optimizer import Optimizer


# pylint: disable=too-few-public-methods
class GradientDescentOptimizer(Optimizer):
    """
    Gradient descent optimizer.
    """

    # pylint: disable=unused-argument
    def __init__(self, *args, learning_rate: float = 0.001, **kwargs) -> None:
        """
        Initializes the optimizer.

        Args:
            learning_rate (float): The learning rate.
        """

        super().__init__()
        self.learning_rate = learning_rate

    def update(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        weights_gradient: np.ndarray,
        biases_gradient: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the weights and biases using gradient descent.

        Args:
            weights (np.ndarray): The weights.
            biases (np.ndarray): The biases.
            weights_gradient (np.ndarray): The weights gradient.
            biases_gradient (np.ndarray): The biases gradient.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The updated weights and biases.
        """

        weights -= self.learning_rate * weights_gradient
        biases -= self.learning_rate * biases_gradient

        return weights, biases
