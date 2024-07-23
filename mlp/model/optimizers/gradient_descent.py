"""
Gradient descent optimizer.
"""

import numpy as np

from .optimizer import Optimizer


# pylint: disable=too-few-public-methods
class GradientDescentOptimizer(Optimizer):
    """
    Gradient descent optimizer.
    """

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate


    def update(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        weights_gradient: np.ndarray,
        biases_gradient: np.ndarray
    ) -> None:
        """
        Updates the weights and biases using gradient descent.
        """

        weights -= self.learning_rate * weights_gradient
        biases -= self.learning_rate * biases_gradient
