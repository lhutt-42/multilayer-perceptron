"""
Sigmoid activation function.
"""

import numpy as np

from .activation import Activation


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid function to the input.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The output.
        """

        return 1 / (1 + np.exp(-x))


    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the sigmoid function.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The output.
        """

        return self(x) * (1 - self(x))
