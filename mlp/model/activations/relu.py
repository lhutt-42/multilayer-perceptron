"""
ReLU activation function.
"""

import numpy as np

from .activation import Activation


class ReluActivation(Activation):
    """
    ReLU activation function.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU function to the input.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The output.
        """

        return np.maximum(0, x)


    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the ReLU function.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The output.
        """

        return np.where(x > 0, 1, 0)
