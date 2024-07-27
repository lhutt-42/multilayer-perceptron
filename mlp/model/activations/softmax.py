"""
Softmax activation function.
"""

import numpy as np

from .activation import Activation


class SoftmaxActivation(Activation):
    """
    Softmax activation function.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function to the input.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The output.
        """

        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)


    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the softmax function.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The output.
        """

        s = self(x)
        return s * (1 - s)
