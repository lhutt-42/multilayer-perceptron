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

        When used with categorical cross-entropy, the gradient simplifies
        to the identity function (gradient is passed through as-is).

        Args:
            x (np.ndarray): The input (softmax output).

        Returns:
            np.ndarray: The gradient (passed through).
        """

        # For softmax + categorical cross-entropy, gradient is just passed through
        return np.ones_like(x)
