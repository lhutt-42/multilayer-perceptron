"""
Sigmoid activation function.
"""

import numpy as np

from .activation import Activation


class SigmoidActivation(Activation):
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

        x_clipped = np.clip(x, -500, 500)
        return np.where(
            x_clipped >= 0,
            1 / (1 + np.exp(-x_clipped)),
            np.exp(x_clipped) / (1 + np.exp(x_clipped))
        )


    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the sigmoid function.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The output.
        """

        s = self(x)
        return s * (1 - s)
