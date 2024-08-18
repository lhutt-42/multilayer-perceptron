"""
Activation interface.
"""

import numpy as np


class Activation:
    """
    Interface for activation functions.
    """

    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs) -> None:
        pass


    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the activation function to the input.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The output.
        """

        raise NotImplementedError


    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the activation function.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The gradient.
        """

        raise NotImplementedError
