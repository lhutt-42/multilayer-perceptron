"""
Interface for regularizers.
"""

import numpy as np


class Regularizer:
    """
    Regularizers interface.
    """

    def __init__(self, l: float = 0.01) -> None:
        """
        Initializes the regularizer.

        Args:
            l (float): The regularization parameter.
        """

        self.l = l


    def penalty(self, values: np.ndarray) -> np.ndarray:
        """
        Computes the penalty term.

        Args:
            values (np.ndarray): The values to regularize.

        Returns:
            np.ndarray: The penalty term.
        """

        raise NotImplementedError("Penalty method not implemented")


    def gradient(self, values: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the penalty term.

        Args:
            values (np.ndarray): The values to regularize.

        Returns:
            np.ndarray: The gradient.
        """

        raise NotImplementedError("Gradient method not implemented")
