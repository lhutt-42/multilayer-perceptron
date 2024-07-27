"""
L2 regularizer.
"""

import numpy as np

from .regularizer import Regularizer


class L2Regularizer(Regularizer):
    """
    L2 Regularizer.
    """

    def penalty(self, values: np.ndarray) -> np.ndarray:
        """
        Computes the penalty term for the L2 regularizer.

        Args:
            values (np.ndarray): The values to regularize.

        Returns:
            np.ndarray: The penalty term.
        """

        return 0.5 * self.l * np.sum(values ** 2)


    def gradient(self, values: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the penalty term for the L2 regularizer.

        Args:
            values (np.ndarray): The values to regularize.

        Returns:
            np.ndarray: The gradient.
        """

        return self.l * values
