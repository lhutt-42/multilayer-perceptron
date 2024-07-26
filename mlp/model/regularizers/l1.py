"""
L1 regularizer.
"""

import numpy as np

from .regularizer import Regularizer


class L1Regularizer(Regularizer):
    """
    L1 Regularizer.
    """

    def penalty(self, values: np.ndarray) -> np.ndarray:
        """
        Computes the penalty term for the L1 regularizer.

        Args:
            values (np.ndarray): The values to regularize.

        Returns:
            np.ndarray: The penalty term.
        """

        return self.l * np.sum(np.abs(values))


    def gradient(self, values: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the penalty term for the L1 regularizer.

        Args:
            values (np.ndarray): The values to regularize.

        Returns:
            np.ndarray: The gradient.
        """

        return self.l * np.sign(values)
