"""
L2 regularizer.
"""

import numpy as np

from .regularizer import Regularizer


class L2Regularizer(Regularizer):
    """
    L2 Regularizer.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *args,
        l: float = 0.01,
        **kwargs
    ) -> None:
        """
        Initializes the regularizer.

        Args:
            l (float): The regularization parameter.
        """

        super().__init__()
        self.l = l


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
