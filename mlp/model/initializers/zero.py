"""
Zero initializer.
"""

from typing import Tuple

import numpy as np

from .initializer import Initializer


# pylint: disable=too-few-public-methods
class ZeroInitializer(Initializer):
    """
    Zero initializer.
    """

    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Initializes the values with zeros.

        Args:
            shape (Tuple[int, int]): The shape of the values.

        Returns:
            np.ndarray: The initialized values.
        """

        return np.zeros(shape)
