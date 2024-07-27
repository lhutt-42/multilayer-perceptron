"""
He initializer.
"""

from typing import Tuple

import numpy as np

from .initializer import Initializer


# pylint: disable=too-few-public-methods
class HeInitializer(Initializer):
    """
    He initializer.
    """

    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Initializes the values with random values.

        Args:
            shape (Tuple[int, int]): The shape of the values.

        Returns:
            np.ndarray: The initialized values.
        """

        return np.random.randn(*shape) * np.sqrt(2 / shape[0])
