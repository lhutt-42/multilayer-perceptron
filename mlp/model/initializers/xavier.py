"""
Xavier initializer.
"""

from typing import Tuple

import numpy as np

from .initializer import Initializer


# pylint: disable=too-few-public-methods
class XavierInitializer(Initializer):
    """
    Xavier initializer.
    """

    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Initializes the values with Xavier initialization.

        Args:
            shape (Tuple[int, int]): The shape of the values.

        Returns:
            np.ndarray: The initialized values.
        """

        print(shape)

        fan_in, fan_out = shape[0], shape[1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=shape)
