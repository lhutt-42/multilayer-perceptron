"""
Interface for initializers.
"""

from typing import Tuple

import numpy as np


# pylint: disable=too-few-public-methods
class Initializer:
    """
    Interface for initializers.
    """

    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs) -> None:
        pass


    def __call__(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Initializes the values.

        Args:
            shape (Tuple[int, int]): The shape of the values.

        Returns:
            np.ndarray: The initialized values.
        """

        raise NotImplementedError
