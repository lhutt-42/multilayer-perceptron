"""
Optimizer interface.
"""

from typing import Tuple

import numpy as np


# pylint: disable=too-few-public-methods
class Optimizer:
    """
    Interface for optimizers.
    """

    def update(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        weights_gradient: np.ndarray,
        biases_gradient: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the weights and biases.

        Args:
            weights (np.ndarray): The weights.
            biases (np.ndarray): The biases.
            weights_gradient (np.ndarray): The weights gradient.
            biases_gradient (np.ndarray): The biases gradient.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The updated weights and biases.
        """

        raise NotImplementedError
