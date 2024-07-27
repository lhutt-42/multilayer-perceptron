"""
Optimizer interface.
"""

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
    ) -> None:
        """
        Updates the weights and biases.
        """

        raise NotImplementedError
