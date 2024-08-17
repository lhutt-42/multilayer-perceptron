"""
Loss function interface.
"""

import numpy as np


class Loss:
    """
    Interface for loss functions.
    """

    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the loss.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            np.ndarray: The loss.
        """

        raise NotImplementedError


    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            np.ndarray: The gradient.
        """

        raise NotImplementedError
