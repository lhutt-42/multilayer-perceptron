"""
Binary cross-entropy loss function for binary classification.
"""

import numpy as np

from .loss import Loss

EPSILON: float = 1e-7  # Used to avoid numerical instability.


class BinaryCrossEntropyLoss(Loss):
    """
    Binary cross-entropy loss function for binary classification.
    """

    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the loss using Binary cross-entropy for binary classification.

        Args:
            y_true (np.ndarray): The true labels (0 or 1).
            y_pred (np.ndarray): The predicted probabilities (sigmoid output).

        Returns:
            float: The loss.
        """
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, EPSILON, 1 - EPSILON)

        loss = -np.mean(
            y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )

        return loss

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the Binary cross-entropy loss.

        When combined with sigmoid activation, the gradient is:
        gradient = (y_pred - y_true)

        Args:
            y_true (np.ndarray): The true labels (0 or 1).
            y_pred (np.ndarray): The predicted probabilities (sigmoid output).

        Returns:
            np.ndarray: The gradient.
        """
        # For sigmoid + binary cross-entropy, gradient is:
        return (y_pred - y_true) / y_pred.shape[0]
