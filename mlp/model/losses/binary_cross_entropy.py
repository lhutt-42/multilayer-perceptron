"""
Binary cross-entropy loss function.
"""

import numpy as np

from .loss import Loss

EPSILON: float = 1e-7 # Used to avoid numerical instability.


class BinaryCrossEntropyLoss(Loss):
    """
    Binary cross-entropy loss function.
    """

    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the loss using binary cross-entropy.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            np.ndarray: The loss.
        """

        y_pred_clipped = np.clip(y_pred, EPSILON, 1 - EPSILON)

        positive_loss = y_true * np.log(y_pred_clipped)
        negative_loss = (1 - y_true) * np.log(1 - y_pred_clipped)

        return -np.mean(positive_loss + negative_loss)


    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss using binary cross-entropy.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            np.ndarray: The gradient.
        """

        y_pred_clipped = np.clip(y_pred, EPSILON, 1 - EPSILON)

        loss_numerator = y_pred_clipped - y_true
        loss_denominator = y_pred_clipped * (1 - y_pred_clipped)

        return loss_numerator / loss_denominator
