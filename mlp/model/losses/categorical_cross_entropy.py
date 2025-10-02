"""
Categorical cross-entropy loss function.
"""

import numpy as np

from .loss import Loss

EPSILON: float = 1e-7  # Used to avoid numerical instability.


class CategoricalCrossEntropyLoss(Loss):
    """
    Categorical cross-entropy loss function.
    """

    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the loss using categorical cross-entropy.

        Args:
            y_true (np.ndarray): The true labels (one-hot encoded).
            y_pred (np.ndarray): The predicted probabilities (softmax output).

        Returns:
            float: The loss.
        """

        y_pred_clipped = np.clip(y_pred, EPSILON, 1 - EPSILON)

        # Compute categorical cross-entropy
        loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)

        return np.mean(loss)


    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the categorical cross-entropy loss.

        When combined with softmax activation, this simplifies to:
        gradient = y_pred - y_true

        Args:
            y_true (np.ndarray): The true labels (one-hot encoded).
            y_pred (np.ndarray): The predicted probabilities (softmax output).

        Returns:
            np.ndarray: The gradient.
        """

        # For softmax + categorical cross-entropy, gradient is simply:
        return (y_pred - y_true) / y_pred.shape[0]
