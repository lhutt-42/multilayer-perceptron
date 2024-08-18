"""
Adam optimizer.
"""

from typing import Tuple

import numpy as np

from .optimizer import Optimizer

EPSILON: float = 1e-7 # Used to avoid numerical instability.


# pylint: disable=too-few-public-methods,too-many-instance-attributes
class AdamOptimizer(Optimizer):
    """
    Adam optimizer.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *args,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        **kwargs
    ) -> None:
        """
        Initializes the optimizer.

        Args:
            learning_rate (float): The learning rate.
            beta1 (float): The exponential decay rate for the first moment estimates.
            beta2 (float): The exponential decay rate for the second moment estimates.
        """

        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def update(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        weights_gradient: np.ndarray,
        biases_gradient: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the weights and biases using gradient descent.

        Args:
            weights (np.ndarray): The weights.
            biases (np.ndarray): The biases.
            weights_gradient (np.ndarray): The weights gradient.
            biases_gradient (np.ndarray): The biases gradient.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The updated weights and biases.
        """

        if self.m_w is None:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)

        self.t += 1

        # Update biased first moment estimate
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_gradient
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * biases_gradient

        # Update biased second raw moment estimate
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (weights_gradient ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (biases_gradient ** 2)

        # Compute bias-corrected first moment estimate
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + EPSILON)
        biases -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + EPSILON)

        return weights, biases
