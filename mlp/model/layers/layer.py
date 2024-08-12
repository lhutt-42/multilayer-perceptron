"""
Layer interface.
"""

from typing import Optional

import numpy as np

from . import (
    Activation,
    Optimizer,
    Initializer,
    ZeroInitializer,
    Regularizer
)


# pylint: disable=too-many-instance-attributes
class Layer:
    """
    Interface for layers.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        layer_size: int,
        activation: Activation,
        weight_initializer: Initializer = ZeroInitializer(),
        bias_initializer: Initializer = ZeroInitializer(),
        optimizer: Optional[Optimizer] = None,
        regularizer: Optional[Regularizer] = None,
        gradient_clipping: Optional[float] = 1.0
    ) -> None:
        """
        Initializes the layer.

        Args:
            layer_size (int): The size of the layer.
            activation (Activation): The activation function.
            weight_initializer (Initializer): The weight initializer.
            bias_initializer (Initializer): The bias initializer.
            optimizer (Optimizer): The optimizer.
            regularizer (Regularizer): The regularizer.
            gradient_clipping (float): The gradient clipping value.
        """

        self.layer_size = layer_size
        self.activation = activation
        self.optimizer = optimizer

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.regularizer = regularizer
        self.gradient_clipping = gradient_clipping

        self.weights: np.ndarray | None = None
        self.biases: np.ndarray | None = None

        self.input: np.ndarray = np.array([])
        self.output: np.ndarray = np.array([])


    def initialize(self, input_size: int) -> None:
        """
        Initializes the layer.

        Args:
            input_size (int): The size of the input.
        """

        self.weights = self.weight_initializer((input_size, self.layer_size))
        self.biases = self.bias_initializer((1, self.layer_size))


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the layer.

        Args:
            x (np.ndarray): The input.

        Returns:
            np.ndarray: The output.
        """

        raise NotImplementedError


    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of the layer.

        Args:
            gradient (np.ndarray): The gradient.

        Returns:
            np.ndarray: The output.
        """

        raise NotImplementedError
