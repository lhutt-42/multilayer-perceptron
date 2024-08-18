"""
Model class.
"""

from copy import deepcopy
from typing import List, Optional
import logging

import numpy as np

from . import (
    Layer,
    Loss,
    Metrics,
    Optimizer,
    BinaryCrossEntropyLoss,
    EarlyStopping,
    save_model,
    load_trained_model
)


class Model:
    """
    Model class.
    """

    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs) -> None:
        self.layers: List[Layer] = []
        self.input_size: int | None = None
        self.output_size: int | None = None

        self.loss: Loss | None = None
        self.metrics: Metrics | None = None


    def add(self, layers: Layer | List[Layer]) -> None:
        """
        Adds one or multiple layer(s) to the model.

        Args:
            layers (Layer | List[Layer]): The layer(s) to add.
        """

        if isinstance(layers, list):
            for layer in layers:
                self.layers.append(layer)
            return

        self.layers.append(layers)


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the model.

        Args:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The output of the model.
        """

        for layer in self.layers:
            x = layer.forward(x)
        return x


    def backward(self, gradient: np.ndarray) -> None:
        """
        Computes the backward pass of the model.

        Args:
            gradient (np.ndarray): The gradient.
        """

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


    def initialize(
        self,
        input_size: int,
        output_size: int,
        optimizer: Optional[Optimizer] = None
    ) -> None:
        """
        Initializes the layers of the model.

        Args:
            input_size (int): The size of the input.
            output_size (int): The size of the output.
            optimizer (Optimizer): The optimizer to use.
        """

        self.input_size = input_size
        self.output_size = output_size

        for i, layer in enumerate(self.layers):
            match layer.layer_size:
                case 'input':
                    layer.layer_size = input_size
                    layer.initialize(input_size)
                case 'output':
                    layer.layer_size = output_size
                    layer.initialize(self.layers[i - 1].layer_size)
                case _:
                    layer.initialize(self.layers[i - 1].layer_size)

            if optimizer is not None and layer.optimizer is None:
                layer.optimizer = deepcopy(optimizer)


    def _should_stop(self, epoch: int, early_stopping: Optional[EarlyStopping]) -> bool:
        """
        Checks if the training should stop.

        Args:
            epoch (int): The current epoch.
            early_stopping (EarlyStopping): The early stopping.

        Returns:
            bool: True if the training should stop, False otherwise.
        """

        if early_stopping is None:
            return False

        test_loss = self.metrics.loss.test_values[-1]
        if early_stopping.should_stop(test_loss, self.layers) is False:
            return False

        early_stopping.restore_weights(self.layers)
        self.metrics.log(epoch)
        logging.info('Early stopping the model at epoch %d.', epoch)

        return True

    # pylint: disable=too-many-arguments
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        *args,
        loss: Loss = BinaryCrossEntropyLoss,
        early_stopping: Optional[EarlyStopping] = None,
        **kwargs
     ) -> None:
        """
        Trains the model.

        Args:
            x_train (np.ndarray): The input data for training.
            y_train (np.ndarray): The target data for training.
            x_test (np.ndarray): The input data for validation.
            y_test (np.ndarray): The target data for validation.
            epochs (int): The number of epochs to train the model.
            loss (Loss): The loss function to use.
            early_stopping (EarlyStopping): The early stopping.
        """

        raise NotImplementedError


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the target data.

        Args:
            x (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted target data.
        """

        return self.forward(x)


    def save(self, directory: str) -> None:
        """
        Saves the model to a file.

        Args:
            directory (str): The path to save the model.
        """

        save_model(self, directory)


    @staticmethod
    def load(path: str) -> None:
        """
        Loads the model from a file.

        Args:
            path (str): The path to the model.
        """

        return load_trained_model(path)
