"""
Accuracy metrics class.
"""

import numpy as np

from .data import Data


class AccuracyMetrics(Data):
    """
    Class to store the accuracy metrics.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the accuracy metrics class.
        """

        super().__init__(*args, **kwargs)
        self.name = 'Accuracy'


    # pylint: disable=arguments-differ
    def add_train(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a training accuracy value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        accuracy = self.calculate_accuracy(y_true, y_pred)
        self.train_values.append(accuracy)


    # pylint: disable=arguments-differ
    def add_test(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a validation accuracy value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        accuracy = self.calculate_accuracy(y_true, y_pred)
        self.test_values.append(accuracy)


    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the accuracy.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            float: The accuracy.
        """

        true_labels = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)

        accuracy = np.mean(true_labels == pred_labels)
        return accuracy
