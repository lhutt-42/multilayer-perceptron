"""
Accuracy metrics class.
"""

import numpy as np

from .data import Data


class AccuracyMetrics(Data):
    """
    Class to store the accuracy metrics.
    """

    # pylint: disable=arguments-differ
    def add_train(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a training accuracy value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        true_labels = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)

        accuracy = np.mean(true_labels == pred_labels)
        self.train_values.append(accuracy)


    # pylint: disable=arguments-differ
    def add_test(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a validation accuracy value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        true_labels = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)

        accuracy = np.mean(true_labels == pred_labels)
        self.test_values.append(accuracy)
