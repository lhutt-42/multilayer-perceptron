"""
Precision metrics class.
"""

import numpy as np

from .data import Data


class PrecisionMetrics(Data):
    """
    Class to store the precision metrics.
    """

    # pylint: disable=arguments-differ
    def add_train(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a training precision value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        precision = self.calculate_precision(y_true, y_pred)
        self.train_values.append(precision)


    # pylint: disable=arguments-differ
    def add_test(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a validation precision value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        precision = self.calculate_precision(y_true, y_pred)
        self.test_values.append(precision)


    @staticmethod
    def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the precision.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            float: The precision.
        """

        true_labels = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)

        true_positive = np.sum((pred_labels == 1) & (true_labels == 1))
        false_positive = np.sum((pred_labels == 1) & (true_labels == 0))

        if true_positive + false_positive == 0:
            return 0.0

        precision = true_positive / (true_positive + false_positive)
        return precision
