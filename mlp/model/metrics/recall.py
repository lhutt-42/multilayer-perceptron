"""
Recall metrics class.
"""

import numpy as np

from .data import Data


class RecallMetrics(Data):
    """
    Class to store the recall metrics.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the recall metrics class.
        """

        super().__init__(*args, **kwargs)
        self.name = "Recall"

    # pylint: disable=arguments-differ
    def add_train(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a training recall value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        recall = self.calculate_recall(y_true, y_pred)
        self.train_values.append(recall)

    # pylint: disable=arguments-differ
    def add_test(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a validation recall value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        recall = self.calculate_recall(y_true, y_pred)
        self.test_values.append(recall)

    @staticmethod
    def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the recall (sensitivity or true positive rate).

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            float: The recall.
        """

        true_labels = np.argmax(y_true, axis=1)
        pred_labels = np.argmax(y_pred, axis=1)

        true_positive = np.sum((pred_labels == 1) & (true_labels == 1))
        false_negative = np.sum((pred_labels == 0) & (true_labels == 1))

        if true_positive + false_negative == 0:
            return 0.0

        recall = true_positive / (true_positive + false_negative)
        return recall
