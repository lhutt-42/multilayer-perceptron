"""
F1 Score metrics class.
"""

import numpy as np

from .data import Data
from .precision import PrecisionMetrics
from .recall import RecallMetrics


class F1ScoreMetrics(Data):
    """
    Class to store the F1 Score metrics.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the F1 Score metrics class.
        """

        super().__init__(*args, **kwargs)
        self.name = "F1 Score"

    # pylint: disable=arguments-differ
    def add_train(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a training F1 Score value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        f1_score = self.calculate_f1_score(y_true, y_pred)
        self.train_values.append(f1_score)

    # pylint: disable=arguments-differ
    def add_test(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> None:
        """
        Adds a validation F1 Score value to the list.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """

        f1_score = self.calculate_f1_score(y_true, y_pred)
        self.test_values.append(f1_score)

    @staticmethod
    def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the F1 Score (harmonic mean of precision and recall).

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            float: The F1 Score.
        """

        precision = PrecisionMetrics.calculate_precision(y_true, y_pred)
        recall = RecallMetrics.calculate_recall(y_true, y_pred)

        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
