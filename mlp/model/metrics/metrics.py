"""
Class to store the metrics.
"""

from __future__ import annotations
from typing import List, Optional
import os

from . import logger
from .loss import LossMetrics
from .accuracy import AccuracyMetrics
from .precision import PrecisionMetrics
from . import save_metrics, load_metrics


class Metrics:
    """
    Class to store the metrics.
    """

    def __init__(
        self,
        loss: Optional[LossMetrics] = None,
        accuracy: Optional[AccuracyMetrics] = None,
        precision: Optional[PrecisionMetrics] = None
    ) -> None:
        """
        Initializes the metrics.

        Args:
            loss (LossMetrics): The loss metrics.
            accuracy (AccuracyMetrics): The accuracy metrics.
            precision (PrecisionMetrics): The precision metrics.
        """

        self.loss = loss or LossMetrics()
        self.accuracy = accuracy or AccuracyMetrics()
        self.precision = precision or PrecisionMetrics()


    def add_train(self, **kwargs) -> None:
        """
        Adds a training value to the list.

        Args:
            kwargs: The arguments to pass to the data classes.
        """

        self.loss.add_train(**kwargs)
        self.accuracy.add_train(**kwargs)
        self.precision.add_train(**kwargs)


    def add_test(self, **kwargs) -> None:
        """
        Adds a validation value to the list.

        Args:
            kwargs: The arguments to pass to the data classes.
        """

        self.loss.add_test(**kwargs)
        self.accuracy.add_test(**kwargs)
        self.precision.add_test(**kwargs)


    def log(self, epoch: int) -> None:
        """
        Logs the metrics.

        Args:
            epoch (int): The epoch.
        """

        logger.info(
            'epoch %6d - '
            'train loss: %4f - '
            'test loss: %4f - '
            'train accuracy: %4f - '
            'test accuracy: %4f - ',
            epoch,
            self.loss.train_values[-1],
            self.loss.test_values[-1],
            self.accuracy.train_values[-1],
            self.accuracy.test_values[-1],
        )


    def save(self, directory: str) -> None:
        """
        Saves the metrics to a file.

        Args:
            directory (str): The path to save the metrics.
        """

        save_metrics(self, os.path.join(directory, 'metrics'))


    @staticmethod
    def load(directory: str, n: int = 2) -> List[Metrics]:
        """
        Loads the metrics from a directory.

        Args:
            directory (str): The path to load the metrics.
            n (int): The number of metrics to load.

        Returns:
            List[Metrics]: The list of metrics.
        """

        return load_metrics(os.path.join(directory, 'metrics'), n)
