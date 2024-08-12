"""
Class to store the metrics.
"""

import logging

from .loss import LossMetrics
from .accuracy import AccuracyMetrics


class Metrics:
    """
    Class to store the metrics.
    """

    def __init__(self) -> None:
        self.loss = LossMetrics()
        self.accuracy = AccuracyMetrics()


    def add_train(self, **kwargs) -> None:
        """
        Adds a training value to the list.

        Args:
            kwargs: The arguments to pass to the data classes.
        """

        self.loss.add_train(**kwargs)
        self.accuracy.add_train(**kwargs)


    def add_test(self, **kwargs) -> None:
        """
        Adds a validation value to the list.

        Args:
            kwargs: The arguments to pass to the data classes.
        """

        self.loss.add_test(**kwargs)
        self.accuracy.add_test(**kwargs)


    def log(self, epoch: int) -> None:
        """
        Logs the metrics.

        Args:
            epoch (int): The epoch.
        """

        logging.info(
            'epoch %6d - '
            'train loss: %4f - '
            'test loss: %4f - '
            'train accuracy: %4f - '
            'test accuracy: %4f',
            epoch,
            self.loss.train_values[-1],
            self.loss.test_values[-1],
            self.accuracy.train_values[-1],
            self.accuracy.test_values[-1]
        )
