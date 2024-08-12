"""
Loss metrics class.
"""

from .data import Data


class LossMetrics(Data):
    """
    Class to store the loss metrics.
    """

    # pylint: disable=arguments-differ
    def add_train(self, loss: float, **kwargs) -> None:
        """
        Adds a training loss value to the list.

        Args:
            loss (float): The training loss value.
        """

        self.train_values.append(loss)


    # pylint: disable=arguments-differ
    def add_test(self, loss: float, **kwargs) -> None:
        """
        Adds a validation loss value to the list.

        Args:
            loss (float): The validation loss value.
        """

        self.test_values.append(loss)
