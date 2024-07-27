"""
Loss metrics class.
"""

from typing import List

class LossMetrics:
    """
    Class to store the loss metrics.
    """

    train_loss: List[float] = []
    val_loss: List[float] = []


    def add_train_loss(self, loss: float) -> None:
        """
        Adds a training loss value to the list.

        Args:
            loss (float): The training loss value.
        """

        self.train_loss.append(loss)


    def add_val_loss(self, loss: float) -> None:
        """
        Adds a validation loss value to the list.

        Args:
            loss (float): The validation loss value.
        """

        self.val_loss.append(loss)
