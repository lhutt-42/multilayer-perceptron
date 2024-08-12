"""
Data class interface.
"""

from typing import List


class Data:
    """
    Interface for the data classes.
    """

    def __init__(self) -> None:
        self.train_values: List[float] = []
        self.test_values: List[float] = []


    def add_train(self, **kwargs) -> None:
        """
        Adds a training value to the list.

        Args:
            kwargs: The arguments to pass to the data classes.
        """

        raise NotImplementedError


    def add_test(self, **kwargs) -> None:
        """
        Adds a validation value to the list.

        Args:
            kwargs: The arguments to pass to the data classes.
        """

        raise NotImplementedError
