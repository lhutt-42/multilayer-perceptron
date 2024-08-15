"""
Data class interface.
"""

from typing import List, Optional


class Data:
    """
    Interface for the data classes.
    """

    def __init__(
        self,
        train_values: Optional[List[float]] = None,
        test_values: Optional[List[float]] = None
    ) -> None:
        """
        Initializes the data class.

        Args:
            train_values (List[float]): The training values.
            test_values (List[float]): The validation values.
        """

        self.train_values: List[float] = train_values or []
        self.test_values: List[float] = test_values or []


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
