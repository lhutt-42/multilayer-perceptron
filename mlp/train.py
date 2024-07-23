"""
This module contains the training logic for the model.
"""

from typing import List

from mlp.parser.file import read_dataset


# pylint: disable=too-many-arguments, unused-argument
def train_model(
    dataset_path: str,
    layers: List[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss: str,
    out_dir: str
) -> None:
    """
    Trains the model.

    Args:
        dataset_path (str): The path to the dataset.
        layers (List[int]): The number of neurons in each layer.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size used during training.
        learning_rate (float): The learning rate of the optimizer.
        loss (str): The loss function to use.
        out_dir (str): The output directory.
    """

    _ = read_dataset(dataset_path)
