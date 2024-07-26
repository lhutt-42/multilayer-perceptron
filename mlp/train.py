"""
This module contains the training logic for the model.
"""

import sys
import logging
from typing import List

from mlp.parser.file import read_dataset
from mlp.model.preprocessing import binarize, normalize


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

    df = read_dataset(dataset_path)

    if df.empty or len(df.columns) < 3:
        logging.error('The dataset does not match the excepted format.')
        sys.exit(1)

    # Drops the index column
    df = df.drop(columns=df.columns[0])

    x = normalize(df)
    y = binarize(df[df.columns[0]])

    print(x, y)
