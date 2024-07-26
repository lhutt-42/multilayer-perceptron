"""
This module contains the training logic for the model.
"""

import sys
import logging
from typing import List, Type

from .parser.file import read_dataset

# pylint: disable=unused-import
from .model.models import MiniBatchModel, BatchModel
from .model.preprocessing import binarize, normalize
from .model.losses import Loss
from .model.layers import DenseLayer
from .model.activations import SigmoidActivation, SoftmaxActivation, ReluActivation
from .model.optimizers import GradientDescentOptimizer
from .model.initializers import RandomInitializer, ZeroInitializer
from .model.regularizers import L1Regularizer, L2Regularizer

# pylint: disable=too-many-arguments, unused-argument
def train(
    dataset_path: str,
    layers: List[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss: Type[Loss],
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

    _, input_shape = x.shape
    _, output_shape = y.shape

    model = BatchModel()

    model.add([
        DenseLayer(
            input_shape,
            SigmoidActivation(),
            GradientDescentOptimizer(learning_rate),
            weight_initializer=RandomInitializer(),
            bias_initializer=RandomInitializer(),
            regularizer=L1Regularizer()
        ),
        DenseLayer(
            output_shape,
            SoftmaxActivation(),
            GradientDescentOptimizer(learning_rate),
            weight_initializer=RandomInitializer(),
            bias_initializer=RandomInitializer(),
            regularizer=L1Regularizer()
        )
    ])

    try:
        model.fit(x, y, loss=loss(), epochs=epochs, batch_size=batch_size)
    except KeyboardInterrupt:
        logging.error('Training interrupted.')
        sys.exit(1)
