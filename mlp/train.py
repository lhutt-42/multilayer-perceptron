"""
This module contains the training logic for the model.
"""

import sys
import logging
from typing import List, Type

from sklearn.model_selection import train_test_split

# pylint: disable=unused-import
from .parser.file import read_dataset
from .model.models import MiniBatchModel, BatchModel
from .model.preprocessing import binarize, normalize
from .model.losses import Loss
from .model.layers import DenseLayer
from .model.activations import SigmoidActivation, SoftmaxActivation, ReluActivation
from .model.optimizers import GradientDescentOptimizer, AdamOptimizer
from .model.initializers import RandomInitializer, ZeroInitializer, HeInitializer, XavierInitializer
from .model.regularizers import L1Regularizer, L2Regularizer
from .model.metrics import Metrics, LossMetrics, AccuracyMetrics


# pylint: disable=too-many-arguments, too-many-locals, unused-argument
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

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    model = BatchModel()

    model.add([
        DenseLayer(
            layer_size=input_shape,
            activation=SigmoidActivation(),
            weight_initializer=XavierInitializer(),
            bias_initializer=ZeroInitializer(),
            regularizer=L1Regularizer()
        ),
        DenseLayer(
            layer_size=18,
            activation=SigmoidActivation(),
            weight_initializer=XavierInitializer(),
            bias_initializer=ZeroInitializer(),
            regularizer=L2Regularizer()
        ),
        DenseLayer(
            layer_size=12,
            activation=SigmoidActivation(),
            weight_initializer=XavierInitializer(),
            bias_initializer=ZeroInitializer(),
            regularizer=L2Regularizer()
        ),
        DenseLayer(
            layer_size=6,
            activation=SigmoidActivation(),
            weight_initializer=XavierInitializer(),
            bias_initializer=ZeroInitializer(),
            regularizer=L2Regularizer()
        ),
        DenseLayer(
            layer_size=output_shape,
            activation=SoftmaxActivation(),
            weight_initializer=HeInitializer(),
            bias_initializer=ZeroInitializer(),
            regularizer=None
        )
    ])

    try:
        model.fit(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=epochs,
            loss=loss(),
            optimizer=AdamOptimizer(learning_rate),
            batch_size=batch_size
        )
    except KeyboardInterrupt:
        logging.error('Training interrupted.')
        sys.exit(1)
