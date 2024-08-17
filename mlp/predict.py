"""
This module contains the training logic for the model.
"""

import sys
import logging

from .parser.file.dataset import read_dataset
from .model.losses import BinaryCrossEntropyLoss
from .model.metrics import AccuracyMetrics
from .model.models import Model
from .model.preprocessing import binarize, normalize


# pylint: disable=duplicate-code
def predict(
    dataset_path: str,
    model_path: str,
) -> None:
    """
    Trains the model.

    Args:
        dataset_path (str): The path to the dataset.
        model_path (str): The path to the model.
    """

    df = read_dataset(dataset_path)

    if df.empty or len(df.columns) < 3:
        logging.error('The dataset does not match the excepted format.')
        sys.exit(1)

    # Drops the index column
    df = df.drop(columns=df.columns[0])

    x = normalize(df)
    y = binarize(df[df.columns[0]])

    model = Model.load(model_path)
    y_pred = model.predict(x)

    accuracy = AccuracyMetrics.calculate_accuracy(y, y_pred)
    logging.info('Model Accuracy: %.4f', accuracy)

    bce = BinaryCrossEntropyLoss()
    loss = bce.forward(y, y_pred)
    logging.info('Model Loss: %.4f', loss)
