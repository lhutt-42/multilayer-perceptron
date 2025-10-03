"""
This module contains the training logic for the model.
"""

import sys
from typing import Tuple

from .logger import logger
from .parser.file.dataset import load_dataset
from .model.losses import BinaryCrossEntropyLoss
from .model.metrics import AccuracyMetrics, PrecisionMetrics, RecallMetrics, F1ScoreMetrics
from .model.models import Model
from .model.preprocessing import binarize, normalize


# pylint: disable=duplicate-code
def predict(
    dataset_path: str,
    model_path: str,
) -> Tuple[float, float, float, float, float]:
    """
    Trains the model.

    Args:
        dataset_path (str): The path to the dataset.
        model_path (str): The path to the model.
    """

    df = load_dataset(dataset_path)

    if df.empty or len(df.columns) < 3:
        logger.error("The dataset does not match the excepted format.")
        sys.exit(1)

    # Drops the index column
    df = df.drop(columns=df.columns[0])

    x = normalize(df)
    y = binarize(df[df.columns[0]])

    model = Model.load(model_path)
    y_pred = model.predict(x)

    loss = BinaryCrossEntropyLoss.forward(y, y_pred)
    logger.info("Model Loss: %.4f", loss)

    accuracy = AccuracyMetrics.calculate_accuracy(y, y_pred)
    logger.info("Model Accuracy: %.4f", accuracy)

    precision = PrecisionMetrics.calculate_precision(y, y_pred)
    logger.info("Model Precision: %.4f", precision)

    recall = RecallMetrics.calculate_recall(y, y_pred)
    logger.info("Model Recall: %.4f", recall)

    f1_score = F1ScoreMetrics.calculate_f1_score(y, y_pred)
    logger.info("Model F1 Score: %.4f", f1_score)

    return loss, accuracy, precision, recall, f1_score
