"""
This module contains the training logic for the model.
"""

import sys
import logging
from typing import List

from sklearn.model_selection import train_test_split

# pylint: disable=unused-import
from .parser.file.dataset import load_dataset
from .parser.file.model import load_new_model

from .model.activations import SigmoidActivation, SoftmaxActivation, ReluActivation
from .model.initializers import RandomInitializer, ZeroInitializer, HeInitializer, XavierInitializer
from .model.layers import DenseLayer
from .model.losses import BinaryCrossEntropyLoss
from .model.metrics import Metrics
from .model.models import MiniBatchModel, BatchModel
from .model.optimizers import GradientDescentOptimizer, AdamOptimizer
from .model.plots import Plot
from .model.preprocessing import binarize, normalize
from .model.regularizers import L1Regularizer, L2Regularizer
from .model.training import EarlyStopping


def plot_metrics(metrics: List[Metrics]) -> None:
    """
    Plots the metrics.

    Args:
        metrics (List[Metrics]): The metrics to plot.
    """

    loss_plot = Plot('Loss')
    for metric in metrics:
        loss_plot.plot_data(metric.loss)
    loss_plot.render()

    accuracy_plot = Plot('Accuracy')
    for metric in metrics:
        accuracy_plot.plot_data(metric.accuracy)
    accuracy_plot.render()

    precision_plot = Plot('Precision')
    for metric in metrics:
        precision_plot.plot_data(metric.precision)
    precision_plot.render()

    Plot.show()


# pylint: disable=too-many-locals, disable=duplicate-code
def train(
    dataset_path: str,
    model_path: str,
    out_dir: str
) -> None:
    """
    Trains the model.

    Args:
        dataset_path (str): The path to the dataset.
        model_path (str): The path to the model.
        out_dir (str): The output directory.
    """

    df = load_dataset(dataset_path)

    if df.empty or len(df.columns) < 3:
        logging.error('The dataset does not match the excepted format.')
        sys.exit(1)

    # Drops the index column
    df = df.drop(columns=df.columns[0])

    x = normalize(df)
    y = binarize(df[df.columns[0]])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    _, input_shape = x.shape
    _, output_shape = y.shape

    model, epochs, optimizer, early_stopping, batch_size = load_new_model(model_path)
    model.initialize(
        input_shape,
        output_shape,
        optimizer=optimizer
    )

    model.fit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
        loss=BinaryCrossEntropyLoss(),
        early_stopping=early_stopping,
        batch_size=batch_size
    )

    model.save(out_dir)

    metrics = Metrics.load(out_dir, n=2)
    metrics.insert(0, model.metrics)
    model.metrics.save(out_dir)

    plot_metrics(metrics)
