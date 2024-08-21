"""
This module contains the training logic for the model.
"""

import sys
from typing import List

from sklearn.model_selection import train_test_split

from .logger import logger
from .parser.file.dataset import load_dataset
from .parser.file.model import load_new_model
from .model.losses import BinaryCrossEntropyLoss
from .model.metrics import Metrics
from .model.plots import Plot
from .model.preprocessing import binarize, normalize


def plot_metrics(
    metrics: List[Metrics],
    plot_multi: bool = False,
    plot_raw: bool = False
) -> None:
    """
    Plots the metrics.

    Args:
        metrics (List[Metrics]): The metrics to plot.
    """

    def _plot_individual(metric_name: str) -> None:
        plot = Plot(metric_name.capitalize())

        for metric in metrics:
            data = getattr(metric, metric_name, None)
            if data is None:
                continue
            plot.plot_data(data)

        plot.render(raw=plot_raw)


    if plot_multi is False:
        _plot_individual('loss')
        _plot_individual('accuracy')
        _plot_individual('precision')

        Plot.show()
        return


    if plot_multi:
        plot = Plot()

        for metric in metrics:
            plot.plot_data(metric.loss)
            plot.plot_data(metric.accuracy)
            plot.plot_data(metric.precision)

        plot.render(raw=plot_raw)

        Plot.show()


# pylint: disable=too-many-locals, disable=duplicate-code, too-many-arguments, unused-argument
def train(
    dataset_path: str,
    model_path: str,
    out_dir: str,
    plot_n: int,
    plot_multi: bool,
    plot_raw: bool
) -> None:
    """
    Trains the model.

    Args:
        dataset_path (str): The path to the dataset.
        model_path (str): The path to the model.
        out_dir (str): The output directory.
        plot_multi (bool): Plot multiple metrics.
        plot_raw (bool): Plot the raw data.
    """

    df = load_dataset(dataset_path)

    if df.empty or len(df.columns) < 3:
        logger.error('The dataset does not match the excepted format.')
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

    metrics = Metrics.load(out_dir, n=plot_n)
    metrics.insert(0, model.metrics)
    model.metrics.save(out_dir)

    plot_metrics(
        metrics,
        plot_multi=plot_multi,
        plot_raw=plot_raw
    )
