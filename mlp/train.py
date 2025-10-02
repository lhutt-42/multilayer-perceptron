"""
This module contains the training logic for the model.
"""

import sys
from typing import List

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from .logger import logger
from .parser.file.dataset import load_dataset
from .parser.file.model import load_new_model
from .model.losses import CategoricalCrossEntropyLoss
from .model.metrics import Metrics
from .model.plots import Plot
from .model.preprocessing import binarize, normalize


def plot_metrics(metrics: List[Metrics], plot_raw: bool = False) -> None:
    """
    Plots the metrics.

    Args:
        metrics (List[Metrics]): The metrics to plot.
    """

    metric_map = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 Score"),
    ]

    # Create a 3x2 grid (top row: 2 plots, bottom rows: 2 each, last empty)
    n_metrics = len(metric_map)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing

    try:
        fig.canvas.manager.set_window_title("Model Metrics")
    except AttributeError:
        pass

    fig.suptitle("Model Metrics", fontsize=14)

    for idx, (attribute, title) in enumerate(metric_map):
        plot = Plot(title)
        has_data = False

        for metric in metrics:
            data = getattr(metric, attribute, None)
            if data is None:
                continue
            plot.plot_data(data)
            has_data = True

        if has_data is False:
            axes[idx].set_visible(False)
            continue

        # Pass color index to use different palette for each metric
        plot.render(raw=plot_raw, ax=axes[idx], show_xlabel=True, color_index=idx)

    # Hide any unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    Plot.show()


# pylint: disable=too-many-locals, disable=duplicate-code, too-many-arguments, unused-argument
def train(
    dataset_path: str,
    model_path: str,
    out_dir: str,
    no_plot: bool,
    plot_n: int,
    plot_raw: bool,
) -> None:
    """
    Trains the model.

    Args:
        dataset_path (str): The path to the dataset.
        model_path (str): The path to the model.
        out_dir (str): The output directory.
        plot_raw (bool): Plot the raw data.
    """

    df = load_dataset(dataset_path)

    if df.empty or len(df.columns) < 3:
        logger.error("The dataset does not match the excepted format.")
        sys.exit(1)

    # Drops the index column
    df = df.drop(columns=df.columns[0])

    x = normalize(df)
    y = binarize(df[df.columns[0]])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    _, input_shape = x.shape
    _, output_shape = y.shape

    model, epochs, optimizer, early_stopping, batch_size = load_new_model(model_path)
    model.initialize(input_shape, output_shape, optimizer=optimizer)

    model.fit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=epochs,
        loss=CategoricalCrossEntropyLoss(),
        early_stopping=early_stopping,
        batch_size=batch_size,
    )

    model.save(out_dir)

    if no_plot is True:
        model.metrics.save(out_dir)
        return

    metrics = Metrics.load(out_dir, n=plot_n)
    metrics.insert(0, model.metrics)
    model.metrics.save(out_dir)

    plot_metrics(metrics, plot_raw=plot_raw)
