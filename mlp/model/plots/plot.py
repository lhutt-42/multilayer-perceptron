"""
This module contains the plotting logic.
"""

import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

from . import logger
from . import Data


class Plot:
    """
    A class to plot the metrics of a model.
    """

    PALETTES = [
        {"train": "#FFA500", "test": "#FF6347"},
        {"train": "#1E90FF", "test": "#00BFFF"},
        {"train": "#32CD32", "test": "#3CB371"},
        {"train": "#9370DB", "test": "#BA55D3"},
        {"train": "#FF69B4", "test": "#FFB6C1"},
    ]

    @dataclass
    class Metrics:
        """
        A class to store the metrics.
        """

        name: str

        train_raw: List[float]
        test_raw: List[float]

        train_smoothed: List[float]
        test_smoothed: List[float]

    @dataclass
    class Palette:
        """
        A class to store the palette.
        """

        train: List[str]
        test: List[str]

    def __init__(self, title: str = "Metrics") -> None:
        """
        Initializes the plotter.

        Args:
            title (str): The title of the plot.
        """

        self.title = title
        self.metrics: Dict[str, List[Plot.Metrics]] = {}
        self.palettes: Dict[str, Plot.Palette] = {}

    def plot_data(self, metric: Data) -> None:
        """
        Stores the data for plotting later.

        Args:
            data (Data): The data to plot.
        """

        sigma = len(metric.train_values) / 100
        metric = Plot.Metrics(
            name=metric.name,
            train_raw=metric.train_values,
            test_raw=metric.test_values,
            train_smoothed=[],
            test_smoothed=[],
        )

        try:
            metric.train_smoothed = gaussian_filter1d(metric.train_raw, sigma=sigma)
            metric.test_smoothed = gaussian_filter1d(metric.test_raw, sigma=sigma)
        except ZeroDivisionError:
            logger.warning("An error occurred while smoothing the data.")
            return

        if self.metrics.get(metric.name) is None:
            self.metrics[metric.name] = []

        self.metrics[metric.name].append(metric)

    def render(
        self,
        raw: bool = False,
        ax: Optional[plt.Axes] = None,
        show_xlabel: bool = True,
        color_index: int = 0,
    ) -> None:
        """
        Renders the plot.

        Args:
            raw (bool): Whether to render the raw data.
            ax (Optional[plt.Axes]): Axis to render on. Creates a new figure when None.
            show_xlabel (bool): Whether to display the X-axis label.
            color_index (int): Index to select color palette for this metric.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        try:
            fig.canvas.manager.set_window_title(self.title)
        except AttributeError:
            pass

        ax.set_title(self.title)

        for i, (name, metrics) in enumerate(self.metrics.items()):
            # Use color_index to pick palette, wrapping around if needed
            palette_idx = color_index % len(Plot.PALETTES)
            self.palettes[name] = Plot.Palette(
                train=Plot.fading_palette(
                    len(metrics), Plot.PALETTES[palette_idx]["train"]
                ),
                test=Plot.fading_palette(
                    len(metrics), Plot.PALETTES[palette_idx]["test"]
                ),
            )

        for name, metrics in self.metrics.items():
            for i, metric in enumerate(metrics):
                train_color = self.palettes[name].train[
                    i % len(self.palettes[name].train)
                ]
                ax.plot(
                    raw and metric.train_raw or metric.train_smoothed,
                    label=f"{metric.name} Train" if i == 0 else None,
                    color=train_color,
                    alpha=min(1.0, 1.0 - i / len(metrics) + 0.2),
                    linestyle=(0, (1, 2)) if i != 0 else None,
                )

                test_color = self.palettes[name].test[i % len(self.palettes[name].test)]
                ax.plot(
                    raw and metric.test_raw or metric.test_smoothed,
                    label=f"{metric.name} Test" if i == 0 else None,
                    color=test_color,
                    alpha=min(1.0, 1.0 - i / len(metrics) + 0.2),
                    linestyle=(0, (1, 2)) if i != 0 else None,
                )

        if len(self.metrics) == 1:
            ax.set_ylabel(self.title)
        if show_xlabel:
            ax.set_xlabel("Epochs")
        else:
            ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)

    @staticmethod
    def fading_palette(n: int, base: str) -> List[str]:
        """
        Generates a fading palette of n colors.

        Args:
            n (int): The number of colors to generate.
            base (str): The base color of the palette.

        Returns:
            List[str]: The list of colors
        """

        color_map = LinearSegmentedColormap.from_list(
            "fading", [base, "#FFFFFF"], N=n + 1
        )

        return [color_map(i) for i in range(n)]

    @staticmethod
    def show() -> None:
        """
        Shows the plots.
        """

        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        except (RuntimeError, ValueError) as e:
            logger.error("Could not render the plot: %s", e)
            sys.exit(1)
