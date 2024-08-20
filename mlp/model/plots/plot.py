"""
This module contains the plotting logic.
"""

import sys
from typing import List, Dict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

from . import logger
from . import Data


class Plot:
    """
    A class to plot the metrics of a model.
    """

    def __init__(self, title: str = 'Metrics') -> None:
        """
        Initializes the plotter.

        Args:
            title (str): The title of the plot.
        """

        self.title = title
        self.data: List[Dict] = []

        self.train_palette: List[str] = []
        self.test_palette: List[str] = []


    def plot_data(self, data: Data) -> None:
        """
        Stores the data for plotting later.

        Args:
            data (Data): The data to plot.
        """

        sigma = len(data.train_values) / 500
        smoothed_train_values = gaussian_filter1d(data.train_values, sigma=sigma)
        smoothed_test_values = gaussian_filter1d(data.test_values, sigma=sigma)

        self.data.append({
            'train': smoothed_train_values,
            'test': smoothed_test_values,
        })


    def render(self) -> None:
        """
        Renders the plot.
        """

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.canvas.manager.set_window_title(self.title)

        self.train_palette = self.fading_palette(len(self.data), '#FF0000')
        self.test_palette = self.fading_palette(len(self.data), '#0000FF')

        for (i, data) in enumerate(self.data):
            train_color = self.train_palette[i % len(self.train_palette)]
            ax.plot(
                data['train'],
                label=f'{self.title} Train' if i == 0 else None,
                color=train_color,
                alpha=min(1.0, 1.0 - i / len(self.data) + 0.2),
                linestyle=(0, (1, 2)) if i != 0 else None
            )

            test_color = self.test_palette[i % len(self.test_palette)]
            ax.plot(
                data['test'],
                label=f'{self.title} Test' if i == 0 else None,
                color=test_color,
                alpha=min(1.0, 1.0 - i / len(self.data) + 0.2),
                linestyle=(0, (1, 2)) if i != 0 else None
            )

        ax.set_xlabel('Epochs')
        ax.set_ylabel(self.title)
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
            'fading',
            [base, '#FFFFFF'],
            N=n + 1
        )

        return [
            color_map(i) for i in range(n)
        ]


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
            logger.error('Could not render the plot: %s', e)
            sys.exit(1)
