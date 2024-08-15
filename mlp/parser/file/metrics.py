"""
This module provides functions to save and load metrics to and from a file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List
from datetime import datetime
import logging
import json
import sys
import os

if TYPE_CHECKING:
    from . import Metrics


def discover_metrics(directory: str, n: int) -> List[str]:
    """
    Discovers the n latest metrics files in the specified directory.

    Args:
        directory (str): The directory path where metrics files are stored.
        n (int): The number of latest metrics files to discover.

    Returns:
        List[str]: A list of file paths to the n latest metrics files.
    """

    try:
        files = [
            file for file in os.listdir(directory) if file.startswith('metrics_')
        ]

        files.sort(reverse=True)

        return [
            os.path.join(directory, file) for file in files[:n]
        ]

    # pylint: disable=broad-except
    except Exception as exception:
        logging.error('An error occurred while discovering metrics: %s', exception)
        sys.exit(1)


def load_metric(path: str) -> Metrics:
    """
    Loads metrics from the provided file path.

    Args:
        path (str): The file path to load metrics from.

    Returns:
        Metrics: A Metrics object loaded from the file.
    """

    # pylint: disable=import-outside-toplevel
    from . import Metrics, LossMetrics, AccuracyMetrics, PrecisionMetrics

    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            metrics = Metrics(
                loss=LossMetrics(
                    train_values=data['loss']['train'],
                    test_values=data['loss']['test']
                ),
                accuracy=AccuracyMetrics(
                    train_values=data['accuracy']['train'],
                    test_values=data['accuracy']['test']
                ),
                precision=PrecisionMetrics(
                    train_values=data['precision']['train'],
                    test_values=data['precision']['test']
                )
            )

            return metrics

    except (FileNotFoundError, PermissionError, IsADirectoryError) as exception:
        logging.error('An error occurred while loading metrics: %s', exception)
        sys.exit(1)

    # pylint: disable=broad-except
    except Exception as exception:
        logging.error('An error occurred while loading metrics: %s', exception)
        sys.exit(1)


def load_metrics(directory: str, n: int) -> List[Metrics]:
    """
    Loads metrics from the provided directory.

    Args:
        directory (str): The directory path to load metrics from.
        n (int): The number of latest metrics files to load.

    Returns:
        List[Metrics]: A list of Metrics objects loaded from the directory.
    """

    files = discover_metrics(directory, n)

    return [
        load_metric(file) for file in files
    ]


def save_metrics(metrics: Metrics, directory: str) -> None:
    """
    Saves the metrics to a file.

    Args:
        metrics (Metrics): The metrics to save.
        directory (str): The path to save the metrics.
    """

    metrics_data = {
        'loss': {
            'train': metrics.loss.train_values,
            'test': metrics.loss.test_values
        },
        'accuracy': {
            'train': metrics.accuracy.train_values,
            'test': metrics.accuracy.test_values
        },
        'precision': {
            'train': metrics.precision.train_values,
            'test': metrics.precision.test_values
        }
    }

    try:
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f'metrics_{timestamp}.json')

        with open(path, 'w', encoding='utf-8') as file:
            logging.debug('Saving the metrics to %s', path)
            json.dump(metrics_data, file)

    except (PermissionError, IsADirectoryError) as exception:
        logging.error('An error occurred while saving the metrics: %s', exception)
        sys.exit(1)

    # pylint: disable=broad-except
    except Exception as exception:
        logging.error('An error occurred while saving the metrics: %s', exception)
        sys.exit(1)
