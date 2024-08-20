"""
This module provides functions to load metrics from a file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List
import json
import sys
import os

from . import logger

if TYPE_CHECKING:
    from . import Metrics


def _discover_metrics(directory: str, n: int) -> List[str]:
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

    except (FileNotFoundError, PermissionError, IsADirectoryError) as exception:
        logger.warning('No metrics available: %s', exception)
        return []

    # pylint: disable=broad-except
    except Exception as exception:
        logger.error('An error occurred while discovering metrics: %s', exception)
        sys.exit(1)


def _load_metric(path: str) -> Metrics:
    """
    Loads metrics from the provided file path.

    Args:
        path (str): The file path to load metrics from.

    Returns:
        Metrics: A Metrics object loaded from the file.
    """

    # pylint: disable=import-outside-toplevel
    from ....model.metrics import (
        Metrics,
        LossMetrics,
        AccuracyMetrics,
        PrecisionMetrics
    )

    try:
        logger.info('Loading metrics from `%s`', path)
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
        logger.error('An error occurred while loading metrics: %s', exception)
        sys.exit(1)

    # pylint: disable=broad-except
    except Exception as exception:
        logger.error('An error occurred while loading metrics: %s', exception)
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

    files = _discover_metrics(directory, n)

    return [
        _load_metric(file) for file in files
    ]
