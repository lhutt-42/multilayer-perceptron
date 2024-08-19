"""
This module provides functions to save metrics to a file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import datetime
import logging
import json
import sys
import os

if TYPE_CHECKING:
    from . import Metrics


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
