"""
This module provides functions to save and load metrics to and from a file.
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


def save_metrics(metrics: Metrics, out_dir: str) -> None:
    """
    Saves the metrics to a file.

    Args:
        metrics (Metrics): The metrics to save.
        out_dir (str): The path to save the metrics.
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
        out_dir = os.path.join(out_dir, 'metrics')
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f'metrics_{timestamp}.json')

        with open(path, 'w', encoding='utf-8') as file:
            logging.debug('Saving the metrics to %s', path)
            json.dump(metrics_data, file)

    # pylint: disable=broad-except
    except Exception as exception:
        logging.error('An error occurred while saving the metrics: %s', exception)
        sys.exit(1)
