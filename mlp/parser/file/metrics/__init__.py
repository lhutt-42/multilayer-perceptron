"""
This module contains the functions to save and load metrics.
"""

# pylint: disable=cyclic-import
from .. import logger

# pylint: disable=wrong-import-order
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from ....model.metrics import Metrics

# pylint: disable=wrong-import-position
from .load import load_metrics
from .save import save_metrics

__all__ = [
    'load_metrics',
    'save_metrics'
]
