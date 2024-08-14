"""
This module contains the training utils for the model.
"""

# pylint: disable=cyclic-import
from ..layers import Layer

from .early_stopping import EarlyStopping

__all__ = [
    'EarlyStopping',
]
