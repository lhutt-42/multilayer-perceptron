"""
This module contains the loss functions used in the neural network.
"""

from .loss import Loss
from .binary_cross_entropy import BinaryCrossEntropyLoss

__all__ = [
    'Loss',
    'BinaryCrossEntropyLoss'
]
