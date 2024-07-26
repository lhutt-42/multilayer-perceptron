"""
This module contains the layers used in the neural network.
"""

# pylint: disable=cyclic-import
from ..activations import Activation
from ..optimizers import Optimizer
from ..initializers import Initializer, ZeroInitializer

from .layer import Layer
from .dense import DenseLayer

__all__ = [
    'Layer',
    'DenseLayer',
]
