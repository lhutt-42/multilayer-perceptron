"""
This module provides functions to load modules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from ...model.activations import Activation
    from ...model.initializers import Initializer
    from ...model.layers import Layer
    from ...model.models import Model
    from ...model.optimizers import Optimizer
    from ...model.regularizers import Regularizer
    from ...model.training import EarlyStopping

# pylint: disable=wrong-import-position
from .module import (
    load_activation,
    load_initializer,
    load_layer,
    load_model,
    load_optimizer,
    load_regularizer,
    load_early_stopping
)

__all__ = [
    'load_activation',
    'load_initializer',
    'load_layer',
    'load_model',
    'load_optimizer',
    'load_regularizer',
    'load_early_stopping'
]
