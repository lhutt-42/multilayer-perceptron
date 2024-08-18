"""
This module contains the functions to save and load models.
"""

from ...modules import (
    load_activation,
    load_initializer,
    load_layer,
    load_model,
    load_optimizer,
    load_regularizer,
    load_early_stopping
)

# pylint: disable=wrong-import-order
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from ....model.activations import Activation
    from ....model.initializers import Initializer
    from ....model.layers import Layer
    from ....model.models import Model
    from ....model.optimizers import Optimizer
    from ....model.regularizers import Regularizer
    from ....model.training import EarlyStopping

# pylint: disable=wrong-import-position
from .load_new import load_new_model
from .load_trained import load_trained_model
from .save import save_model

__all__ = [
    'load_new_model',
    'load_trained_model',
    'save_model'
]
