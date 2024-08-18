"""
This module provides functions to load modules.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from importlib import import_module

if TYPE_CHECKING:
    from . import (
        Activation,
        Initializer,
        Layer,
        Model,
        Optimizer,
        Regularizer,
        EarlyStopping,
    )


def _load(module: str, name: str, data: Optional[Any]) -> Any | None:
    if data is None:
        return None

    _module = import_module(module, package='mlp')
    _class = getattr(_module, name)

    return _class(
        **data.__dict__ if hasattr(data, '__dict__') else data
    )


def load_activation(name: str, data: Optional[Any]) -> Activation | None:
    """
    Loads activation data and returns an activation function.

    Args:
        name (str): The name of the activation function.
        data (Optional[Any]): The activation data.

    Returns:
        Activation: The activation function
    """

    return _load('.model.activations', name, data)


def load_initializer(name: str, data: Optional[Any]) -> Initializer | None:
    """
    Loads initializer data and returns an initializer.

    Args:
        name (str): The name of the initializer.
        data (Optional[Any]): The initializer data.

    Returns:
        Initializer: The initializer.
    """

    return _load('.model.initializers', name, data)


def load_layer(name: str, data: Optional[Any]) -> Layer | None:
    """
    Loads layer data and returns a layer.

    Args:
        name (str): The name of the layer.
        data (Optional[Any]): The layer data.

    Returns:
        Layer: The layer.
    """

    return _load('.model.layers', name, data)


def load_model(name: str, data: Optional[Any]) -> Model | None:
    """
    Loads model data and returns a model.

    Args:
        name (str): The name of the model.
        data (Optional[Any]): The model data.

    Returns:
        Model: The model.
    """

    return _load('.model.models', name, data)


def load_optimizer(name: str, data: Optional[Any]) -> Optimizer | None:
    """
    Loads optimizer data and returns an optimizer.

    Args:
        name (str): The name of the optimizer.
        data (Optional[Any]): The optimizer data.

    Returns:
        Optimizer: The optimizer.
    """

    return _load('.model.optimizers', name, data)


def load_regularizer(name: str, data: Optional[Any]) -> Regularizer | None:
    """
    Loads regularizer data and returns a regularizer.

    Args:
        name (str): The name of the regularizer.
        data (Optional[Any]): The regularizer data.

    Returns:
        Regularizer: The regularizer.
    """

    return _load('.model.regularizers', name, data)


def load_early_stopping(data: Optional[Any]) -> EarlyStopping | None:
    """
    Loads early stopping data and returns an early stopping module.

    Args:
        data (Optional[Any]): The early stopping data.

    Returns:
        EarlyStopping: The early stopping module.
    """

    return _load('.model.training', 'EarlyStopping', data)
