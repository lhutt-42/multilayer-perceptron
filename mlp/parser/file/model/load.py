"""
This module provides functions to load the model from a file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
from importlib import import_module
import logging
import json
import sys

import numpy as np

if TYPE_CHECKING:
    from . import Model, Layer


# pylint: disable=too-many-locals
def _load_layer(layer_data: Dict[str, Any]) -> Layer:
    """
    Loads a layer from a dictionary.

    Args:
        layer_data (Dict[str, Any]): The dictionary containing the layer data.

    Returns:
        Layer: The loaded layer.
    """

    layer_size = layer_data['layer_size']
    if layer_size is None or not isinstance(layer_size, int):
        raise ValueError('Layer size is required.')

    if layer_data['type'] is None:
        raise ValueError('Layer type is required.')
    layer_module = import_module('.model.layers', package='mlp')
    layer = getattr(layer_module, layer_data['type'])

    if layer_data['activation'] is None:
        raise ValueError('Activation function is required.')
    activation_module = import_module('.model.activations', package='mlp')
    activation = getattr(activation_module, layer_data['activation'])

    if layer_data['weight_initializer'] is None:
        raise ValueError('Weight initializer is required.')
    weight_initializer_module = import_module('.model.initializers', package='mlp')
    weight_initializer = getattr(weight_initializer_module, layer_data['weight_initializer'])

    if layer_data['bias_initializer'] is None:
        raise ValueError('Bias initializer is required.')
    bias_initializer_module = import_module('.model.initializers', package='mlp')
    bias_initializer = getattr(bias_initializer_module, layer_data['bias_initializer'])

    optimizer_module = import_module('.model.optimizers', package='mlp')
    optimizer = layer_data['optimizer'] \
        and getattr(optimizer_module, layer_data['optimizer'])

    regularizer_module = import_module('.model.regularizers', package='mlp')
    regularizer = layer_data['regularizer'] \
        and getattr(regularizer_module, layer_data['regularizer'])

    gradient_clipping = layer_data['gradient_clipping']

    weights = layer_data['weights']
    if weights is None or not isinstance(weights, list):
        raise ValueError('Weights are required.')

    biases = layer_data['biases']
    if biases is None or not isinstance(biases, list):
        raise ValueError('Biases are required.')

    return layer(
        layer_size=layer_size,
        activation=activation(),
        weight_initializer=weight_initializer(),
        bias_initializer=bias_initializer(),
        optimizer=optimizer and optimizer(),
        regularizer=regularizer and regularizer(),
        gradient_clipping=gradient_clipping,
        weights=np.array(weights),
        biases=np.array(biases)
    )


def load_model(path: str) -> Model:
    """
    Loads the model from a file.

    Args:
        path (str): The path to the model.

    Returns:
        Model: The loaded model.
    """

    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            model_type = data['type']
            if model_type is None:
                raise ValueError('Model type is required.')

            model_module = import_module('.model.models', package='mlp')
            model_class = getattr(model_module, model_type)

            model = model_class()
            model.add([
                _load_layer(layer_data) for layer_data in data['layers']
            ])

            return model

    except (ValueError, AttributeError, ImportError) as exception:
        logging.error('An error occurred while loading the model: %s', exception)
        sys.exit(1)

    except (FileNotFoundError, PermissionError, IsADirectoryError) as exception:
        logging.error('An error occurred while loading the model: %s', exception)
        sys.exit(1)

    #pylint: disable=broad-except
    except Exception as exception:
        logging.error('An error occurred while loading the model: %s', exception)
        sys.exit(1)
