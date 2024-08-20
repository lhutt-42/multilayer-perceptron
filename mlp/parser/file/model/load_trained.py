"""
This module provides functions to load the model from a json file.
"""

from __future__ import annotations
import sys
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from . import logger
from . import (
    load_activation,
    load_initializer,
    load_layer,
    load_model,
    load_optimizer,
    load_regularizer,
)

if TYPE_CHECKING:
    from . import (
        Model, Layer
    )


# pylint: disable=too-many-locals
def _load_layer(data: Any) -> Layer:
    """
    Loads a layer from a dictionary.

    Args:
        data (Any): The dictionary containing the layer data.

    Returns:
        Layer: The loaded layer.
    """

    layer_size = data['layer_size']
    if layer_size is None or not isinstance(layer_size, int) or layer_size <= 0:
        raise ValueError('Layer size is required.')

    weights = data['weights']
    if weights is None or not isinstance(weights, list):
        raise ValueError('Weights are required.')

    biases = data['biases']
    if biases is None or not isinstance(biases, list):
        raise ValueError('Biases are required.')

    return load_layer(
        data['type'],
        {
            'layer_size': layer_size,
            'activation': load_activation(
                data['activation'],
                {}
            ),
            'weight_initializer': load_initializer(
                data['weight_initializer'],
                {}
            ),
            'bias_initializer': load_initializer(
                data['bias_initializer'],
                {}
            ),
            'optimizer': load_optimizer(
                data['optimizer'],
                data['optimizer'] and {}
            ),
            'regularizer': load_regularizer(
                data['regularizer'],
                data['regularizer'] and {}
            ),
            'gradient_clipping': data['gradient_clipping'],
            'weights': np.array(weights),
            'biases': np.array(biases)
        }
    )


def load_trained_model(path: str) -> Model:
    """
    Loads the model from a json file.

    Args:
        path (str): The path to the model.

    Returns:
        Model: The loaded model.
    """

    try:
        logger.info('Loading trained model from `%s`', path)
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if data is None:
                raise ValueError('Model data is required.')

            model_type = data['type']
            if model_type is None:
                raise ValueError('Model type is required.')

            model = load_model(
                model_type,
                data
            )

            model.add([
                _load_layer(layer_data) for layer_data in data['layers']
            ])

            logger.info('Model loaded from %s', path)
            return model

    except (ValueError, AttributeError, ImportError) as exception:
        logger.error('An error occurred while loading the model: %s', exception)
        sys.exit(1)

    except (FileNotFoundError, PermissionError, IsADirectoryError) as exception:
        logger.error('An error occurred while loading the model: %s', exception)
        sys.exit(1)

    #pylint: disable=broad-except
    except Exception as exception:
        logger.error('An error occurred while loading the model: %s', exception)
        sys.exit(1)
