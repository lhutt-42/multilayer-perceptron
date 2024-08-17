"""
This module provides functions to save the model to a file.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict
import logging
import json
import sys
import os

if TYPE_CHECKING:
    from . import Model, Layer


def _type_to_str(argument: Any) -> str | None:
    return type(argument).__name__ if argument else None


def _save_layer(layer: Layer) -> Dict:
    if layer.weights is None or layer.biases is None:
        raise ValueError('The weights and biases must be initialized before saving the model.')

    return {
        'type': _type_to_str(layer),

        'layer_size': layer.layer_size,
        'activation': _type_to_str(layer.activation),

        'weight_initializer': _type_to_str(layer.weight_initializer),
        'bias_initializer': _type_to_str(layer.bias_initializer),

        'optimizer': _type_to_str(layer.optimizer),
        'regularizer': _type_to_str(layer.regularizer),
        'gradient_clipping': layer.gradient_clipping,

        'weights': layer.weights.tolist(),
        'biases': layer.biases.tolist()
    }


def save_model(model: Model, directory: str) -> None:
    """
    Saves the model to a file.

    Args:
        model (Model): The model to save.
        directory (str): The path to save the model.
    """

    model_data = {
        'type': _type_to_str(model),
        'layers': [
            _save_layer(layer) for layer in model.layers
        ],
    }

    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'model.json')

        with open(path, 'w', encoding='utf-8') as file:
            logging.debug('Saving the model to %s', path)
            json.dump(model_data, file)

    except (PermissionError, IsADirectoryError) as exception:
        logging.error('An error occurred while saving the model: %s', exception)
        sys.exit(1)

    # pylint: disable=broad-except
    except Exception as exception:
        logging.error('An error occurred while saving the model: %s', exception)
        sys.exit(1)
