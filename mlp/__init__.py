"""
This module contains the primary functions of the project.
"""

from . import parser
from . import model
from .split import split
from .train import train
from .predict import predict

__all__ = [
    'parser',
    'model',
    'split',
    'train',
    'predict'
]
