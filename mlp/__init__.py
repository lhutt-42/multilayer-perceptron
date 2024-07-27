"""
This module contains the primary functions of the project.
"""

from .split import split
from .train import train
from . import parser
from . import model

__all__ = [
    'parser',
    'model',
    'split',
    'train',
]
