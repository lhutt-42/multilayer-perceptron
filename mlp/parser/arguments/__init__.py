"""
This module contains parsing related to arguments.
"""

import argparse

from .split import create_split_subparser
from .train import create_train_subparser
from .predict import create_predict_subparser


def create_parser() -> argparse.ArgumentParser:
    """
    Creates the main parser.

    Returns:
        argparse.ArgumentParser: The main parser.
    """

    parser = argparse.ArgumentParser(
        prog='mlp',
        description='Multilayer Perceptron.'
    )
    subparsers = parser.add_subparsers(required=True, dest='command')

    create_split_subparser(subparsers)
    create_train_subparser(subparsers)
    create_predict_subparser(subparsers)

    return parser


__all__ = [
    'create_parser'
]
