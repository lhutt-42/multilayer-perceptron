"""
This module contains the train subparser.
"""

import argparse

from mlp.model.losses import BinaryCrossEntropyLoss


class LayersAction(argparse.Action):
    """
    Custom action to ensure each layer size is a positive integer.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if not all(v > 0 for v in values):
            raise argparse.ArgumentError(self, 'Each layer size must be a positive integer.')
        setattr(namespace, self.dest, values)


class PositiveAction(argparse.Action):
    """
    Custom action to ensure the value is strictly positive.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if values <= 0:
            raise argparse.ArgumentError(self, f'The {self.dest} must be strictly positive.')
        setattr(namespace, self.dest, values)


LOSS_FUNCTIONS = {
    'bce': BinaryCrossEntropyLoss
}

class LossAction(argparse.Action):
    """
    Custom action to convert the loss argument to the corresponding class.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, LOSS_FUNCTIONS[values])


def create_train_subparser(subparsers) -> None:
    """
    Creates the subparser for the train command.

    Args:
        subparsers: The subparsers object.
    """

    parser_train = subparsers.add_parser(
        'train',
        help='Trains the model.'
    )

    parser_train.add_argument(
        'dataset',
        help='Path to the dataset.',
    )

    parser_train.add_argument(
        '--layers',
        nargs='*',
        type=int,
        action=LayersAction,
        default=[24, 24, 24],
        help='Number of neurons in each layer.'
    )

    parser_train.add_argument(
        '--epochs',
        type=int,
        action=PositiveAction,
        default=100_000,
        help='The number of epochs to train the model.'
    )

    parser_train.add_argument(
        '--batch-size',
        type=int,
        action=PositiveAction,
        default=32,
        help='The batch size used during training.'
    )

    parser_train.add_argument(
        '--learning-rate',
        type=float,
        action=PositiveAction,
        default=1e-4,
        help='The learning rate of the optimizer.'
    )

    parser_train.add_argument(
        '--loss',
        type=str,
        choices=LOSS_FUNCTIONS.keys(),
        action=LossAction,
        default=BinaryCrossEntropyLoss,
        help='The loss function to use.'
    )

    parser_train.add_argument(
        '--out-dir',
        type=str,
        default='./models',
        help='Output directory.'
    )
