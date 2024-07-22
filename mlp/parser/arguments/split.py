"""
This module contains the split subparser.
"""

import argparse


class SizeAction(argparse.Action):
    """
    Custom action to validate the size of the test set.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if values < 0.0 or values > 1.0:
            raise argparse.ArgumentError(self, 'The value must be between 0 and 1.')
        setattr(namespace, self.dest, values)


def create_split_subparser(subparsers) -> None:
    """
    Creates the subparser for the split command.

    Args:
        subparsers: The subparsers object.
    """

    parser_split = subparsers.add_parser(
        'split',
        help='Splits the dataset.'
    )

    parser_split.add_argument(
        'dataset',
        help='Path to the dataset.',
    )

    parser_split.add_argument(
        '--test-size',
        type=float,
        action=SizeAction,
        default=0.1,
        help='Size of the test set'
    )

    parser_split.add_argument(
        '--out-dir',
        default='./data',
        help='Output directory.'
    )
