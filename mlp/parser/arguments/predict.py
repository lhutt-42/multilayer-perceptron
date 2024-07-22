"""
This module contains the predict subparser.
"""


def create_predict_subparser(subparsers) -> None:
    """
    Creates the subparser for the predict command.

    Args:
        subparsers: The subparsers object.
    """

    subparsers.add_parser(
        'predict',
        help='Trains the model.'
    )
