"""
This module contains the train subparser.
"""


def create_train_subparser(subparsers) -> None:
    """
    Creates the subparser for the train command.

    Args:
        subparsers: The subparsers object.
    """

    subparsers.add_parser(
        'train',
        help='Trains the model.'
    )
