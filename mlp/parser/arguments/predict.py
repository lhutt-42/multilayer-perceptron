"""
This module contains the predict subparser.
"""


def create_predict_subparser(subparsers) -> None:
    """
    Creates the subparser for the predict command.

    Args:
        subparsers: The subparsers object.
    """

    parser_predict = subparsers.add_parser(
        'predict',
        help='Trains the model.'
    )

    parser_predict.add_argument(
        'dataset',
        help='Path to the dataset.',
    )

    parser_predict.add_argument(
        'model',
        help='Path to the model.',
    )
