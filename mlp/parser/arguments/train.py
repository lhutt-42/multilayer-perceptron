"""
This module contains the train subparser.
"""

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
        'model',
        help='Path to the model.',
    )

    parser_train.add_argument(
        '--out-dir',
        type=str,
        default='./models',
        help='Output directory.'
    )

    parser_train.add_argument(
        '--plot-n',
        type=int,
        default=2,
        help='Number of past metrics to plot.'
    )

    parser_train.add_argument(
        '--plot-multi',
        action='store_true',
        help='Plot multiple metrics.'
    )

    parser_train.add_argument(
        '--plot-raw',
        action='store_true',
        help='Plot the raw data.'
    )
