"""
Main script of the project.
"""

import logging

from mlp.parser import create_parser
from mlp import split
from mlp import train


def main() -> None:
    """
    Main function of the script.
    """

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    parser = create_parser()
    args = parser.parse_args()

    match args.command:
        case 'split':
            split(
                args.dataset,
                args.test_size,
                args.out_dir
            )
        case 'train':
            train(
                args.dataset,
                args.layers,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.loss,
                args.out_dir
            )
        case 'predict':
            ...

if __name__ == '__main__':
    main()
