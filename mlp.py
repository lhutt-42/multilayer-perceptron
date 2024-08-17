"""
Main script of the project.
"""

import logging

import numpy as np

from mlp.parser import create_parser
from mlp import split, train, predict


def main() -> None:
    """
    Main function of the script.
    """

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    parser = create_parser()
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)

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
            predict(
                args.dataset,
                args.model
            )

if __name__ == '__main__':
    main()
