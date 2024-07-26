"""
Main script of the project.
"""

import logging

from mlp.parser.arguments import create_parser
from mlp.split import split_dataset
from mlp.train import train_model


def main() -> None:
    """
    Main function of the script.
    """

    logging.basicConfig(level=logging.DEBUG)

    parser = create_parser()
    args = parser.parse_args()

    match args.command:
        case 'split':
            split_dataset(
                args.dataset,
                args.test_size,
                args.out_dir
            )
        case 'train':
            train_model(
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
