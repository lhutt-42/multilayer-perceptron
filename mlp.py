"""
Main script of the project.
"""

from mlp.parser.arguments import create_parser
from mlp.split import split_dataset

def main():
    """
    Main function of the script.
    """

    parser = create_parser()
    args = parser.parse_args()

    match args.command:
        case 'split':
            split_dataset(args.dataset, args.test_size, args.out_dir)
        case 'train':
            ...
        case 'predict':
            ...

if __name__ == '__main__':
    main()
