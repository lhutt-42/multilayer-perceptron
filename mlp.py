"""
Main script of the project.
"""

from mlp.parser.arguments import create_parser
from mlp.parser.file import read_dataset

def main():
    """
    Main function of the script.
    """

    parser = create_parser()
    args = parser.parse_args()

    pd = read_dataset(args.dataset)
    print(pd)

    match args.command:
        case 'split':
            ...
        case 'train':
            ...
        case 'predict':
            ...

if __name__ == '__main__':
    main()
