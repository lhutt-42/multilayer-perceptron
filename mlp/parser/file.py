"""
This module contains the functions to interact with files.
"""

import sys
import logging

import pandas as pd

def read_dataset(path: str) -> pd.DataFrame:
    """
    Reads the dataset from a CSV file.

    Args:
        path: The path to the CSV file.

    Returns:
        pd.DataFrame: The dataset.
    """

    try:
        return pd.read_csv(path, header=None)

    except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
        logging.error('Could not read the file `%s`: %s', path, e)
        sys.exit(1)

    except pd.errors.EmptyDataError:
        logging.error('The file `%s` is empty.', path)
        sys.exit(1)

    except pd.errors.ParserError:
        logging.error('The file `%s` could not be parsed.', path)
        sys.exit(1)
