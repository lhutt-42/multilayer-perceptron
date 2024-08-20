"""
This module contains the functions to save and load datasets.
"""

import os
import sys

import pandas as pd

from . import logger


def load_dataset(path: str) -> pd.DataFrame:
    """
    Reads the dataset from a CSV file.

    Args:
        path: The path to the CSV file.

    Returns:
        pd.DataFrame: The dataset.
    """

    try:
        logger.info('Loading the dataset from `%s`.', path)
        return pd.read_csv(path, header=None)

    except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
        logger.error('Could not read the file `%s`: %s', path, e)
        sys.exit(1)

    except pd.errors.EmptyDataError:
        logger.error('The file `%s` is empty.', path)
        sys.exit(1)

    except pd.errors.ParserError:
        logger.error('The file `%s` could not be parsed.', path)
        sys.exit(1)

    # pylint: disable=broad-except
    except Exception as e:
        logger.error('Could not read the file `%s`: %s', path, e)
        sys.exit(1)


def save_dataset(dataset: pd.DataFrame, path: str) -> None:
    """
    Saves the dataset to a CSV file.

    Args:
        dataset: The dataset to save.
        path: The path to the CSV file.
    """

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dataset.to_csv(path, index=False, header=False)
        logger.info('Saved the dataset to `%s`.', path)

    except (PermissionError, IsADirectoryError) as e:
        logger.error('Could not write the file `%s`: %s', path, e)
        sys.exit(1)

    # pylint: disable=broad-except
    except Exception as e:
        logger.error('Could not write the file `%s`: %s', path, e)
        sys.exit(1)
