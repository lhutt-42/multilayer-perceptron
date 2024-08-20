"""
This module contains the functions to normalize the data.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import logger


def normalize(df: pd.DataFrame) -> np.ndarray:
    """
    Normalizes the values of a DataFrame.

    Args:
        df (pd.DataFrame): The data to normalize.

    Returns:
        np.ndarray: The normalized values.
    """

    try:
        df = df.select_dtypes(include=np.number).to_numpy()
        return StandardScaler().fit_transform(df)

    except (KeyError, ValueError) as e:
        logger.error('Could not normalize the values: %s', e)
        sys.exit(1)
