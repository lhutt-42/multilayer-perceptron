"""
This module contains the functions to binarize the data.
"""

import sys
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def binarize(ser: pd.Series) -> np.ndarray:
    """
    Binarizes the values of a series.

    Args:
        ser (pd.Series): The series to binarize.

    Returns:
        np.ndarray: The binarized values.
    """

    try:
        return OneHotEncoder(sparse_output=False).fit_transform(ser.to_frame())

    except (KeyError, ValueError) as e:
        logging.error('Could not binarize the values: %s', e)
        sys.exit(1)
