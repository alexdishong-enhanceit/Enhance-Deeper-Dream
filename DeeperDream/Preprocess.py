
import numpy as np


def FlatNormalize(x, cols: list = []):
    """
    Change the range features of data to [0, 1]
    This will normalize in place

    :param x: The data to be normalized - expects each column to be a feature and each row to be a a data point.
    :param cols: List of indices of columns you want to normalize.
    """
    if not cols:
        cols = list(range(x.shape[1]))
    x[:, cols] -= np.min(x[:, cols], axis=0)
    x[:, cols] /= np.max(x[:, cols], axis=0)
    return x