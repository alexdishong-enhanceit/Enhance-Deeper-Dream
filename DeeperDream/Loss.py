import numpy as np


def BinaryCrossEntropy(y, p_hat):
    return -1 * np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))


def MSE(y, y_hat):
    return np.trace((y - y_hat).T @ (y - y_hat)) / y.shape[0]


def CrossEntropy(y, p_hat):
    """
    One-Hot Encoded
    """
    return -np.sum(y * np.log(p_hat))