import numpy as np


def Sigmoid(z):
    return 1 / (1+np.exp(-z))


def SoftMax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
