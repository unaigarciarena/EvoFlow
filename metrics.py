"""
Auxiliary predefined functions. Can be expanded by the user.
"""
import numpy as np


def mse(true, prediction):
    return (true-prediction)**2/true.shape[0]


def accuracy_error(true, prediction):

    true = np.argmax(true, axis=1)
    prediction = np.argmax(prediction, axis=1)

    return 1-np.sum(true == prediction)/true.shape[0]
