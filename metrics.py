"""
Auxiliary predefined functions. Can be expanded by the user.
"""
import numpy as np

evals = []
best = 1


def mse(true, prediction):
    return (true-prediction)**2/true.shape[0]


def accuracy_error(true, prediction):
    global evals

    if len(true.shape) > 1:
        true = np.argmax(true, axis=1)
    if len(prediction.shape) > 1:
        prediction = np.argmax(prediction, axis=1)

    evals += [1-np.sum(true == prediction)/true.shape[0]]
    return 1-np.sum(true == prediction)/true.shape[0]


def ret_evals():
    global evals
    return evals