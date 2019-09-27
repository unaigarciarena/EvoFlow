import numpy as np


def mse(true, prediction):
    return (true-prediction)**2/true.shape[0]


def accuracy_error(true, prediction):
    return 1-np.sum(true == prediction)/true.shape[0]
