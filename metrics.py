"""
Auxiliary predefined functions. Can be expanded by the user.
"""
import numpy as np
from sklearn.metrics import accuracy_score

evals = []
best = 1


def mse(true, prediction):
    return np.sum((true-prediction)**2)/true.shape[0]


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


# You can print the variables computed in each instruction for understanding better what we are doing
def balanced_accuracy(true, prediction):
    # We compute the number of classes (n), and the number of examples per class
    classes, count = np.unique(true, return_counts=True)
    # We compute the weights for each class. The summation of the weights of all examples belonging to a class must be 1/n
    class_weights = [1/len(classes)/i for i in count]
    # We assign weights for each example, depending on their class
    example_weights = [class_weights[i] for i in true]
    # We compute the accuracy weighting each example according to the representation of the class it belongs to in the data
    return accuracy_score(true, prediction, sample_weight=example_weights)
