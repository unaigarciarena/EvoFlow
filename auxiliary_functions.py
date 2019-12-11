import numpy as np

from Network import MLPDescriptor, ConvDescriptor, initializations, activations


def batch(x, size, i):
    """
    :param x: Poplation; set of solutions intended to be fed to the net in the input
    :param size: Size of the batch desired
    :param i: Index of the last solution used in the last epoch
    :return: The index of the last solution in the batch (to be provided to this same
             function in the next epoch, the solutions in the actual batch, and their
             respective fitness scores
    """

    if i + size > x.shape[0]:  # In case there are not enough solutions before the end of the array

        index = i + size-x.shape[0]  # Select all the individuals until the end and restart
        return np.concatenate((x[i:, :], x[:index, :]))
    else:  # Easy case
        index = i+size
        return x[i:index, :]


def init_conv_desc(input_dim, output_dim, max_lay, max_size):

    layers = np.random.choice([0, 1, 2], size=np.random.randint(np.max((max_lay, 2))))  # Number of layers = 2

    filters = [np.random.randint(1, 5) if layers[i] == 2 else -1 for i in range(layers.shape[0])]
    init = [np.random.randint(0, 2) if layers[i] == 2 else -1 for i in range(layers.shape[0])]
    act = [np.random.randint(0, 2) if layers[i] == 2 else -1 for i in range(layers.shape[0])]

    sizes = np.concatenate((np.random.randint(2, 4, size=(layers.shape[0], 2)), np.random.randint(1, 3, size=(layers.shape[0], 1))), axis=1)
    strides = np.random.randint(1, 2, size=layers.shape[0])
    descriptor = ConvDescriptor(input_dim, output_dim, layers, filters, strides, sizes, act, init)

    return descriptor
