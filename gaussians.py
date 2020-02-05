import numpy as np
import matplotlib.pyplot as plt
import losses
import tensorflow as tf
from deap.tools._hypervolume.pyhv import _HyperVolume
from scipy.stats import entropy
import os
import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

probs = [8, 4, 2, 25, 1]

sq2 = 1/np.sqrt(2)
means_8 = [(0, -1), (-sq2, -sq2), (-1, 0), (-sq2, sq2), (sq2, -sq2), (1, 0), (sq2, sq2), (0, 1)]
means_4 = [(0, -1), (-1, 0), (1, 0), (0, 1)]
means_25 = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2),
            (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

means_1 = [(0, 0)]
means_2 = [(0, 0), (0, 1)]

problems = {8: means_8, 25: means_25, 1: means_1, 2: means_2, 4: means_4}
std_dev = 0.05


def create_data(modes=8, samples=1000, shuffle=True):
    data = []
    samp = samples // modes

    for i, mean in enumerate(problems[modes]):
        data += [(np.random.normal(mean, (std_dev, std_dev), size=(samp, 2)))]

    while np.sum([x.shape[0] for x in data]) < samples:

        data += [(np.random.normal(problems[modes][np.random.randint(0, len(problems[modes]))], (std_dev, std_dev), size=(1, 2)))]

    data = np.concatenate(data)

    if shuffle:
        np.random.shuffle(data)

    return data


def create_multi_data_set(modes=2, samples=100, sets=50):

    data = []
    for i in range(sets):
        for mode in range(modes):
            data += [create_data(probs[mode], samples, shuffle=False).flatten()]

    return np.array(data)


def create_data_set(mode=8, samples=100, sets=50):

    data = []
    for i in range(sets):
        data += [create_data(mode, samples, shuffle=False).flatten()]
    return np.array(data)


def assign_centers(data, modes=8):

    if modes == 8:
        y1 = data[:, 1] - ((np.sqrt(2) + 1) * data[:, 0])
        y2 = data[:, 1] - (-1 / (np.sqrt(2) + 1) * data[:, 0])
        y3 = data[:, 1] - (-(np.sqrt(2) + 1) * data[:, 0])
        y4 = data[:, 1] - (1 / (np.sqrt(2) + 1) * data[:, 0])

        ys = np.concatenate((y1.reshape(-1, 1), y2.reshape(-1, 1), y3.reshape(-1, 1), y4.reshape(-1, 1)), axis=1)

        return np.sum(ys > 0, axis=1) + (ys[:, 2] > 0)*3

    if modes == 25:
        ys = [0 if i[1] < 0.5 else 1 if i[1] < 1.5 else 2 if i[1] < 2.5 else 3 if i[1] < 3.5 else 4 for i in data]
        xs = [0 if i[0] < 0.5 else 5 if i[0] < 1.5 else 10 if i[0] < 2.5 else 15 if i[0] < 3.5 else 20 for i in data]

        return np.array(ys) + np.array(xs)

    if modes == 1:
        return np.zeros(data.shape[0])

    if modes == 2:
        return data[:, 1] > 0.5

    if modes == 4:
        y1 = data[:, 1] - ((np.sqrt(2) + 1) * data[:, 0])
        y2 = data[:, 1] - (-1 / (np.sqrt(2) + 1) * data[:, 0])

        return int(y1 > 0)*2 + int(y2 > 0)


def weighted_distance(points, alpha=0.005, modes=8):

    if alpha is None:
        alpha = 1./points.shape[0]
    assign, std_devs, means, centers = evaluate_samples(points, modes)
    return alpha * assign + (1 - alpha) * std_devs, centers


def plt_center_assignation(data, centers=None, modes=None, save=False, name="", show_centroids=True):
    cs = np.unique(centers)

    markers = ["+", "x", "*", "_", "d"]

    if centers is None:
        plt.plot(data[:, 0], data[:, 1], "bo")
    else:
        for c in cs:
            d = data[centers == c]
            plt.plot(d[:, 0], d[:, 1], markers[c % len(markers)], markersize=15)
    if centers is not None and show_centroids:
        centroids = np.array(problems[modes])
        plt.plot(centroids[:, 0], centroids[:, 1], "bo", markersize=10)
    if save:
        plt.savefig(name)
        plt.clf()
    else:
        plt.show()


def evaluate_samples(data, modes=8):

    centers = assign_centers(data, modes=modes)

    distribution = np.unique(centers, return_counts=True)
    assign = np.std(list(distribution[1]) + [0]*(8-distribution[1].shape[0]))
    std_devs = []
    means = []
    for i, c in enumerate(distribution[0]):
        d = data[centers == c]
        std_devs += [np.sqrt(np.mean((d - problems[modes][i]) ** 2, axis=0))]
        means += [np.mean(d, axis=0)-problems[modes][i]]

    std_devs = np.sum(np.abs(np.array(std_devs)-std_dev))/modes
    means = np.sum(np.abs(means))/modes
    return assign, std_devs, means, centers


def evaluate_sets(datasets, modes=8):
    assignations = []
    std_devs = []
    centers = []
    for data in datasets:
        assign, std_dev_est, means, cs = evaluate_samples(data, modes)
        assignations += [assign]
        std_devs += [std_dev_est]
        centers += [cs]


def mmd(candidate, target, modes=8):
    centers = assign_centers(candidate, modes=modes)
    sess = tf.compat.v1.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
    res = losses.mmd_loss(tf.Variable(candidate, dtype="float32", name="Candidate"), tf.Variable(target, dtype="float32", name="Target"), 1)
    sess.run(tf.global_variables_initializer())
    res = sess.run(res)
    if res < 0.00001:
        res = 10000
    sess.close()
    return res, centers


def mmd_sets(candidates, targets, modes):

    objectives = []
    distances = []
    centers = []
    hv = _HyperVolume([0] * modes)
    for candidate in candidates:
        dists = []
        for i in modes:
            res, _ = mmd(np.reshape(candidate, (-1, 2)), np.reshape(targets[i], (-1, 2)), probs[i])
            dists += [res]
        objectives += [dists]
    objectives = np.array(objectives)
    _, _, index = pareto_frontier(objectives[:, 0], objectives[:, 1])
    objectives = objectives[index]
    candidates = candidates[index]
    for i in range(candidates.shape[0]):
        centers += [[]]
        for j in range(modes):
            centers[-1] += [assign_centers(candidates[i].reshape((-1, 2)), probs[j])]

    a = hv.compute(objectives)

    return a, candidates, np.array(centers)


def pareto_frontier(xs, ys, maxy=False):
    """
    Method to take two equally-sized lists and return just the elements which lie
    on the Pareto frontier, sorted into order.
    Default behaviour is to find the maximum for both X and Y, but the option is
    available to specify maxX = False or maxY = False to find the minimum for either
    or both of the parameters.
    """
    # Sort the list in either ascending or descending order of X
    x_index = np.argsort(xs)
    my_list = [[xs[i], ys[i]] for i in x_index]

    # Start the Pareto frontier with the first value in the sorted list
    p_front = [my_list[0]]
    indices = [x_index[0]]
    # Loop through the sorted list
    for i, pair in enumerate(my_list[1:]):
        if maxy:
            if pair[1] >= p_front[-1][1]:  # Look for higher values of Y
                p_front.append(pair)       # and add them to the Pareto frontier
                indices += [x_index[i+1]]
        else:

            if pair[1] <= p_front[-1][1]:  # Look for lower values of Y
                p_front.append(pair)       # and add them to the Pareto frontier
                indices += [x_index[i+1]]

    # Turn resulting pairs back into a list of Xs and Ys
    p_frontx = [pair[0] for pair in p_front]
    p_fronty = [pair[1] for pair in p_front]
    return p_frontx, p_fronty, indices


def discretization_probs(dist, rng):

    disc = []
    aux = 0
    for i in range(1, rng.shape[0]):
        for j in range(1, rng.shape[0]):
            d = dist[dist[:, 0] < rng[i]]
            d = d[rng[i-1] < d[:, 0]]
            d = d[d[:, 1] < rng[j]]
            d = d[rng[j-1] < d[:, 1]]
            disc += [d.shape[0]]
    disc = np.array(disc)
    disc = disc + 0.0001
    return disc


def kullback_leibler(target, candidate, modes=8, disc=0):
    discs = {8: [np.arange(-1.5, 1.5, 0.4), np.arange(-1.5, 1.5, 0.2), np.arange(-1.5, 1.5, 0.1)],
             25: [np.arange(-0.5, 4.5, 1), np.arange(-0.5, 4.5, 0.5), np.arange(-0.5, 4.5, 0.25)]}

    centers = assign_centers(candidate, modes)
    target_probs = discretization_probs(target, discs[modes][disc])
    cand_probs = discretization_probs(candidate, discs[modes][disc])

    return entropy(target_probs, cand_probs), centers


def perfect_example():
    data = create_data(8, 1000, False)
    a = sns.jointplot(data[:, 0], data[:, 1], kind="kde", bw=0.08)
    a.ax_marg_x.remove()
    a.ax_marg_y.remove()

    plt.show()


if __name__ == "__main__":
    samps = 1000
    n_centers = 8
    dist = create_data(modes=n_centers, samples=samps)
    dist1 = np.load("GausResults/Samples_8_4_16_1.3463554_46.05456042289734.npy")
    dist2 = create_data(modes=4, samples=samps)
    rnd = np.random.rand(samps, 2) * 2 - 1
    print("Perfect")
    print(weighted_distance(dist))
    print(mmd(dist, dist, modes=n_centers)[0])
    print("Approx")
    print(weighted_distance(dist1))
    print(mmd(dist, dist1, modes=n_centers)[0])
    print("4 modes")
    print(weighted_distance(dist2))
    print(mmd(dist, dist2, modes=n_centers)[0])
    print("Random")
    print(weighted_distance(rnd))
    print(mmd(dist, rnd, modes=n_centers)[0])
