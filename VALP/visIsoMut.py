import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import kruskal

first_seeds = 30
second_seeds = 100
outputs = ["Regression", "Classification", "Sampling"]


def load_data():
    path = "../isoMut/"
    np_data = np.zeros((2, first_seeds, second_seeds, 3))
    pd_data = pd.DataFrame(columns=["Seed1", "Seed2", "Random", "Regression", "Classification", "Sampling", "Result", "Mutation", "AffectedOutputs"])
    orig_res = np.zeros((30, 3))
    for seed1 in range(first_seeds):
        orig_path = path + "orig_results" + str(seed1+1) + ".npy"
        if os.path.isfile(orig_path):
            orig_res[seed1] = np.load(orig_path)
        else:
            print(orig_path)
        for seed2 in range(second_seeds):
            int_path = path + "results" + str(seed1+1) + "_" + str(seed2) + ".npy"
            if os.path.isfile(int_path):
                data = np.load(int_path, allow_pickle=True)
                pd_data.loc[pd_data.shape[0]] = [seed1, seed2, 0, data[0], data[1], data[2], data[3], data[4], data[5:]]
                np_data[0, seed1, seed2] = data[:3]
            else:
                print(int_path)

            random_path = path + "resultsrandom" + str(seed1+1) + "_" + str(seed2) + ".npy"
            if os.path.isfile(random_path):
                data = np.load(random_path, allow_pickle=True)
                np_data[1, seed1, seed2] = data[:3]
                pd_data.loc[pd_data.shape[0]] = [seed1, seed2, 1, data[0], data[1], data[2], data[3], data[4], data[5:]]
            else:
                print(random_path)

    np.save("../isoMut/origs.npy", orig_res)
    np.save("../isoMut/results.npy", np_data)
    pd_data.to_csv("../isoMut/results.csv")


def distributions():
    orig_res = np.load("../isoMut/origs.npy")
    np_data = np.load("../isoMut/results.npy")
    # pd_data = pd.read_csv("../isoMut/results.csv")
    objs = ["Regression", "Classification", "Sampling"]
    aux_lim = 1
    for obj in range(3):
        orig = np.reshape(orig_res[:, obj], (-1, 1))
        rnd = np_data[1, :, :, obj]
        inte = np_data[0, :, :, obj]
        min = np.min([np.min(orig), np.min(rnd), np.min(inte)]) + 0.01
        max = np.max([np.min(orig), np.max(rnd), np.max(inte)])
        inte[inte > aux_lim] = aux_lim
        rnd[rnd > aux_lim] = aux_lim
        orig[orig > aux_lim] = aux_lim
        orig = orig - min
        orig = orig / max
        rnd = rnd - min
        rnd = rnd / max
        inte = inte - min
        inte = inte / max

        #rnd = rnd-orig
        #inte = inte-orig
        #rnd = np.log(rnd)
        #inte = np.log(inte)
        min = np.min([np.min(rnd), np.min(inte)])
        max = np.max([np.max(rnd), np.max(inte)])
        bins = np.arange(min, max+(max-min)/500, (max-min)/500)
        sns.distplot(rnd, label="Random mutation", norm_hist=True, kde=False, bins=bins, hist_kws={"cumulative": True})
        sns.distplot(inte, label="Intelligent mutation", norm_hist=True, kde=False, bins=bins, hist_kws={"cumulative": True})
        sns.distplot(orig, label="Original", norm_hist=True, kde=False, bins=bins, hist_kws={"cumulative": True})
        plt.legend()
        plt.show()


def per_mut():
    pd_data = pd.read_csv("../isoMut/results.csv", index_col=0)
    print(pd_data)
    # columns=["Seed1", "Seed2", "Intelligent", "Mutation", "Result", "Regression", "Classification", "Sampling", "AffectedOutputs"]

    rnd = pd_data[pd_data["Random"] == 1]
    inte = pd_data[pd_data["Random"] == 0]
    aux_lims = [0.1, 1, 1]
    for obj in range(3):
        rndo = rnd[["o" + str(obj) in x for x in rnd["AffectedOutputs"]]]
        inteo = inte[["o" + str(obj) in x for x in inte["AffectedOutputs"]]]
        for mut in ["c", "bypass", "add", "divide", "reinit"]:
            rndom = rndo[[mut in x for x in rndo["Mutation"]]]
            inteom = inteo[[mut in x for x in inteo["Mutation"]]]
            print(rndom.shape, inteom.shape)
            rndom = rndom[outputs[obj]]
            inteom = inteom[outputs[obj]]

            inteom[inteom > aux_lims[obj]] = aux_lims[obj]
            rndom[rndom > aux_lims[obj]] = aux_lims[obj]
            min = np.min([np.min(rndom), np.min(inteom)]) + 0.01
            max = np.max([np.max(rndom), np.max(inteom)])
            rndom = rndom - min
            rndom = rndom / max
            inteom = inteom - min
            inteom = inteom / max

            min = np.min([np.min(rndom), np.min(inteom)])
            max = np.max([np.max(rndom), np.max(inteom)])
            bins = np.arange(min, max + (max - min) / 500, (max - min) / 500)

            sns.distplot(rndom, label="Random mutation", norm_hist=True, kde=False, bins=bins, hist_kws={"cumulative": False})
            sns.distplot(inteom, label="Intelligent mutation", norm_hist=True, kde=False, bins=bins, hist_kws={"cumulative": False})
            plt.title("Mutation: " + mut + ", Objective: " + outputs[obj])
            plt.legend()
            plt.savefig(mut+outputs[obj] + ".pdf")
            plt.clf()


def tmp():
    for i in range(30):
        print(np.load("model" + str(i) + "/orig_results" + str(i) + ".npy"))


def per_model():
    np_data = np.load("../isoMut/results.npy")
    pd_data = pd.read_csv("../isoMut/results.csv", index_col=0)
    for i in range(30):
        data = pd_data[pd_data["Seed1"] == i]
        rnd = data[data["Random"] == 1]
        int = data[data["Random"] == 0]
        print(i, np.unique(int["Mutation"], return_counts=True))


def stackplots():
    # np_data = np.zeros((2, first_seeds, second_seeds, 3))
    np_data = np.load("../isoMut/results.npy")
    int = np_data[0]
    rnd = np_data[1]
    lim = 1
    for obj in range(len(outputs)):
        into = int[:, :, obj].flatten()
        rndo = rnd[:, :, obj].flatten()
        into[into > lim] = lim
        rndo[rndo > lim] = lim
        min = np.min([np.min(into), np.min(rndo)])
        max = np.max([np.max(into), np.max(rndo)])*1.1
        hinto = np.histogram(into, bins=np.arange(min, max, (max - min) / 100))
        hrndo = np.histogram(rndo, bins=np.arange(min, max, (max - min) / 100))

        y = np.concatenate((np.reshape(np.append(hinto[0], 0), (1, -1)), np.reshape(np.append(hrndo[0], 0), (1, -1))), axis=0)
        for i in range(y.shape[1]-1):
            y[:, i+1] = y[:, i] + y[:, i+1]
        y = y.astype("float32")
        for i in range(y.shape[1]):
            print(y[:, i] / np.sum(y[:, i]))
            y[:, i] = y[:, i]/np.sum(y[:, i])
            print(y[:, i])
        plt.plot([-20, 1], [0.5, 0.5])
        plt.stackplot(hinto[1], y)
        plt.show()


def weights():
    data = np.load("loss_weights4.npy")[:3948]

    colors = ["r", "b", "g", "y", "c"]
    patterns = ["solid", "dashed"]

    for i in range(data.shape[1]//2):
        plt.plot(np.log(data[:, i]), color=colors[i], linestyle=patterns[0])
        plt.plot(np.log(data[:, i+data.shape[1]//2]), color=colors[i], linestyle=patterns[1])

    legend_elements = [Line2D([0], [0], marker='s', color=colors[0], label='Scatter', markerfacecolor=colors[0], markersize=15), Line2D([0], [0], marker='s', color=colors[1], label='Scatter', markerfacecolor=colors[1], markersize=15),
                       Line2D([0], [0], marker='s', color=colors[2], label='Scatter', markerfacecolor=colors[2], markersize=15), Line2D([0], [0], marker='s', color=colors[3], label='Scatter', markerfacecolor=colors[3], markersize=15),
                       Line2D([0, 1], [0, 1], linestyle=patterns[0], color='black'), Line2D([0, 1], [0, 1], linestyle=patterns[1], color='black')]
    plt.legend(legend_elements, ["Obj1", "Obj2", "Obj3", "Obj4", "Loss", "Weight"])

    plt.show()


def compare_trainings(log=True):
    n_networks = 5
    n_seeds = 30
    epochs = 10000

    df = pd.DataFrame(columns=["Seed", "Method", "Epoch", "Obj", "Value"])
    strategies = ["Static", "Dynamic", "Sequential", "o0", "o1", "o2", "Scale Static", "Scale Dynamic"]
    data = np.zeros((len(strategies), n_seeds, epochs, 4))
    list_data = [[]*len(strategies)]
    if log:
        pd_path = "pdTrainLog.csv"
        np_path = "trainLog.npy"
    else:
        pd_path = "pdTrain.csv"
        np_path = "train.npy"
    if not os.path.isfile(np_path):
        for seed in range(1, n_seeds+1):
            for strati in range(len(strategies)):
                print(seed)
                aux = np.load("/home/unai/PycharmProjects/EvoFlow/IntTrain/model_" + str(strati) + "_" + str(n_networks) + "_" + str(seed) + "/loss_weights" + str(seed) + "_" + str(strati) + "_" + str(n_networks) + "_.npy")
                if strati == 2:
                    aux = aux[:10000]
                else:
                    aux = aux[:, :aux.shape[1] // 2]

                for i in np.arange(1, epochs):
                    aux[i] = (aux[i] > 0) * aux[i] + (aux[i] <= 0) * aux[i-1]

                if aux.shape[1] > 4:
                    aux[:, 3] = np.mean(aux[:, 3:], axis=1)
                    aux = aux[:, :4]

                data[strati, seed-1] = aux

        np.save(np_path, data)
    else:
        data = np.load(np_path)

    if not os.path.isfile(pd_path):
        for seed in range(0, n_seeds):
            for strati, strat in enumerate(strategies):
                print(seed)
                aux = data[strati, seed]
                for i in np.arange(0, epochs, 100):

                    df.loc[df.shape[0]] = [seed, strat, i, 0, np.log(aux[i, 0])]
                    df.loc[df.shape[0]] = [seed, strat, i, 1, np.log(aux[i, 1])]
                    df.loc[df.shape[0]] = [seed, strat, i, 2, np.log(aux[i, 2])]
                    df.loc[df.shape[0]] = [seed, strat, i, 3, np.log(aux[i, 3])]

        df.to_csv(pd_path)
    else:
        df = pd.read_csv(pd_path, index_col=0)

    df = df[~((df["Method"] == "o0") & (df["Obj"] == 1))]
    df = df[~((df["Method"] == "o0") & (df["Obj"] == 2))]
    df = df[~((df["Method"] == "o0") & (df["Obj"] == 3))]

    df = df[~((df["Method"] == "o1") & (df["Obj"] == 0))]
    df = df[~((df["Method"] == "o1") & (df["Obj"] == 2))]
    df = df[~((df["Method"] == "o1") & (df["Obj"] == 3))]

    #df = df[~((df["Method"] == "o2") & (df["Obj"] == 0))]
    #df = df[~((df["Method"] == "o2") & (df["Obj"] == 1))]

    sns.lineplot(x="Epoch", y="Value", hue="Method", style="Obj", data=df)
    plt.show()


def statistical_test(force=False):
    data = np.load("trainLog.npy")
    test_path = "Tests.npy"
    interval = 100
    limit = 0.05
    if not os.path.isfile(test_path) or force:
        tests = np.zeros((data.shape[0], data.shape[2] // interval, data.shape[3]))
        for obj in range(4):
            for epoch in np.arange(0, data.shape[2], interval):
                for i in range(data.shape[0]):
                    for j in range(i+1, data.shape[0]):
                        try:
                            a = kruskal(data[i, :, epoch, obj], data[j, :, epoch, obj], nan_policy="omit")[1]
                            print(a)
                        except ValueError:
                            a = 100
                            print(i, j)
                        if a <= limit:
                            if np.mean(data[i, :, epoch, obj]) < np.mean(data[j, :, epoch, obj]):
                                tests[i, epoch//interval, obj] += 1
                            else:
                                tests[j, epoch // interval, obj] += 1
        np.save(test_path, tests)
    else:
        tests = np.load(test_path)
    methods = [[0, 1, 2, 5, 6, 7], [0, 1, 2, 4, 6, 7], [0, 1, 2, 3, 6, 7], [0, 1, 2, 5, 6, 7]]
    objs = ["Regression", "Classification", "Sampling", "KL"]
    for obj in range(4):
        sns.heatmap(tests[methods[obj], :80, obj])
        plt.xlabel("Epoch*100")
        plt.ylabel("Method")
        plt.yticks(np.arange(0.5, len(methods)+2, 1), ["Static", "Dynamic", "Sequential", "Specific", "Scale Static", "Scale Dynamic"])
        plt.title(objs[obj])
        plt.show()
    print(tests)

if __name__ == "__main__":
    #load_data()
    #distributions()
    #tmp()
    #per_mut()
    #per_model()
    #stackplots()
    #weights()
    #compare_trainings()
    statistical_test()
