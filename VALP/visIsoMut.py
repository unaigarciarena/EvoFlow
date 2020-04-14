import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

first_seeds = 30
second_seeds = 100


def load_data():
    path = "../isoMut/"
    np_data = np.zeros((2, first_seeds, second_seeds, 3))
    pd_data = pd.DataFrame(columns=["Seed1", "Seed2", "Intelligent", "Mutation", "Result", "Regression", "Classification", "Sampling", "Result", "Mutation", "AffectedOutputs"])
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
                pd_data.loc[pd_data.shape[0]] = [seed1, seed2, 0, -2, -2, data[0], data[1], data[2], data[3], data[4], data[5:]]
                np_data[0, seed1, seed2] = data[:3]
            else:
                print(int_path)

            random_path = path + "resultsrandom" + str(seed1+1) + "_" + str(seed2) + ".npy"
            if os.path.isfile(random_path):
                data = np.load(random_path, allow_pickle=True)
                np_data[1, seed1, seed2] = data[:3]
                pd_data.loc[pd_data.shape[0]] = [seed1, seed2, 1, -2, -2, data[0], data[1], data[2], data[3], data[4], data[5:]]
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
        orig = orig - min
        orig = orig / max
        rnd = rnd - min
        rnd = rnd / max
        inte = inte - min
        inte = inte / max
        inte[inte>aux_lim] = aux_lim
        rnd[rnd>aux_lim] = aux_lim
        #rnd = rnd/orig
        #inte = inte/orig
        #rnd = np.log(rnd)
        #inte = np.log(inte)
        #rnd[rnd > 5] = 5
        #inte[inte > 5] = 5
        min = np.min([np.min(rnd), np.min(inte)])
        max = np.max([np.max(rnd), np.max(inte)])
        bins = np.arange(min, max+(max-min)/500, (max-min)/500)
        sns.distplot(rnd, label="Random mutation", norm_hist=True, kde=False, bins=bins)
        sns.distplot(inte, label="Intelligent mutation", norm_hist=True, kde=False, bins=bins)
        #sns.distplot(orig, label="Original", norm_hist=True, kde=False, bins=bins)
        plt.legend()
        plt.show()


def per_mut():
    pd_data = pd.read_csv("../isoMut/results.csv")
    # columns=["Seed1", "Seed2", "Intelligent", "Mutation", "Result", "Regression", "Classification", "Sampling", "AffectedOutputs"]
    rnd = pd_data[pd_data["Intelligent"]==1]
    inte = pd_data[pd_data["Intelligent"]==0]
    print(rnd)
    print(inte)


def tmp():
    for i in range(30):
        print(np.load("model" + str(i) + "/orig_results" + str(i) + ".npy"))


if __name__ == "__main__":
    load_data()
    #distributions()
    #tmp()
    per_mut()