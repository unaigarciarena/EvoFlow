import numpy as np
import matplotlib.pyplot as plt

evolution = np.load("/home/unai/Escritorio/EvoFlow/simple_res.npy")[:10000]
random = np.load("/home/unai/Escritorio/EvoFlow/simple_res_rand.npy")[:10000]
wann = np.load("/home/unai/Escritorio/EvoFlow/temp_evals.npy")[:10000]

for i in range(evolution.shape[0]-1):
    evolution[i+1] = np.min([evolution[i], evolution[i+1]])
    random[i + 1] = np.min([random[i], random[i + 1]])
    wann[i + 1] = np.min([wann[i], wann[i + 1]])

plt.plot(wann, label="Evoflow")
#plt.plot(np.log(random), label="Random")
#plt.legend()
plt.ylabel("accuracy error")
plt.xlabel("Evaluations")
plt.show()