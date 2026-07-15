
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


start_latent = 1
end_latent = 26
n_loops = 1 #0

uid = "SWIRL_WHALE_v1_"

data_dir = "/data/nlahaye/NatureNet/Blue_Whale_LL_Test/"

arhmm_stds = []
arhmm_mean = []
swirl_stds = []
swirl_mean = []

swirl_test_stds = []
swirl_test_mean = []

latents = []
for latent in range(start_latent, end_latent+1, 5):
    arhmm_ll = []
    swirl_ll = []
    swirl_test_ll = []
    for i in range(n_loops):

        ar_fname = os.path.join(data_dir, uid + "loop_" + str(i) + "_latent_" + str(latent) + "_states_" + str(latent) + "_hidden_12345_seed_arhmm_s_LLs.npz")
        swirl_fname = os.path.join(data_dir, uid + "loop_" + str(i) + "_latent_" + str(latent) + "_states_" + str(latent) + "_12345_naturenet_MLP_S1_new1_LL.npz")

        swirl_test_fname = os.path.join(data_dir, uid + "loop_" + str(i) + "_latent_" + str(latent) + "_states_" + str(latent) + "_12345_naturenet_MLP_S1_new1_LL_test.npz")

        if not os.path.exists(ar_fname) or not os.path.exists(swirl_fname):
            print("NO FILE", ar_fname, swirl_fname)
            continue
   
        print(swirl_fname)
 
        arhmm = np.load(ar_fname, allow_pickle=True)
        swrl = np.load(swirl_fname, allow_pickle=True)
        swrl_test = np.load(swirl_test_fname, allow_pickle=True)

        arhmm_ll.append(arhmm["log_likelihood"][-1])
        swirl_ll.append(swrl["LL"])
        swirl_test_ll.append(swrl_test["LL"])

    if len(arhmm_ll) < 1 or len(swirl_ll) < 1:
        continue
    latents.append(latent)
    print(min(arhmm_ll), max(arhmm_ll), min(swirl_ll), max(swirl_ll), min(swirl_test_ll), max(swirl_test_ll))

    arhmm_mean.append(np.mean(arhmm_ll))
    arhmm_stds.append(np.std(arhmm_ll))

    swirl_stds.append(np.std(swirl_ll))
    swirl_mean.append(np.mean(swirl_ll))

    swirl_test_mean.append(np.mean(swirl_test_ll))
    swirl_test_stds.append(np.std(swirl_test_ll))

print(np.subtract(arhmm_mean, arhmm_stds), np.add(arhmm_mean, arhmm_stds), np.subtract(swirl_mean, swirl_stds), np.add(swirl_mean, swirl_stds))

plt.plot(latents, arhmm_mean, color='red', label="ARHMM")
plt.fill_between(latents, np.subtract(arhmm_mean, arhmm_stds), np.add(arhmm_mean, arhmm_stds), color='red', alpha=0.2)

plt.plot(latents, swirl_mean, color='blue', label="SWIRL")
plt.fill_between(latents, np.subtract(swirl_mean, swirl_stds), np.add(swirl_mean, swirl_stds), color='blue', alpha=0.2)

plt.plot(latents, swirl_test_mean, color='green', label="SWIRL_TEST")
plt.fill_between(latents, np.subtract(swirl_test_mean, swirl_test_stds), np.add(swirl_test_mean, swirl_test_stds), color='green', alpha=0.2)


plt.legend()

plt.savefig("LL_ARHMM.png")

