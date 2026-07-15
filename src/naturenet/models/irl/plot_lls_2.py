
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


fnames = [
"/data/nlahaye/NatureNet/Blue_Whale_v1/s1_simple/SIMPLE_SWIRL_WHALE_v1_8_12345_naturenet_MLP_S1_new1_LL.npz",

"/data/nlahaye/NatureNet/Blue_Whale_v1/s2_simple/SIMPLE_SWIRL_WHALE_v1_8_12345_naturenet_MLP_S2_new2_LL.npz",
#"/data/nlahaye/NatureNet/Blue_Whale_v1/SIMPLE_SWIRL_WHALE_v1_8_hidden_12345_seed_arhmm_s_LLs.npz",


#"/data/nlahaye/NatureNet/Blue_Whale_v1/SWIRL_WHALE_v1_8_hidden_12345_seed_arhmm_s_LLs.npz",
"/data/nlahaye/NatureNet/Blue_Whale_v1/s2_complex/SWIRL_WHALE_v1_8_12345_naturenet_MLP_S2_new2_LL.npz",
"/data/nlahaye/NatureNet/Blue_Whale_v1/s1_complex/SWIRL_WHALE_v1_8_12345_naturenet_MLP_S1_new1_LL.npz",
] 


labels = ["s1_simple", "s2_simple", "s2_complex", "s1_complex"] #"simple_arhmm", "complex_arhmm",


for i in range(len(labels)):
    ll = np.load(fnames[i])
    if "arhmm" in fnames[i]:
        val = ll["log_likelihood"][-1]
    else:
        val = ll["LL"]

    plt.scatter(i, val, label=labels[i])

plt.legend()
plt.show()
plt.savefig("LL_Complex_vs_Simple_2.png")





