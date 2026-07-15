
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


start_latent = 1
end_latent = 11
n_loops = 10

uid = "SWIRL_WHALE_v1_"

data = [
"/home/nlahaye/NatureNet/src/naturenet/models/irl/training_data/10_12345_MLP_S1_net1.npz",
"/home/nlahaye/NatureNet/src/naturenet/models/irl/training_data/15_12345_MLP_S1_net1.npz",
"/home/nlahaye/NatureNet/src/naturenet/models/irl/training_data/20_12345_MLP_S1_net1.npz",
"/home/nlahaye/NatureNet/src/naturenet/models/irl/training_data/2_12345_MLP_S1_net1.npz",
"/home/nlahaye/NatureNet/src/naturenet/models/irl/training_data/30_12345_MLP_S1_net1.npz",
"/home/nlahaye/NatureNet/src/naturenet/models/irl/training_data/4_12345_MLP_S1_net1.npz",
"/home/nlahaye/NatureNet/src/naturenet/models/irl/training_data/6_12345_MLP_S1_net1.npz",
"/home/nlahaye/NatureNet/src/naturenet/models/irl/training_data/8_12345_MLP_S1_net1.npz",

#"/data/nlahaye/NatureNet/Blue_Whale_v1/8_12345_MLP_S1_net1.npz",
#"/data/nlahaye/NatureNet/Blue_Whale_v1/8_12345_MLP_S2_net2.npz"
  
]
 
latents = [10, 15, 20, 2, 30, 4, 6, 8, "8_S1_new", "8_S2"]

colors = ["orange", "red", "blue", "green", "violet", "cyan", "darkkhaki", "rosybrown", "lightgreen", "pink"]


for i in range(len(data)):

        model_data = np.load(data[i], allow_pickle=True)
        latent = latents[i]
        color = colors[i]

        print(model_data.keys(), latent, color)
        plt.plot(list(range(len(model_data["LL_list"]))), model_data["LL_list"], color=color, label=str(latent) + "_states")
plt.legend()
plt.savefig("Loss.png")




