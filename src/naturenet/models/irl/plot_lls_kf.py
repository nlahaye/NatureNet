
import numpy as np
import os

import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


start_latent = 1
end_latent = 26
n_loops = 1 #0
k_folds = 5

uid = "SWIRL_WHALE_v1_"

data_dir = "/data/nlahaye/NatureNet/Blue_Whale_LL_Test/"


arhmm_stds = []
arhmm_mean = []

swirl_stds = []
swirl_mean = []
swirl_s2_stds = []
swirl_s2_mean = []

swirl_test_stds = []
swirl_test_mean = []

swirl_s2_test_stds = []
swirl_s2_test_mean = []

swirl_test_se = []
swirl_s2_test_se = []

swirl_nn_stds = []
swirl_nn_mean = []

swirl_nn_test_stds = []
swirl_nn_test_mean = []

swirl_nn_test_se = []


latents = []

latent_set = [1,2,3,4,5,6,7,8,9,10,11,12,16,21,40]

cl = 0.95 # confidence level

for latent in latent_set: #range(start_latent, end_latent+1, 5):
    arhmm_ll = []

    swirl_ll = []
    swirl_test_ll = []
    swirl_s2_ll = []
    swirl_s2_test_ll = []

    swirl_nn_ll = []
    swirl_nn_test_ll = []
    for i in range(k_folds):

        ar_fname = os.path.join(data_dir, uid + "loop_0_latent_" + str(latent) + "_states_" + str(latent) + "_hidden_12345_seed_arhmm_s_LLs.npz")

        swirl_nn_fname = os.path.join(data_dir, uid + "loop_0_latent_" + str(latent) + "_states_" + str(latent) + "_12345_naturenet_MLP_S1_new1_LL_KF_" + str(i) + ".npz")

        swirl_s2_fname = os.path.join(data_dir, uid + "loop_0_latent_" + str(latent) + "_states_" + str(latent) + "_12345_naturenet_iter2_LL2_KF_" + str(0) + ".npz")
        swirl_fname = os.path.join(data_dir, uid + "loop_0_latent_" + str(latent) + "_states_" + str(latent) + "_12345_naturenet_S1_iter2_LL1_KF_" + str(0) + ".npz")
  
        swirl_nn_test_fname = os.path.join(data_dir, uid + "loop_0_latent_" + str(latent) + "_states_" + str(latent) + "_12345_naturenet_MLP_S1_new1_LL_test_KF_" + str(i) + ".npz")

        swirl_test_fname = os.path.join(data_dir, uid + "loop_0_latent_" + str(latent) + "_states_" + str(latent) + "_12345_naturenet_S1_iter2_LL1_test_KF_" + str(0) + ".npz") #TODO - fix folding for these and  update naming
        swirl_s2_test_fname = os.path.join(data_dir, uid + "loop_0_latent_" + str(latent) + "_states_" + str(latent) + "_12345_naturenet_iter2_LL2_test_KF_" + str(0) + ".npz")

        if not os.path.exists(ar_fname) or not os.path.exists(swirl_fname) or not os.path.exists(swirl_nn_fname) or not os.path.exists(swirl_s2_fname):
            print("NO FILE", ar_fname, swirl_fname, swirl_s2_fname, swirl_nn_fname)
            continue
   
        print(swirl_fname)
 
        arhmm = np.load(ar_fname, allow_pickle=True)

        swrl = np.load(swirl_fname, allow_pickle=True)
        swrl_test = np.load(swirl_test_fname, allow_pickle=True)

        swrl_s2 = np.load(swirl_s2_fname, allow_pickle=True)
        swrl_s2_test = np.load(swirl_s2_test_fname, allow_pickle=True)


        swrl_nn = np.load(swirl_nn_fname, allow_pickle=True)
        swrl_nn_test = np.load(swirl_nn_test_fname, allow_pickle=True)
 
        arhmm_ll.append(arhmm["log_likelihood"][-1])

        swirl_ll.append(swrl["LL"])
        swirl_test_ll.append(swrl_test["LL"])

        swirl_s2_ll.append(swrl_s2["LL"])
        swirl_s2_test_ll.append(swrl_s2_test["LL"])

        swirl_nn_ll.append(swrl_nn["LL"])
        swirl_nn_test_ll.append(swrl_nn_test["LL"])

    if len(arhmm_ll) < 1 or len(swirl_ll) < 1 or len(swirl_s2_ll) < 1 or len(swirl_nn_ll) < 1:
        continue
    latents.append(latent)
    print(min(arhmm_ll), max(arhmm_ll), min(swirl_ll), max(swirl_ll),  min(swirl_s2_ll), max(swirl_s2_ll), min(swirl_test_ll), max(swirl_test_ll),  min(swirl_s2_test_ll), max(swirl_s2_test_ll),  min(swirl_nn_ll), max(swirl_nn_ll), min(swirl_nn_test_ll), max(swirl_nn_test_ll))

    ci = stats.t.interval(cl, df=len(arhmm_ll)-1, loc=np.mean(arhmm_ll), scale=np.std(arhmm_ll, ddof=1) / np.sqrt(len(arhmm_ll)))
 
    arhmm_mean.append(np.mean(arhmm_ll))
    arhmm_stds.append(ci)
 
    ci = stats.t.interval(cl, df=len(swirl_ll)-1, loc=np.mean(swirl_ll), scale=np.std(swirl_ll, ddof=1) / np.sqrt(len(swirl_ll)))

    swirl_stds.append(ci)
    swirl_mean.append(np.mean(swirl_ll))

    ci = stats.t.interval(cl, df=len(swirl_test_ll)-1, loc=np.mean(swirl_test_ll), scale=np.std(swirl_test_ll, ddof=1) / np.sqrt(len(swirl_test_ll)))

    swirl_test_mean.append(np.mean(swirl_test_ll))
    swirl_test_stds.append(ci)
    swirl_test_se.append(stats.sem(swirl_test_ll))


    ci = stats.t.interval(cl, df=len(swirl_s2_ll)-1, loc=np.mean(swirl_s2_ll), scale=np.std(swirl_s2_ll, ddof=1) / np.sqrt(len(swirl_s2_ll)))

    swirl_s2_stds.append(ci)
    swirl_s2_mean.append(np.mean(swirl_s2_ll))

    ci = stats.t.interval(cl, df=len(swirl_s2_test_ll)-1, loc=np.mean(swirl_s2_test_ll), scale=np.std(swirl_s2_test_ll, ddof=1) / np.sqrt(len(swirl_s2_test_ll)))

    swirl_s2_test_mean.append(np.mean(swirl_s2_test_ll))
    swirl_s2_test_stds.append(ci)
    swirl_s2_test_se.append(stats.sem(swirl_s2_test_ll))


    ci = stats.t.interval(cl, df=len(swirl_nn_ll)-1, loc=np.mean(swirl_nn_ll), scale=np.std(swirl_nn_ll, ddof=1) / np.sqrt(len(swirl_nn_ll)))

    swirl_nn_stds.append(ci)
    swirl_nn_mean.append(np.mean(swirl_nn_ll))

    ci = stats.t.interval(cl, df=len(swirl_nn_test_ll)-1, loc=np.mean(swirl_nn_test_ll), scale=np.std(swirl_nn_test_ll, ddof=1) / np.sqrt(len(swirl_nn_test_ll)))

    swirl_nn_test_mean.append(np.mean(swirl_nn_test_ll))
    swirl_nn_test_stds.append(ci)
    swirl_nn_test_se.append(stats.sem(swirl_nn_test_ll))


print(arhmm_stds, swirl_stds, swirl_test_stds)

print(swirl_test_mean)
print(swirl_test_se)

ub = []
lb = []
for i in range(len(arhmm_stds)):
    ub.append(arhmm_stds[i][1])
    lb.append(arhmm_stds[i][0])

#plt.plot(latents, arhmm_mean, color='red', label="ARHMM")
#plt.fill_between(latents, lb, ub, color='red', alpha=0.2)

ub = []
lb = []
for i in range(len(swirl_stds)):
    ub.append(swirl_stds[i][1])
    lb.append(swirl_stds[i][0])

plt.plot(latents, swirl_mean, color='blue', label="SWIRL_S1")
plt.fill_between(latents, lb, ub, color='blue', alpha=0.2)

ub = []
lb = []
max_ind = 1
for i in range(len(swirl_test_stds)):
    ub.append(swirl_test_stds[i][1])
    lb.append(swirl_test_stds[i][0])

    if i > 0 and ub[-1] >= lb[0]:
        print(ub[-1], lb[0], swirl_test_mean[i], i)
        max_ind = i


print("MAX IND S1", max_ind)

#t1 = np.arange(1, 50, 1)
#t2 = np.arange(min(lb), -1.5, 0.1)

#plt.plot(t1, [lb[0]]*len(t1), linestyle='--', color='orange')
#plt.plot([max_ind]*len(t2), t2, linestyle='--', color='orange')

plt.plot(latents, swirl_test_mean, color='green', label="SWIRL_S1_TEST")
plt.fill_between(latents, lb, ub, color='green', alpha=0.2)

ub = []
lb = []
for i in range(len(swirl_s2_stds)):
    ub.append(swirl_s2_stds[i][1])
    lb.append(swirl_s2_stds[i][0])

plt.plot(latents, swirl_s2_mean, color='brown', label="SWIRL_S2")
plt.fill_between(latents, lb, ub, color='brown', alpha=0.2)

ub = []
lb = []
max_ind = 1
for i in range(len(swirl_s2_test_stds)):
    ub.append(swirl_s2_test_stds[i][1])
    lb.append(swirl_s2_test_stds[i][0])

    if i > 0 and ub[-1] >= lb[0]:
        print(ub[-1], lb[0], swirl_s2_test_mean[i], i)
        max_ind = i


print("MAX IND S2", max_ind)

plt.plot(latents, swirl_s2_test_mean, color='magenta', label="SWIRL_S2_TEST")
plt.fill_between(latents, lb, ub, color='magenta', alpha=0.2)




ub = []
lb = []
for i in range(len(swirl_nn_stds)):
    ub.append(swirl_nn_stds[i][1])
    lb.append(swirl_nn_stds[i][0])

plt.plot(latents, swirl_nn_mean, color='purple', label="SWIRL_NN_S1")
plt.fill_between(latents, lb, ub, color='purple', alpha=0.2)

ub = []
lb = []
max_ind = 1
for i in range(len(swirl_nn_test_stds)):
    ub.append(swirl_nn_test_stds[i][1])
    lb.append(swirl_nn_test_stds[i][0])

    if i > 0 and ub[-1] >= lb[0]:
        print(ub[-1], lb[0], swirl_nn_test_mean[i], i)
        max_ind = i


print("MAX IND NN S1", max_ind)


plt.plot(latents, swirl_nn_test_mean, color='orange', label="SWIRL_NN_S1_TEST")
plt.fill_between(latents, lb, ub, color='orange', alpha=0.2)



plt.legend()

plt.savefig("LL_NN.png")

