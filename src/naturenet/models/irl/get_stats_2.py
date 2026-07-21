
#Code adapted from https://github.com/BRAINML-GT/SWIRL
from naturenet.models.irl.run_swirl_kfold import run_swirl_init, run_swirl_final
from naturenet.models.irl.swirl_training import *
from naturenet.models.irl.swirl_training_top_level import *
from naturenet.models.irl.swirl_utils import *
import copy
import numpy as np
import numpy.random as npr
from scipy.special import logsumexp
import os
import pickle
import sparse

import sparse
import scipy

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit

#from jax.lib import xla_bridge

import jax.extend

from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

from sit_fuse.utils import read_yaml

import argparse

out_dir = "/data/nlahaye/NatureNet/Blue_Whale_LL_Test/"

def loop_arhmm(yml_conf):


    base_uid = copy.deepcopy(yml_conf["run_uid"])

    yml_conf["out_dir"] = out_dir

    for n_states in range(yml_conf["min_latent"], yml_conf["max_latent"], 1):
        yml_conf["n_hidden"] = n_states
        yml_conf["n_hidden_init"] = n_states
        for i in range(yml_conf["num_runs"]):
            uid = base_uid + "_loop_" + str(i) + "_latent_" + str(n_states) + "_states"
            yml_conf["run_uid"] = uid

            print(uid, yml_conf["run_uid"])

            run_swirl_init(yml_conf)            
            run_swirl_final(yml_conf)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    #if yml_conf["run_arhmm"]:
    loop_arhmm(yml_conf)
    #run_arhmm(yml_conf)
 
    #if yml_conf["run_swirl_init"]:
    #    run_swirl_init(yml_conf)

    #if yml_conf["run_swirl_final"]:
    #    run_swirl_final(yml_conf)

