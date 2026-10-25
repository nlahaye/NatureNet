
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
Final NN-SWIRL run (S-2 and/or S-1) on an animal holdout split.

Fixes vs. the original: crashed before training (one_hotx2_partial_nn(n_actions=...) + 2-value unpack);
tested on all[-5:] (in-sample); loaded the linear ARHMM file ('Rs_start' does not exist there); positions /
actions misaligned with the new preprocessing; outputs overwritten across runs. Now: holdout_split()
(holdout_fraction / holdout_groups / fold_seed, as run_swirl.py), NN ARHMM from run_arhmm_holdout(), same
fit as the K-fold scripts, and per-sequence ids saved with every LL / latent-path file for plotting.

Config: run_arhmm (False), run_s2_nn (True), run_s1_nn (True) + the nn_* keys in run_swirl_nn_kfold.py.
"""
import argparse

from naturenet.models.irl.run_arhmm_kfold_nn import run_arhmm_holdout
from naturenet.models.irl.run_swirl_nn_kfold import fit_swirl_nn, load_arhmm_nn, prepare_nn_data
from naturenet.models.irl.swirl_data import holdout_split, load_swirl_dataset

from sit_fuse.utils import read_yaml


def run_swirl_nn(yml_conf, data=None):
    data = prepare_nn_data(yml_conf, data)
    tr, te, hold = holdout_split(yml_conf, data["groups"])
    print("NN-SWIRL final:", len(tr), "train /", len(te), "test sequences; held-out animals:", hold)
    arhmm = load_arhmm_nn(yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["n_hidden_init"], yml_conf["seed"])
    out = {}
    if yml_conf.get("run_s2_nn", True):
        out["s2"] = fit_swirl_nn(yml_conf, data, tr, te, arhmm, "s2", tag="")
    if yml_conf.get("run_s1_nn", True):
        out["s1"] = fit_swirl_nn(yml_conf, data, tr, te, arhmm, "s1", tag="")
    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    data = load_swirl_dataset(yml_conf)
    if yml_conf.get("run_arhmm", False):
        run_arhmm_holdout(yml_conf, data=data)
    run_swirl_nn(yml_conf, data=data)
