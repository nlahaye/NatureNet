
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
SWIRL with the final hyper-parameters.

Train / test split is by animal: ``holdout_groups`` (list of base ids) or ``holdout_fraction``
(default 0.2, seeded with ``fold_seed``). holdout_fraction: 0 trains on everything and skips the test LLs.
(The old "test" set was the last 5 training sequences, i.e. not held out.)
"""
import argparse
import json
import os

import numpy as np

from naturenet.models.irl.run_arhmm import run_arhmm
from naturenet.models.irl.run_swirl_kfold import (
    fit_swirl_final,
    fit_swirl_init,
    fold_trans_prob,
    load_arhmm,
    prepare_data,
)
from naturenet.models.irl.swirl_data import holdout_split, load_swirl_dataset

from sit_fuse.utils import read_yaml


def _split(yml_conf, data):
    tr, te, hold = holdout_split(yml_conf, data["groups"])
    with open(os.path.join(yml_conf["out_dir"], yml_conf["run_uid"] + "_holdout_groups.json"), "w") as f:
        json.dump(hold, f, indent=1)
    print("Final run:", len(tr), "train /", len(te), "test sequences; held-out animals:", hold)
    return tr, te


def run_swirl_init(yml_conf, data=None):
    data = prepare_data(yml_conf, data)
    tr, _ = _split(yml_conf, data)
    T = fold_trans_prob(yml_conf, data, tr)
    arhmm = load_arhmm(yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["n_hidden_init"], yml_conf["seed"])
    fit_swirl_init(yml_conf, data, T, tr, arhmm, tag="")


def run_swirl_final(yml_conf, data=None):
    data = prepare_data(yml_conf, data)
    tr, te = _split(yml_conf, data)
    T = fold_trans_prob(yml_conf, data, tr)
    arhmm = load_arhmm(yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["n_hidden_init"], yml_conf["seed"])
    fit_swirl_final(yml_conf, data, T, tr, te, arhmm, tag="", name_fmt="final")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    data = load_swirl_dataset(yml_conf)

    if yml_conf["run_arhmm"]:
        run_arhmm(yml_conf, data=data)

    if yml_conf["run_swirl_init"] or yml_conf["run_swirl_final"]:
        data = prepare_data(yml_conf, data)
    if yml_conf["run_swirl_init"]:
        run_swirl_init(yml_conf, data)
    if yml_conf["run_swirl_final"]:
        run_swirl_final(yml_conf, data)
