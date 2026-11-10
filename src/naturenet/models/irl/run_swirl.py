
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
SWIRL final fit (one model on the final split, for maps, time series and reward projection).

    python -m naturenet.models.irl.run_swirl -y full_plan/whales_swirl_final.yaml

Train / test split is by animal: ``holdout_groups`` (list of base ids) or ``holdout_fraction``
(default 0.2, seeded with ``fold_seed``). holdout_fraction: 0 trains on everything and skips the test LLs.

Same pipeline as one run_swirl_kfold fold, so the final fit matches the CV settings:
  * segment_policy: variable is supported (padded rows trimmed for the ARHMM, masked in SWIRL);
  * ARHMM pseudo-count / support from the YAML (arhmm_pseudo_count, arhmm_pseudo_support "T" uses the final T);
  * ARHMM: <run_uid>_<K>_hidden_<seed>_seed_arhmm_s_KF_final.npz; SWIRL init / final files without a fold tag
    (name_fmt "final": <run_uid>_<K>_<seed>_S1_naturenet_iter2_temp_2.npz, ..._S1_naturenet_iter2_LL1.npz, ...),
    which is what plot_swirl_new / swirl_track_series / reward_env_projection read.
Steps follow run_arhmm / run_swirl_init / run_swirl_final in the YAML; existing outputs are skipped unless overwrite.
"""

import argparse
import json
import os

import numpy as np
import numpy.random as npr

from naturenet.models.irl.run_arhmm_kfold import run_arhmm_internal, set_pseudo_count
from naturenet.models.irl.run_swirl_kfold import (
    fit_swirl_final,
    fit_swirl_init,
    fold_trans_prob,
    load_arhmm,
    prepare_data,
)
from naturenet.models.irl.swirl_data import align_data, holdout_split, load_swirl_dataset, step_mask

from sit_fuse.utils import read_yaml

FINAL = "final"   # ARHMM file tag (_KF_final)


def _split(yml_conf, data):
    tr, te, hold = holdout_split(yml_conf, data["groups"])
    os.makedirs(yml_conf["out_dir"], exist_ok=True)
    with open(os.path.join(yml_conf["out_dir"], yml_conf["run_uid"] + "_holdout_groups.json"), "w") as f:
        json.dump(hold, f, indent=1)
    print("Final run:", len(tr), "train /", len(te), "test sequences; held-out animals:", hold)
    return tr, te


def _arhmm_path(yml_conf):
    return os.path.join(yml_conf["out_dir"], yml_conf["run_uid"] + "_" + str(yml_conf["n_hidden_init"]) + "_hidden_" +
                        str(yml_conf["seed"]) + "_seed_arhmm_s_KF_" + FINAL + ".npz")


def run_arhmm(yml_conf, data=None):
    """ARHMM init on the train animals of the final split (variable rows trimmed to their real steps)."""
    if os.path.exists(_arhmm_path(yml_conf)) and not yml_conf.get("overwrite", False):
        print("final ARHMM exists; skipping")
        return
    data = _data(yml_conf, data)
    tr, te = _split(yml_conf, data)
    xs, _, _ = align_data(data)
    if data.get("variable", False):
        n_real = step_mask(data, offset=1).sum(1)
        rows = np.empty(len(xs), dtype=object)
        for i in range(len(xs)):
            rows[i] = xs[i, :n_real[i]]
    else:
        rows = xs
    if yml_conf.get("arhmm_pseudo_support") == "T":
        set_pseudo_count(yml_conf, fold_trans_prob(yml_conf, data, tr))
    else:
        set_pseudo_count(yml_conf)
    if yml_conf.get("kinematics"):
        raise NotImplementedError("final run with a kinematics block: use run_swirl_kfold (the final baselines are "
                                  "action-only)")
    npr.seed(int(yml_conf["seed"]))
    run_arhmm_internal(yml_conf["seed"], yml_conf["emission_dim"], yml_conf["n_hidden_init"],
                       yml_conf["latent_state_dim"], yml_conf["n_states"], rows[tr], yml_conf["out_dir"],
                       yml_conf["run_uid"], FINAL, test_positions=rows[te] if len(te) else None,
                       epochs=int(yml_conf.get("arhmm_epochs", 100)))


def _data(yml_conf, data):
    return prepare_data(yml_conf, data if data is not None else load_swirl_dataset(yml_conf, allow_variable=True))


def _arhmm(yml_conf):
    return load_arhmm(yml_conf["out_dir"], yml_conf["run_uid"], yml_conf["n_hidden_init"], yml_conf["seed"], FINAL)


def run_swirl_init(yml_conf, data=None):
    out = os.path.join(yml_conf["out_dir"], yml_conf["run_uid"] + "_swirl_run_" + str(yml_conf["n_hidden_init"]) +
                       "_" + str(yml_conf["seed"]) + "_naturenet_init1.npz")
    if os.path.exists(out) and not yml_conf.get("overwrite", False):
        print("final SWIRL init exists; skipping")
        return
    data = _data(yml_conf, data)
    tr, _ = _split(yml_conf, data)
    fit_swirl_init(yml_conf, data, fold_trans_prob(yml_conf, data, tr), tr, _arhmm(yml_conf), tag="")


def run_swirl_final(yml_conf, data=None):
    out = os.path.join(yml_conf["out_dir"], yml_conf["run_uid"] + "_" + str(yml_conf["n_hidden"]) + "_" +
                       str(yml_conf["seed"]) + "_S1_naturenet_iter2_LL1.npz")
    if os.path.exists(out) and not yml_conf.get("overwrite", False):
        print("final SWIRL exists; skipping")
        return
    data = _data(yml_conf, data)
    tr, te = _split(yml_conf, data)
    fit_swirl_final(yml_conf, data, fold_trans_prob(yml_conf, data, tr), tr, te, _arhmm(yml_conf), tag="",
                    name_fmt="final")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    yml_conf.setdefault("n_hidden", yml_conf["n_hidden_init"])
    data = prepare_data(yml_conf, load_swirl_dataset(yml_conf, allow_variable=True))

    if yml_conf.get("run_arhmm", True):
        run_arhmm(yml_conf, data)
    if yml_conf.get("run_swirl_init", True):
        run_swirl_init(yml_conf, data)
    if yml_conf.get("run_swirl_final", True):
        run_swirl_final(yml_conf, data)
