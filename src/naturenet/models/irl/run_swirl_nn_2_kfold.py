
#Code adapted from https://github.com/BRAINML-GT/SWIRL
"""
NN-SWIRL S-2 K-fold. Shares everything with run_swirl_nn_kfold.py (mode="s2").

Fixes vs. the original: per-fold ARHMM init (was one '_seed_arhmm_s.npz' that nothing in the K-fold loop
wrote, and that would have been fit on the test animals), no bfloat16, no one_hotx2(n_actions=...) TypeError,
no jnp.roll wrap-around for the previous state at t=0 (x_prev = states[:, :-2]), LLs scored with the S-2
policy, factored S-2 value iteration (no dense S^2 x A x S^2 tensor).
"""
import argparse

from naturenet.models.irl.run_arhmm_kfold_nn import run_arhmm
from naturenet.models.irl.run_swirl_nn_kfold import run_swirl_nn_kfold
from naturenet.models.irl.swirl_data import load_swirl_dataset

from sit_fuse.utils import read_yaml


def run_swirl_nn(yml_conf, data=None, folds_to_run=None):
    """S-2 NN K-fold (name kept for get_stats_3_s2.py)."""
    run_swirl_nn_kfold(yml_conf, "s2", data, folds_to_run)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    parser.add_argument("--fold", type=int, nargs="*", default=None)
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    data = load_swirl_dataset(yml_conf)
    if yml_conf.get("run_arhmm", False):
        run_arhmm(yml_conf, data=data, folds_to_run=args.fold)
    if yml_conf.get("run_swirl_nn", True):
        run_swirl_nn(yml_conf, data=data, folds_to_run=args.fold)
